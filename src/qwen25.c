#include "qwen25.h"

#include <cblas-openblas.h>
#include <cblas.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "config.h"
#include "safetensor.h"

typedef struct Embedding {
	float* table;
	int vocab_size;
	int hidden_dim;
} Embedding;

typedef struct Linear {
	float* weight;
	float* bias;
} Linear;

typedef struct Norm {
	float* weight;
	float eps;
} Norm;

typedef struct Qwen25_05B_Model {
	Embedding embedding;

	struct Layer {
		Norm norm;
		struct {
			Linear q;
			Linear k;
			Linear v;
			Linear o;
		} attention;
		Norm post_norm;
		struct {
			Linear gate;
			Linear up;
			Linear down;
		} mlp;
	} layers[24];
	Norm norm;

	int max_seq;
	float* k_cache;
	float* v_cache;
	int cache_seq_len;

	const ModelConfig* config;
	SafeTensors* sf;
} Qwen25_05B_Model;

float* malloc_fp32(int rows, int cols, int trans, uint16_t* bf16data) {
	float* p = malloc(rows * cols * sizeof(float));
	for (int i = 0; i < rows * cols; i++) {
		uint32_t f32 = (uint32_t)bf16data[i] << 16;
		p[i] = *(float*)&f32;
	}
	if (trans) {
		// then p is [cols, rows]
		float* q = malloc(rows * cols * sizeof(float));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				q[i * cols + j] = p[j * rows + i];
			}
		}
		free(p);
		return q;
	}
	return p;
}

void* Qwen25_05B(ModelConfig* config, const char* model_safetensors) {
	Qwen25_05B_Model* m = calloc(1, sizeof(Qwen25_05B_Model));
	SafeTensors* sf = load_safetensors(model_safetensors);
	m->config = config;
	m->sf = sf;
	char buffer[64];

	const int V = config->vocab_size;
	const int D = config->hidden_size;
	const int Dh = D / config->num_attention_heads;
	const int Hkv = config->num_key_value_heads;
	const int Df = config->intermediate_size;

	m->embedding.table = malloc_fp32(V, D, 0, get_tensor(sf, "model.embed_tokens.weight"));
	m->embedding.vocab_size = V;
	m->embedding.hidden_dim = D;

	for (int l = 0; l < config->num_hidden_layers; l++) {
		struct Layer* layer = &m->layers[l];

		sprintf(buffer, "model.layers.%d.input_layernorm.weight", l);
		layer->norm.weight = malloc_fp32(D, 1, 0, get_tensor(sf, buffer));
		layer->norm.eps = config->rms_norm_eps;

		sprintf(buffer, "model.layers.%d.self_attn.q_proj.weight", l);
		layer->attention.q.weight = malloc_fp32(D, D, 1, get_tensor(sf, buffer));
		sprintf(buffer, "model.layers.%d.self_attn.q_proj.bias", l);
		layer->attention.q.bias = malloc_fp32(D, 1, 0, get_tensor(sf, buffer));

		sprintf(buffer, "model.layers.%d.self_attn.k_proj.weight", l);
		layer->attention.k.weight = malloc_fp32(D, Hkv * Dh, 1, get_tensor(sf, buffer));
		sprintf(buffer, "model.layers.%d.self_attn.k_proj.bias", l);
		layer->attention.k.bias = malloc_fp32(Hkv * Dh, 1, 0, get_tensor(sf, buffer));

		sprintf(buffer, "model.layers.%d.self_attn.v_proj.weight", l);
		layer->attention.v.weight = malloc_fp32(D, Hkv * Dh, 1, get_tensor(sf, buffer));
		sprintf(buffer, "model.layers.%d.self_attn.v_proj.bias", l);
		layer->attention.v.bias = malloc_fp32(Hkv * Dh, 1, 0, get_tensor(sf, buffer));

		sprintf(buffer, "model.layers.%d.self_attn.o_proj.weight", l);
		layer->attention.o.weight = malloc_fp32(D, D, 1, get_tensor(sf, buffer));

		sprintf(buffer, "model.layers.%d.post_attention_layernorm.weight", l);
		layer->post_norm.weight = malloc_fp32(D, 1, 0, get_tensor(sf, buffer));
		layer->post_norm.eps = config->rms_norm_eps;

		sprintf(buffer, "model.layers.%d.mlp.gate_proj.weight", l);
		layer->mlp.gate.weight = malloc_fp32(D, Df, 1, get_tensor(sf, buffer));
		sprintf(buffer, "model.layers.%d.mlp.up_proj.weight", l);
		layer->mlp.up.weight = malloc_fp32(D, Df, 1, get_tensor(sf, buffer));
		sprintf(buffer, "model.layers.%d.mlp.down_proj.weight", l);
		layer->mlp.down.weight = malloc_fp32(Df, D, 1, get_tensor(sf, buffer));
	}

	m->norm.weight = malloc_fp32(D, 1, 0, get_tensor(sf, "model.norm.weight"));
	m->norm.eps = config->rms_norm_eps;

	m->max_seq = 4096;
	int size = config->num_hidden_layers * Hkv * m->max_seq * Dh;
	m->k_cache = malloc(size * sizeof(float));
	m->v_cache = malloc(size * sizeof(float));
	m->cache_seq_len = 0;

	free_safetensors(m->sf);
	return m;
}

void Qwen25_05B_free(Qwen25_05B_Model* model) {
	free(model->embedding.table);
	for (int l = 0; l < model->config->num_hidden_layers; l++) {
		struct Layer* layer = &model->layers[l];
		free(layer->norm.weight);
		free(layer->attention.q.weight);
		free(layer->attention.q.bias);
		free(layer->attention.k.weight);
		free(layer->attention.k.bias);
		free(layer->attention.v.weight);
		free(layer->attention.v.bias);
		free(layer->attention.o.weight);
		free(layer->post_norm.weight);
		free(layer->mlp.gate.weight);
		free(layer->mlp.up.weight);
		free(layer->mlp.down.weight);
	}
	free(model->norm.weight);
	free(model->k_cache);
	free(model->v_cache);
	free(model);
}

// tokens (n,)
// return (n, d)
float* embed_tokens(Embedding embd, const int* tokens, int n) {
	int d = embd.hidden_dim;

	float* embeddings = malloc(n * d * sizeof(float));
	for (int i = 0; i < n; i++) {
		memcpy(embeddings + i * d, embd.table + tokens[i] * d, d * sizeof(float));
	}

	return embeddings;
}

// x = (x / rms(x)) * weight
void rms_norm(float* x, int hidden_dim, Norm norm) {
	float mean_sq = 0.0f;
	for (int i = 0; i < hidden_dim; i++) {
		float v = x[i];
		mean_sq += v * v;
	}
	mean_sq /= hidden_dim;

	float rms = sqrtf(mean_sq + norm.eps);
	for (int j = 0; j < hidden_dim; j++) {
		x[j] = x[j] / rms * norm.weight[j];
	}
}

void rope(float* Q, int current_pos, int len, int num_heads, int head_dim, float rope_theta) {
	int half_dim = head_dim / 2;

	for (int i = 0; i < len; i++) {
		for (int h = 0; h < num_heads; h++) {
			for (int m = 0; m < half_dim; m++) {
				int pos = current_pos + i;
				float angle = pos * powf(rope_theta, -1.0f * m / half_dim);

				int idx = (i * num_heads + h) * head_dim + m;
				float x0 = Q[idx];
				float x1 = Q[idx + half_dim];
				Q[idx] = x0 * cosf(angle) - x1 * sinf(angle);
				Q[idx + half_dim] = x0 * sinf(angle) + x1 * cosf(angle);
			}
		}
	}
}

// [d0, d1, d2] -> [d1, d0, d2]
void transpose(float* in, float* out, int d0, int d1, int d2) {
	for (int i = 0; i < d0; i++) {
		for (int j = 0; j < d1; j++) {
			for (int k = 0; k < d2; k++) {
				// out[j][i][k] = in[i][j][k]
				int idx_in = ((i * d1) + j) * d2 + k;
				int idx_out = ((j * d0) + i) * d2 + k;
				out[idx_out] = in[idx_in];
			}
		}
	}
}

void softmax(float* x, int n) {
	float max_val = x[0];
	for (int i = 1; i < n; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	for (int i = 0; i < n; i++) {
		x[i] /= sum;
	}
}

float silu(float x) { return x / (1.0f + expf(-x)); }

static void debugpx(float* x, int d0, int d1) {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%12.8f, ", x[i * d1 + j]);
		}
		printf("...\n");
	}
}

static void debugpx3(float* x, int d0, int d1, int d2) {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%8.4f, ", x[i * d2 + j]);
		}
		printf("...\n");
	}
}

int Qwen25_05B_inference(Qwen25_05B_Model* model, const int* tokens, int seq_len) {
	const int N = seq_len;
	const int D = model->config->hidden_size;
	const int Hq = model->config->num_attention_heads;
	const int Hkv = model->config->num_key_value_heads;
	const int Dh = D / Hq;

	const int Df = model->config->intermediate_size;
	const float v = model->config->vocab_size;

	float* X = embed_tokens(model->embedding, tokens, N);
	float* R = malloc(N * D * sizeof(float));

	const int Nmax = model->max_seq;
	const int dN = seq_len - model->cache_seq_len;	// just N or 1
	float* X1 = X + (N - dN) * D;
	float* Q = malloc(dN * Hq * Dh * sizeof(float));   // [dN, h, hd]
	float* K = malloc(dN * Hkv * Dh * sizeof(float));  // [dN, kvh, hd]
	float* V = malloc(dN * Hkv * Dh * sizeof(float));  // [dN, kvh, hd]
	float* T = malloc(N * D * sizeof(float));	   // [n, d]
	float* A = malloc(dN * N * sizeof(float));	   // [dn, n]

	float* O = malloc(dN * D * sizeof(float));   // [h, dn, hd]
	float* G = malloc(dN * Df * sizeof(float));  // [dn, d_ff]
	float* U = malloc(dN * Df * sizeof(float));  // [dn, d_ff]
	float* logits = malloc(v * sizeof(float));

	for (int l = 0; l < model->config->num_hidden_layers; l++) {
		float* W_Q = model->layers[l].attention.q.weight;  // [d, d]
		float* W_K = model->layers[l].attention.k.weight;  // [d, d]
		float* W_V = model->layers[l].attention.v.weight;  // [d, d]
		float* W_O = model->layers[l].attention.o.weight;  // [d, d]
		float* W_G = model->layers[l].mlp.gate.weight;	   // [d, df]
		float* W_U = model->layers[l].mlp.up.weight;	   // [d, df]
		float* W_D = model->layers[l].mlp.down.weight;	   // [df, d]

		// 1. input norm
		memcpy(R, X1, dN * D * sizeof(float));
		for (int i = 0; i < dN; i++) {
			rms_norm(X1 + i * D, D, model->layers[l].norm);
		}

		// 2. self attention
		// 2.1 project
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, D, D, 1.0f, X1, D, W_Q, D, 0.0f, Q, D);
		for (int i = 0; i < dN * D; i++) {
			Q[i] += model->layers[l].attention.q.bias[i % D];
		}
		const int kvd = Hkv * Dh;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, kvd, D, 1.0f, X1, D, W_K, kvd, 0.0f, K, kvd);
		for (int i = 0; i < dN * kvd; i++) {
			K[i] += model->layers[l].attention.k.bias[i % kvd];
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, kvd, D, 1.0f, X1, D, W_V, kvd, 0.0f, V, kvd);
		for (int i = 0; i < dN * kvd; i++) {
			V[i] += model->layers[l].attention.v.bias[i % kvd];
		}

		// 2.2 RoPE
		rope(Q, N - dN, dN, Hq, Dh, model->config->rope_theta);
		rope(K, N - dN, dN, Hkv, Dh, model->config->rope_theta);

		// 2.3 Attention
		// [n, d] -> [n, h, dk] -> [h, n, dk]
		transpose(Q, T, dN, Hq, Dh);
		memcpy(Q, T, dN * D * sizeof(float));
		transpose(K, T, dN, Hkv, Dh);
		memcpy(K, T, dN * kvd * sizeof(float));
		transpose(V, T, dN, Hkv, Dh);
		memcpy(V, T, dN * kvd * sizeof(float));
		// TODO: int stride = 0l;

		// append to cache
		float* k_cache = model->k_cache + l * Hkv * Nmax * Dh;	// [Hkv, Nmax, Dh], while K: [Hkv, dN, Dh]
		float* v_cache = model->v_cache + l * Hkv * Nmax * Dh;
		for (int i = 0; i < dN; i++) {
			int p = N - dN + i;
			for (int h = 0; h < Hkv; h++) {
				memcpy(k_cache + h * Nmax * Dh + p * Dh, K + h * dN * Dh + i * Dh, Dh * sizeof(float));
				memcpy(v_cache + h * Nmax * Dh + p * Dh, V + h * dN * Dh + i * Dh, Dh * sizeof(float));
			}
		}

		for (int head = 0; head < Hq; head++) {
			int kvh = head / (Hq / Hkv);
			float* Kh = k_cache + kvh * Nmax * Dh;	// [n, hd]
			float* Vh = v_cache + kvh * Nmax * Dh;	// [n, hd]
			float* Qh = Q + (head)*dN * Dh;		// [dn, hd]
			float* Oh = O + (head)*dN * Dh;		// [dn, hd]

			const float a = 1.0f / sqrtf((float)Dh);
			// QK^T/sqrt(d_k)
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dN, N, Dh, a, Qh, Dh, Kh, Dh, 0.0f, A, N);
			// ...+M
			for (int i = 0; i < dN; i++) {
				for (int j = N - dN + i + 1; j < N; j++) {
					A[i * N + j] = -1e4f;
				}
			}
			// Softmax(...)
			for (int i = 0; i < dN; i++) {
				softmax(A + i * N, N);
			}
			// O = AV
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, Dh, N, 1.f, A, N, Vh, Dh, 0, Oh, Dh);
		}

		// concat multi head
		// O: [h, dn, dk] -> [dn, h, dk] -> [dn, d]
		transpose(O, T, Hq, dN, Dh);
		memcpy(O, T, dN * D * sizeof(float));

		// 2.4 Residual
		// X_mid = X + OW_O
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, D, D, 1.0f, O, D, W_O, D, 0.0f, X1, D);
		for (int i = 0; i < dN * D; i++) {
			X1[i] += R[i];
		}
		memcpy(R, X1, dN * D * sizeof(float));

		// correct

		// 3. FFN
		for (int i = 0; i < dN; i++) {
			rms_norm(X1 + i * D, D, model->layers[l].post_norm);
		}
		// (SiLU(G) * (U)) x D
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, Df, D, 1.0f, X1, D, W_G, Df, 0.0f, G, Df);
		for (int i = 0; i < dN * Df; i++) {
			G[i] = silu(G[i]);
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, Df, D, 1.0f, X1, D, W_U, Df, 0.0f, U, Df);
		for (int i = 0; i < dN * Df; i++) {
			U[i] = G[i] * U[i];
		}
		// Residual
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dN, D, Df, 1.0f, U, Df, W_D, D, 1.0f, R, D);
		memcpy(X1, R, dN * D * sizeof(float));
	}
	model->cache_seq_len += dN;

	for (int i = 0; i < dN; i++) {
		rms_norm(X1 + i * D, D, model->norm);
	}
	float* y_n = X1 + (dN - 1) * D;
	float* W_E = model->embedding.table;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, v, D, 1.0f, y_n, D, W_E, D, 0.0f, logits, v);

	softmax(logits, v);
	float max_prob = logits[0];
	int i_max = 0;
	for (int i = 0; i < v; i++) {
		if (logits[i] > max_prob) {
			max_prob = logits[i];
			i_max = i;
		}
	}

	free(X);
	free(R);
	free(Q);
	free(K);
	free(V);
	free(T);
	free(A);
	free(O);
	free(G);
	free(U);
	free(logits);
	return i_max;
}
