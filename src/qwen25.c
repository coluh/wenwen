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
	Model* m = calloc(1, sizeof(Model));
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

	m->max_seq = 512;  // TODO: config this!
	int size = config->num_hidden_layers * Hkv * m->max_seq * Dh;
	m->k_cache = malloc(size * sizeof(float));
	m->v_cache = malloc(size * sizeof(float));
	m->cache_seq_len = 0;

	free_safetensors(m->sf);
	return m;
}

void Qwen25_05B_free(Model* model) {
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
void rms_norm(float* x, int hidden_dim, RMSNorm norm) {
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

void softmax(float* x, int n, float* y) {
	float max_val = x[0];
	for (int i = 1; i < n; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		y[i] = expf(x[i] - max_val);
		sum += y[i];
	}
	for (int i = 0; i < n; i++) {
		y[i] /= sum;
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

void rmsnorm1(RMSNorm* n, float* X, int B, int S, int D) {
	n->ctx.dim = D;
	for (int i = 0; i < B * S; i++) {
		float* Xi = X + i * D;		      // [D,]
		float* Xn = n->ctx.X_normed + i * D;  // [D,]

		float sq_sum = cblas_sdot(D, Xi, 1, Xi, 1);
		float mean_sq = sq_sum / D;
		float rms = sqrtf(mean_sq + n->eps);
		n->ctx.rms[i] = rms;
		float rrms = 1.0f / rms;
		for (int d = 0; d < D; d++) {
			Xn[d] = Xi[d] * rrms;
			Xi[d] = Xn[d] * n->weight[d];
		}
	}
}

void rope1(struct RoPE* rope, float* T, int B, int S, int H, int Dh, bool reverse) {
	int half_dim = Dh / 2;
	for (int b = 0; b < B; b++) {  // not used
		for (int s = 0; s < S; s++) {
			for (int h = 0; h < H; h++) {  // not used
				for (int d = 0; d < half_dim; d++) {
					int idx = ((b * S + s) * H + h) * Dh + d;
					float x0 = T[idx];
					float x1 = T[idx + 1];
					float cosv = rope->cosv[s * half_dim + d];
					float sinv = rope->sinv[s * half_dim + d];
					if (reverse) sinv = -sinv;
					T[idx] = x0 * cosv - x1 * sinv;
					T[idx + 1] = x0 * sinv + x1 * cosv;
				}
			}
		}
	}
}

ModelRunner* new_modelrunner(Model* model, int batch_size, int max_seq_len) {
	ModelRunner* mr = malloc(sizeof(ModelRunner));

	mr->B = batch_size;
	mr->S = max_seq_len;
	mr->V = model->config->vocab_size;
	mr->L = model->config->num_hidden_layers;
	mr->D = model->config->hidden_size;
	mr->Hq = model->config->num_attention_heads;
	mr->Hkv = model->config->num_key_value_heads;
	mr->Df = model->config->intermediate_size;
	mr->Dh = mr->D / mr->Hq;
	mr->Dq = mr->D;
	mr->Dkv = mr->Hkv * mr->Dh;

	int B = mr->B, S = mr->S, D = mr->D, Dkv = mr->Dkv, V = mr->V, Df = mr->Df;

	// middle variables for backward
	for (int l = 0; l < mr->L; l++) {
		struct Layer* layer = &model->layers[l];
		layer->norm.ctx.X_normed = malloc(B * S * D * sizeof(float));
		layer->norm.ctx.rms = malloc(B * S * sizeof(float));
		layer->attention.ctx.X_in = malloc(B * S * D * sizeof(float));
		layer->attention.ctx.Q = malloc(B * S * D * sizeof(float));
		layer->attention.ctx.K = malloc(B * S * Dkv * sizeof(float));
		layer->attention.ctx.V = malloc(B * S * Dkv * sizeof(float));
		layer->attention.ctx.A = malloc(B * S * S * sizeof(float));
		layer->attention.ctx.P = malloc(B * S * S * sizeof(float));
		layer->attention.ctx.O = malloc(B * S * D * sizeof(float));
		layer->post_norm.ctx.X_normed = malloc(B * S * D * sizeof(float));
		layer->post_norm.ctx.rms = malloc(B * S * sizeof(float));
		layer->mlp.ctx.X_in = malloc(B * S * D * sizeof(float));
		layer->mlp.ctx.G = malloc(B * S * Df * sizeof(float));
		layer->mlp.ctx.G_act = malloc(B * S * Df * sizeof(float));
		layer->mlp.ctx.U = malloc(B * S * Df * sizeof(float));
		layer->mlp.ctx.H = malloc(B * S * Df * sizeof(float));
	}
	model->norm.ctx.X_normed = malloc(B * S * D * sizeof(float));
	model->norm.ctx.rms = malloc(B * S * sizeof(float));
	model->X_final = malloc(B * S * D * sizeof(float));

	const float theta = model->config->rope_theta;
	int half_dim = mr->Dh / 2;
	mr->rope.freqs = malloc(half_dim * sizeof(float));
	mr->rope.cosv = malloc(mr->S * half_dim * sizeof(float));
	mr->rope.sinv = malloc(mr->S * half_dim * sizeof(float));
	for (int i = 0; i < half_dim; i++) {
		mr->rope.freqs[i] = 1.0f / powf(theta, (float)i / half_dim);
	}
	for (int p = 0; p < mr->S; p++) {
		for (int i = 0; i < half_dim; i++) {
			float angle = p * mr->rope.freqs[i];
			mr->rope.cosv[p * half_dim + i] = cosf(angle);
			mr->rope.sinv[p * half_dim + i] = sinf(angle);
		}
	}

	mr->grad.embed = malloc(V * D * sizeof(float));
	mr->grad.layer = malloc(mr->L * sizeof(struct LayerGrad));
	for (int l = 0; l < mr->L; l++) {
		mr->grad.layer[l].norm = malloc(D * sizeof(float));
		mr->grad.layer[l].Wq = malloc(D * D * sizeof(float));
		mr->grad.layer[l].Wk = malloc(D * Dkv * sizeof(float));
		mr->grad.layer[l].Wv = malloc(D * Dkv * sizeof(float));
		mr->grad.layer[l].Wo = malloc(D * D * sizeof(float));
		mr->grad.layer[l].post_norm = malloc(D * sizeof(float));
		mr->grad.layer[l].gate = malloc(D * Df * sizeof(float));
		mr->grad.layer[l].up = malloc(D * Df * sizeof(float));
		mr->grad.layer[l].down = malloc(Df * D * sizeof(float));
	}
	mr->grad.norm = malloc(D * sizeof(float));

	return mr;
}

void free_modelrunner(ModelRunner* mr) {
	free(mr->rope.freqs);
	free(mr->rope.cosv);
	free(mr->rope.sinv);
	for (int l = 0; l < mr->L; l++) {
		struct Layer* layer = &mr->model->layers[l];
		free(layer->norm.ctx.X_normed);
		free(layer->norm.ctx.rms);
		free(layer->attention.ctx.X_in);
		free(layer->attention.ctx.Q);
		free(layer->attention.ctx.K);
		free(layer->attention.ctx.V);
		free(layer->attention.ctx.A);
		free(layer->attention.ctx.P);
		free(layer->attention.ctx.O);
		free(layer->post_norm.ctx.X_normed);
		free(layer->post_norm.ctx.rms);
		free(layer->mlp.ctx.X_in);
		free(layer->mlp.ctx.G);
		free(layer->mlp.ctx.G_act);
		free(layer->mlp.ctx.U);
		free(layer->mlp.ctx.H);
	}
	free(mr->model->norm.ctx.X_normed);
	free(mr->model->norm.ctx.rms);
	free(mr->model->X_final);
	free(mr->grad.embed);
	for (int l = 0; l < mr->L; l++) {
		free(mr->grad.layer[l].norm);
		free(mr->grad.layer[l].Wq);
		free(mr->grad.layer[l].Wk);
		free(mr->grad.layer[l].Wv);
		free(mr->grad.layer[l].Wo);
		free(mr->grad.layer[l].post_norm);
		free(mr->grad.layer[l].gate);
		free(mr->grad.layer[l].up);
		free(mr->grad.layer[l].down);
	}
	free(mr->grad.layer);
	free(mr->grad.norm);
}

void zero_grad(ModelRunner* mr) {
	int B = mr->B, S = mr->S, D = mr->D, Dkv = mr->Dkv, V = mr->V, L = mr->L, Df = mr->Df;
	memset(mr->grad.embed, 0, V * D * sizeof(float));
	for (int l = 0; l < L; l++) {
		struct LayerGrad* layer = &mr->grad.layer[l];
		memset(layer->norm, 0, D * sizeof(float));
		memset(layer->Wq, 0, D * D * sizeof(float));
		memset(layer->Wk, 0, D * Dkv * sizeof(float));
		memset(layer->Wv, 0, D * Dkv * sizeof(float));
		memset(layer->Wo, 0, D * D * sizeof(float));
		memset(layer->post_norm, 0, D * sizeof(float));
		memset(layer->gate, 0, D * Df * sizeof(float));
		memset(layer->up, 0, D * Df * sizeof(float));
		memset(layer->down, 0, Df * D * sizeof(float));
	}
	memset(mr->grad.norm, 0, D * sizeof(float));
}

float* model_forward(ModelRunner* mr, const int* inputs, bool grad) {
	const int B = mr->B;
	const int S = mr->S;
	const int v = mr->V;
	const int D = mr->D;
	const int Hq = mr->Hq;
	const int Hkv = mr->Hkv;
	const int Dh = mr->Dh;
	const int Dq = mr->D;
	const int Dkv = mr->Dkv;
	const int Df = mr->Df;

	int* eos_pos = malloc(B * sizeof(int));
	for (int b = 0; b < B; b++) {
		eos_pos[b] = -1;
		for (int i = 1; i < S; i++) {
			if (inputs[b * S + i] == 151643) {  // TODO: config
				eos_pos[b] = i;
				break;
			}
		}
	}

	float* X = malloc(B * S * D * sizeof(float));
	// float* Q = malloc(B * S * D * sizeof(float));	      // [B, S, D], or say [B, S, Hq, Dh]
	// float* K = malloc(B * S * Hkv * Dh * sizeof(float));  // [B, S, Hkv, Dh]
	// float* V = malloc(B * S * Hkv * Dh * sizeof(float));  // [B, S, Hkv, Dh]
	// float* A = malloc(B * S * S * sizeof(float));	      // [B, S, S], seperate A per batch
	// float* O = malloc(B * S * D * sizeof(float));	      // [B, S, Hq, Dh]
	// float* G = malloc(B * S * Df * sizeof(float));
	// float* U = malloc(B * S * Df * sizeof(float));
	float* logits = malloc(B * S * v * sizeof(float));

	// embed tokens
	// inputs: [B, S]
	// embedding: [V, D]
	// X: [B, S, D] = [embedding[v] for v in x.flatten()].reshape(B, S, -1)
	for (int b = 0; b < B; b++) {
		for (int s = 0; s < S; s++) {
			int token = inputs[b * S + s];
			float* v = mr->model->embedding.table + token * D;
			memcpy(X + (b * S + s) * D, v, D * sizeof(float));
		}
	}

	for (int l = 0; l < mr->model->config->num_hidden_layers; l++) {
		float* X_att = mr->model->layers[l].attention.ctx.X_in;
		float* Q = mr->model->layers[l].attention.ctx.Q;
		float* K = mr->model->layers[l].attention.ctx.K;
		float* V = mr->model->layers[l].attention.ctx.V;
		float* A = mr->model->layers[l].attention.ctx.A;
		float* P = mr->model->layers[l].attention.ctx.P;
		float* O = mr->model->layers[l].attention.ctx.O;
		float* X_ffn = mr->model->layers[l].mlp.ctx.X_in;
		float* G = mr->model->layers[l].mlp.ctx.G;
		float* G_act = mr->model->layers[l].mlp.ctx.U;
		float* U = mr->model->layers[l].mlp.ctx.U;
		float* H = mr->model->layers[l].mlp.ctx.U;
		float* Wq = mr->model->layers[l].attention.q.weight;  // [D, D] = [D, Hq*Dh]
		float* Wk = mr->model->layers[l].attention.k.weight;  // [D, Hkv*Dh]
		float* Wv = mr->model->layers[l].attention.v.weight;  // [D, Hkv*Dh]
		float* Wo = mr->model->layers[l].attention.o.weight;  // [D, D]
		float* Wg = mr->model->layers[l].mlp.gate.weight;     // [D, Df]
		float* Wu = mr->model->layers[l].mlp.up.weight;	      // [D, Df]
		float* Wd = mr->model->layers[l].mlp.down.weight;     // [Df, D]

		// save for residual
		memcpy(X_att, X, B * S * D * sizeof(float));

		// RMSNorm on last dimension
		rmsnorm1(&mr->model->layers[l].norm, X, B, S, D);

		// X: [B, S, D]
		// get Q: [B, S, D], K, V: [B, S, Hkv*Dh]
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, Dq, D, 1.0f, X, D, Wq, Dq, 0.0f, Q, Dq);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, Dkv, D, 1.0f, X, D, Wk, Dkv, 0.0f, K,
			    Dkv);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, Dkv, D, 1.0f, X, D, Wv, Dkv, 0.0f, V,
			    Dkv);

		// RoPE Q, K
		rope1(&mr->rope, Q, B, S, Hq, Dh, false);
		rope1(&mr->rope, K, B, S, Hkv, Dh, false);

		// get attention scores
		// avoid transpose
		for (int b = 0; b < B; b++) {
			float* Qb = Q + b * S * Hq * Dh;   // [S, Hq, Dh]
			float* Kb = K + b * S * Hkv * Dh;  // [S, Hkv, Dh]
			float* Vb = V + b * S * Hkv * Dh;  // [S, Hkv, Dh]
			float* Ab = A + b * S * S;	   // [S, S]
			float* Pb = P + b * S * S;	   // [S, S]
			float* Ob = O + b * S * Hq * Dh;   // [S, Hq, Dh]
			for (int h = 0; h < Hq; h++) {
				int hq = h;
				int hkv = h * Hkv / Hq;

				// Qh:    [S, Dh] = Qb[:, hq, :]
				// Kh/Vh: [S, Dh] = Kb/Vb[:, hkv, :]
				// Oh:    [S, Dh] = Ob[:, hq, :]
				float* Qh = Qb + hq * Dh;   // start of Qh, lasting Dh, skip Hq*Dh
							    // lasting Dh, skip Hq*Dh...
				float* Kh = Kb + hkv * Dh;  // stride = Hkv*Dh
				float* Vh = Vb + hkv * Dh;
				float* Ah = Ab;
				float* Ph = Pb;
				float* Oh = Ob + hq * Dh;

				// Ah = softmax(Q @ K^T / sqrt(Dh) + M)
				// 	[S, Dh] @ [Dh, S]
				float a = 1.0f / sqrtf(Dh);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, S, Dh, a, Qh, Hq * Dh, Kh,
					    Hkv * Dh, 0.0f, Ah, S);
				// attention mask
				if (eos_pos[b] >= 0) {
					for (int i = 0; i < S; i++) {
						for (int j = eos_pos[b] + 1; j < S; j++) {
							Ah[i * S + j] = -1e9f;
						}
					}
				}
				for (int i = 0; i < S; i++) {
					// causal mask
					for (int j = i + 1; j < S; j++) {
						Ah[i * S + j] = -1e9f;
					}
					// softmax
					softmax(Ah + i * S, S, Ph + i * S);
				}

				// Oh = Ah @ Vh
				// 	[S, S] @ [S, Dh]
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, Dh, S, 1.0f, Ph, S, Vh,
					    Hkv * Dh, 0.0f, Oh, Hq * Dh);
			}
		}

		// residual
		// X = R + O @ Wo
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, D, D, 1.0f, O, D, Wo, D, 0.0f, X, D);
		for (int i = 0; i < B * S * D; i++) {
			X[i] += X_att[i];
		}

		// save for residual
		memcpy(X_ffn, X, B * S * D * sizeof(float));

		// FFN
		rmsnorm1(&mr->model->layers[l].post_norm, X, B, S, D);
		// (SiLU(G) * (U)) x D
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, Df, D, 1.0f, X, D, Wg, Df, 0.0f, G, Df);
		for (int i = 0; i < B * S * Df; i++) {
			G_act[i] = silu(G[i]);
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, Df, D, 1.0f, X, D, Wu, Df, 0.0f, U, Df);
		for (int i = 0; i < B * S * Df; i++) {
			H[i] = G_act[i] * U[i];
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * S, D, Df, 1.0f, H, Df, Wd, D, 0.0f, X, D);
		for (int i = 0; i < B * S * D; i++) {
			X[i] += X_ffn[i];
		}
	}

	rmsnorm1(&mr->model->norm, X, B, S, D);
	memcpy(mr->model->X_final, X, B * S * D * sizeof(float));

	// X: [B, S, D], E: [V, D]
	// -> logits: [B, S, V]
	const float* We = mr->model->embedding.table;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B * S, v, D, 1.0f, X, D, We, D, 0.0f, logits, v);

	free(eos_pos);
	free(X);
	return logits;
}

void rmsnorm_backward(RMSNorm* n, float* dy, float* X_normed, float* out_grad, float* out_dx) {
	const int D = n->ctx.dim;
	for (int j = 0; j < D; j++) {
		out_grad[j] += dy[j] * X_normed[j];
	}

	float* w = n->weight;
	float rms = n->ctx.rms;
	float dim = n->ctx.dim;
	float sum = 0;
	for (int j = 0; j < D; j++) {
		sum += dy[j] * w[j] * X_normed[j];
	}
	float* dx = dy;
	for (int j = 0; j < D; j++) {
		dx[j] = (1 / rms) * (dy[j] * w[j] - X_normed[j] / dim * sum);
	}
}

void ffn_backward(float* dY, struct FFN* ffn, float* out_dWg, float* out_dWu, float* out_dWd, float* out_dX, int N,
		  int D, int Df) {
	// TODO: residual

	// Y = (SiLU(X @ Wg) x (X @ Wu)) @ Wd
	// let H be (SiLU(X @ Wg) x (X @ Wu))
	float* dH = malloc(N * Df * sizeof(float));

	// Y = H @ Wd => dWd = H^T @ dY, dH = dY @ Wd^T
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Df, D, N, 1.0f, ffn->ctx.H, Df, dY, D, 0.0f, out_dWd, D);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, Df, D, 1.0f, dY, D, ffn->down.weight, D, 0.0f, dH, Df);

	// let G_act = SiLU(G)
	// H = G_act x U => dG_act = dH x U, dU = dH x G_act
	float* dG_act = out_dWg;
	float* dU = out_dWu;
	float* dG = out_dWg;
	for (int i = 0; i < N; i++) {
		dG_act[i] = dH[i] * ffn->ctx.U[i];  // [D, Df]
		dU[i] = dH[i] * ffn->ctx.G_act[i];

		// SiLU(x) = x / (1 + e^(-x)) => dSiLU/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
		// y = SiLU(x) => dx = dy * dSiLU
		float g = ffn->ctx.G[i];
		float sig = 1.0f / (1.0f + expf(-g));
		float dsilu = sig * (1 + g * (1 - sig));
		dG[i] = dG_act[i] * dsilu;
	}

	// G = X @ Wg => dWg = X^T @ dG, dX = dG @ Wg^T
	// U = X @ Wu => dWu = X^T @ dU, dX = dU @ Wu^T
	float* X = ffn->ctx.X_in;
	float* Wg = ffn->gate.weight;
	float* Wu = ffn->up.weight;
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, N, Df, 1.0f, X, N, dG, Df, 0.0f, out_dWg, Df);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, Df, 1.0f, dG, Df, Wg, Df, 1.0f, out_dX, D);
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, N, Df, 1.0f, X, N, dU, Df, 0.0f, out_dWu, Df);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, Df, 1.0f, dU, Df, Wu, Df, 1.0f, out_dX, D);
}

void attention_backward(struct Attention* att, float* dY, float* out_dWq, float* out_dWk, float* out_dWv,
			float* out_dWo, float* out_dX, struct RoPE* rope) {
	const int B = att->ctx.B, S = att->ctx.S, Hq = att->ctx.Hq, Hkv = att->ctx.Hkv, Dh = att->ctx.Dh;
	const int N = B * S, D = Hq * Dh, Dkv = Hkv * Dh;

	// let S = (Q @ K^T) / d, P = softmax(S), O = PV, Y = OWo
	// Y = O @ Wo => dWo = O^T @ dY, dO = dY @ Wo^T

	float* O = att->ctx.O;
	float* Wo = att->o.weight;
	float* dO = malloc(B * S * D * sizeof(float));
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0f, O, D, dY, D, 0.0f, out_dWo, D);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0f, dY, D, Wo, D, 0.0f, dO, D);
	float* dV = calloc(B * S * Hkv * Dh, sizeof(float));
	float* dQ = calloc(B * S * Hq * Dh, sizeof(float));
	float* dK = calloc(B * S * Hkv * Dh, sizeof(float));

	for (int b = 0; b < B; b++) {
		for (int h = 0; h < Hq; h++) {
			int hq = h, hkv = h * Hkv / Hq;

			// O = P @ V => dV = P^T @ dOh, dP = dOh @ V^T
			float* P = att->ctx.P + ((b * Hq + h) * S * S);		// [S, S]
			float* V = att->ctx.V + (b * S * Hkv * Dh + hkv * Dh);	// [S, Dh], ld=Hkv*Dh
			float* dOh = dO + (b * S * Hq * Dh + hq * Dh);		// [S, Dh], ld=Hq*Dh
			float* dVh = dV + (b * S * Hkv * Dh + hkv * Dh);	// [S, Dh], ld=Hkv*Dh
			float* dP = malloc(S * S * sizeof(float));		// [S, S]
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, S, Dh, S, 1.0f, P, S, dOh, Hq * Dh, 0.0f,
				    dVh, Hkv * Dh);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, S, Dh, 1.0f, dOh, Hq * Dh, V, Hkv * Dh,
				    0.0f, dP, S);

			// P = softmax(A) => dA = P x (dP - sum), while sum = sigma(P*dP)
			float* dA = dP;	 // reuse memory
			for (int s = 0; s < S; s++) {
				float sum = 0.0f;
				for (int i = 0; i < S; i++) {
					sum += P[s * S + i] * dP[s * S + i];
				}
				for (int i = 0; i < S; i++) {
					dA[s * S + i] = P[s * S + i] * (dP[s * S + i] - sum);
				}
			}

			// A = Q @ K^T => dQ = dA @ K, dK = dA^T @ Q
			float a = 1.0f / sqrtf(Dh);
			float* Q = att->ctx.Q + b * S * Hq * Dh + hq * Dh;    // [S, Dh], ld=Hq*Dh
			float* K = att->ctx.K + b * S * Hkv * Dh + hkv * Dh;  // [S, Dh], ld=Hkv*Dh
			float* dQh = dQ + b * S * Hq * Dh + hq * Dh;	      // [S, Dh], ld=Hq*Dh
			float* dKh = dK + b * S * Hkv * Dh + hkv * Dh;	      // [S, Dh], ld=Hkv*Dh
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, Dh, S, 1.0f, dA, S, K, Hkv * Dh, 0.0f,
				    dQh, Hq * Dh);
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, S, Dh, S, 1.0f, dA, S, Q, Hq * Dh, 0.0f,
				    dKh, Hkv * Dh);
		}
	}

	rope1(rope, dQ, B, S, Hq, Dh, true);
	rope1(rope, dK, B, S, Hkv, Dh, true);

	// Wq...dX
	float* X = att->ctx.X_in;
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0f, X, D, dQ, D, 0.0f, out_dWq, D);
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, Dkv, N, 1.0f, X, D, dK, Dkv, 0.0f, out_dWk, Dkv);
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, Dkv, N, 1.0f, X, D, dV, Dkv, 0.0f, out_dWv, Dkv);
	float *Wq = att->q.weight, *Wk = att->k.weight, *Wv = att->v.weight;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0f, dQ, D, Wq, D, 1.0f, out_dX, D);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, Dkv, 1.0f, dK, Dkv, Wk, Dkv, 1.0f, out_dX, D);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, Dkv, 1.0f, dV, Dkv, Wv, Dkv, 1.0f, out_dX, D);
}

void model_backward(ModelRunner* mr, const int* inputs, float* dlogits) {
	Model* m = mr->model;
	int B = mr->B, S = mr->S, D = mr->D, V = mr->V, L = mr->L, Df = mr->Df;
	int BS = B * S;
	float* dX = malloc(B * S * D * sizeof(float));

	// Y = X @ W_E^T
	//   => dW_E^T = X^T @ dY
	//   => dW_E = (X^T @ dY)^T = dY^T @ X
	//   => dWe = dlogits^T @ X
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, V, D, BS, 1.0f, dlogits, V, mr->model->X_final, D, 1.0f,
		    mr->grad.embed, D);
	// dX = dY @ W_E
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BS, D, V, 1.0f, dlogits, V, m->embedding.table, D, 0, dX,
		    D);

	// RMSNorm
	// y = x / rms * w, x_hat = x / rms
	//   => dw = dy * x_hat
	//      dx = (1/rms) * (dy * w - x_hat / d * sum(dy * w * x_hat))
	for (int i = 0; i < B * S; i++) {
		float* Xn = mr->model->norm.ctx.X_normed + i * D;
		rmsnorm_backward(&mr->model->norm, dX + i * D, Xn, mr->grad.norm, dX + i * D);
	}

	for (int l = L - 1; l >= 0; l--) {
		struct Layer* layer = &mr->model->layers[l];
		struct LayerGrad* grad = &mr->grad.layer[l];

		struct FFN* ffn = &mr->model->layers[l].mlp;
		float* dX_res = malloc(BS * D * sizeof(float));
		memcpy(dX_res, dX, B * S * D * sizeof(float));
		ffn_backward(dX, ffn, grad->gate, grad->up, grad->down, dX, BS, D, Df);
		for (int i = 0; i < BS; i++) {
			float* Xn = layer->post_norm.ctx.X_normed + i * D;
			rmsnorm_backward(&layer->post_norm, dX + i * D, Xn, grad->post_norm, dX + i * D);
		}
		for (int i = 0; i < B * S * D; i++) {
			dX[i] += dX_res[i];
		}

		memcpy(dX_res, dX, B * S * D * sizeof(float));
		attention_backward(&layer->attention, dX, grad->Wq, grad->Wk, grad->Wv, grad->Wo, dX, &mr->rope);
		for (int i = 0; i < BS; i++) {
			float* Xn = layer->norm.ctx.X_normed + i * D;
			rmsnorm_backward(&layer->norm, dX + i * D, Xn, grad->norm, dX + i * D);
		}
		for (int i = 0; i < B * S * D; i++) {
			dX[i] += dX_res[i];
		}
	}

	for (int i = 0; i < B * S; i++) {
		int token = inputs[i];
		if (token == -100) continue;
		float* dtoken = dX + i * D;
		float* dvector = mr->grad.embed + token * D;  // embedding vector for that token
		for (int d = 0; d < D; d++) {
			dvector[d] += dtoken[d];
		}
	}

	free(dX);
}

int inference(Model* model, const int* tokens, int seq_len) {
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
				softmax(A + i * N, N, A + i * N);
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

	softmax(logits, v, logits);
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
