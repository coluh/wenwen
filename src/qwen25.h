#ifndef __QWEN25_H__
#define __QWEN25_H__

#include <stdbool.h>

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

typedef struct RMSNorm {
	float* weight;
	float eps;

	// if grad
	struct RMSNormContext {
		float* X_normed;  // [N, D] x / rms
		float* rms;	  // [N,]
		float dim;
	} ctx;
} RMSNorm;

typedef struct Model {
	Embedding embedding;

	struct Layer {
		RMSNorm norm;
		struct Attention {
			Linear q;
			Linear k;
			Linear v;
			Linear o;
			// middle variables for backward
			struct AttentionContext {
				int B, S, Hq, Hkv, Dh;
				float* X_in;
				float *Q, *K, *V, *A, *P, *O;
			} ctx;
		} attention;
		RMSNorm post_norm;
		struct FFN {
			Linear gate;
			Linear up;
			Linear down;
			// middle variables for backward
			struct FFNContext {
				float* X_in;
				float* G;      // X @ Wg
				float* G_act;  // SiLU(G)
				float* U;      // X @ Wu
				float* H;      // SiLU(G) @ U
			} ctx;
		} mlp;

	} layers[24];
	RMSNorm norm;
	// middle variables for backward
	float* X_final;	 // [B, S, D]

	// inference KV Cache
	int max_seq;
	float* k_cache;
	float* v_cache;
	int cache_seq_len;

	const ModelConfig* config;
	SafeTensors* sf;
} Model;

void* Qwen25_05B(ModelConfig* config, const char* model_safetensors);
void Qwen25_05B_free(Model* model);

typedef struct ModelRunner {
	Model* model;
	int B, S, V, L, D, Hq, Hkv, Dq, Dkv, Dh, Df;

	// rope cache
	struct RoPE {
		float* freqs;
		float* cosv;
		float* sinv;
	} rope;

	// grads for every paramater
	struct {
		float* embed;  // [V, D]
		struct LayerGrad {
			float* norm;	   // [D]
			float* Wq;	   // [D, D]
			float* Wk;	   // [D, Dkv]
			float* Wv;	   // [D, Dkv]
			float* Wo;	   // [D, D]
			float* post_norm;  // [D]
			float* gate;	   // [D, Df]
			float* up;	   // [D, Df]
			float* down;	   // [Df, D]
		}* layer;
		float* norm;  // [D]
	} grad;

} ModelRunner;

ModelRunner* new_modelrunner(Model* model, int batch_size, int max_seq_len);
void free_modelrunner(ModelRunner* mr);

void zero_grad(ModelRunner* mr);

// in: [ batch_size, max_seq_len ]
// out: [ batch_size, max_seq_len, vocab_size ]
float* model_forward(ModelRunner* mr, const int* inputs, bool grad);

// inputs: [batch_size, max_seq_len]
// dlogits: [batch_size, max_seq_len, vocab_size]
void model_backward(ModelRunner* mr, const int* inputs, float* dlogits);

// return token id
int inference(Model* model, const int* tokens, int n_tokens);

#endif
