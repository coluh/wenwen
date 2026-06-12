#ifndef __QWEN25_H__
#define __QWEN25_H__

#include <stdbool.h>

#include "config.h"

typedef struct Model Model;

void* Qwen25_05B(ModelConfig* config, const char* model_safetensors);
void Qwen25_05B_free(Model* model);

typedef struct RMSNormContext {
	float* X_hat;  // x / rms
	float rms;
	float dim;
} RMSNormContext;

typedef struct RMSNorm {
	float *gemma;
	float eps;

	// if grad
	RMSNormContext ctx;
} RMSNorm;

void rmsnorm_backward(RMSNorm *n, float *dy, float *out_dx);

typedef struct ModelRunner {
	Model* model;
	int B, S;
	int V, L;
	int D, Hq, Hkv, Df;
	int Dh, Dkv;

	// rope cache
	struct {
		float* freqs;
		float* cosv;
		float* sinv;
	} rope;

	// middle variables
	struct LayerContext {
	}* layer;
	float* X_normed;  // [B, S, D]
	float rms;
	float dim;
	float* X_final;	 // [B, S, D]

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
