#ifndef __QWEN25_H__
#define __QWEN25_H__

#include <stdbool.h>

#include "config.h"

typedef struct Qwen25_05B_Model Qwen25_05B_Model;

void* Qwen25_05B(ModelConfig* config, const char* model_safetensors);
void Qwen25_05B_free(Qwen25_05B_Model* model);

// in: [ batch_size, max_seq_len ]
// out: [ batch_size, max_seq_len, vocab_size ]
float* Qwen25_05B_forward(Qwen25_05B_Model* model, const int* x, int B, int S, bool grad);

// return token id
int Qwen25_05B_inference(Qwen25_05B_Model* model, const int* tokens, int n_tokens);

#endif
