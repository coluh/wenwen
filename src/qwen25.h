#ifndef __QWEN25_H__
#define __QWEN25_H__

#include <stdbool.h>

#include "config.h"

typedef struct Qwen25_05B_Model Qwen25_05B_Model;

void* Qwen25_05B(ModelConfig* config, const char* model_safetensors);
void Qwen25_05B_free(Qwen25_05B_Model* model);

float* Qwen25_05B_forward(Qwen25_05B_Model* model, const int* x, int n, bool grad);

// return token id
int Qwen25_05B_inference(Qwen25_05B_Model* model, const int* tokens, int n_tokens);

#endif
