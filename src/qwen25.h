#ifndef __QWEN25_H__
#define __QWEN25_H__

#include "config.h"

typedef struct Qwen25_05B_Model Qwen25_05B_Model;

void* Qwen25_05B(ModelConfig* config, const char *model_safetensors);
void Qwen25_05B_free(Qwen25_05B_Model *model);

// return token id
int Qwen25_05B_inference(Qwen25_05B_Model* model, const int* tokens, int n_tokens);

#endif
