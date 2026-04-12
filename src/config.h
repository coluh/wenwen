#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdlib.h>
typedef struct ModelConfig {
	int vocab_size;
	int eos_token_id;
	int hidden_size;
	int num_hidden_layers;
	float rms_norm_eps;
	int num_attention_heads;
	int num_key_value_heads;
	float rope_theta;
	int intermediate_size;
} ModelConfig;

static inline ModelConfig* read_config(const char* config_json) {
	ModelConfig* c = malloc(sizeof(struct ModelConfig));
	c->vocab_size = 151936;
	c->eos_token_id = 151643;
	c->hidden_size = 896;
	c->num_hidden_layers = 24;
	c->rms_norm_eps = 1e-6f;
	c->num_attention_heads = 14;
	c->num_key_value_heads = 2;
	c->rope_theta = 1000000.0;
	c->intermediate_size = 4864;
	return c;
}

#endif
