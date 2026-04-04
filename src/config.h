#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdlib.h>
typedef struct ModelConfig {
	int vocab_size;
	int hidden_size;
	int num_layers;
	float rms_norm_eps;
	int num_heads;
	int num_kv_heads;
	float rope_theta;
	int intermediate_size;
} ModelConfig;

static inline ModelConfig* read_config(const char* config_json) {
	ModelConfig* c = malloc(sizeof(struct ModelConfig));
	c->vocab_size = 151936;
	c->hidden_size = 896;
	c->num_layers = 24;
	c->rms_norm_eps = 1e-6f;
	c->num_heads = 14;
	c->num_kv_heads = 2;
	c->rope_theta = 1000000.0;
	c->intermediate_size = 4864;
	return c;
}

#endif
