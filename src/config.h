#ifndef __CONFIG_H__
#define __CONFIG_H__


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

ModelConfig* read_config(const char* config_json);

#endif
