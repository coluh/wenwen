#include "config.h"

#include <stdio.h>
#include <stdlib.h>

#include "cJSON.h"

ModelConfig* read_config(const char* config_json) {
	FILE* fp = fopen(config_json, "r");
	fseek(fp, 0, SEEK_END);
	long len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* data = malloc(len + 1);
	fread(data, 1, len, fp);
	fclose(fp);
	data[len] = '\0';

	cJSON* root = cJSON_Parse(data);
	free(data);

	ModelConfig* c = malloc(sizeof(struct ModelConfig));
	c->vocab_size = cJSON_GetObjectItem(root, "vocab_size")->valueint;
	c->eos_token_id = cJSON_GetObjectItem(root, "eos_token_id")->valueint;
	c->hidden_size = cJSON_GetObjectItem(root, "hidden_size")->valueint;
	c->num_hidden_layers = cJSON_GetObjectItem(root, "num_hidden_layers")->valueint;
	c->rms_norm_eps = cJSON_GetObjectItem(root, "rms_norm_eps")->valuedouble;
	c->num_attention_heads = cJSON_GetObjectItem(root, "num_attention_heads")->valueint;
	c->num_key_value_heads = cJSON_GetObjectItem(root, "num_key_value_heads")->valueint;
	c->rope_theta = cJSON_GetObjectItem(root, "rope_theta")->valuedouble;
	c->intermediate_size = cJSON_GetObjectItem(root, "intermediate_size")->valueint;

	cJSON_Delete(root);
	return c;
}
