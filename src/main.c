#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "qwen25.h"
#include "tokenizer.h"

int main() {

	const char *path = "./Qwen2.5-0.5B";

	Tokenizer t = new_tokenizer("./Qwen2.5-0.5B/vocab.json", "./Qwen2.5-0.5B/merges.txt");

	const char* s = u8"你好，我是Leo，我是游戏行业的老兵";
	printf("input: %s\n", s);

	int n;
	int* tokens = tokenize(t, s, &n);
	printf("tokens: [ ");
	for (int i = 0; i < n; i++) {
		printf("%d", tokens[i]);
		if (i != n - 1) {
			printf(", ");
		}
	}
	printf(" ]\n");

	ModelConfig* config = read_config("./Qwen2.5-0.5B/config.json");
	void* model = Qwen25_05B(config, "./Qwen2.5-0.5B/model.safetensors");

	int cap = 32;
	int *outputs = malloc(cap*sizeof(int));
	int i = 0;
	while (1) {
		printf("\r %d / %d", i, 32);
		fflush(stdout);
		int token = Qwen25_05B_inference(model, tokens, n);
		if (token == config->eos_token_id) {
			break;
		}

		n++;
		tokens = realloc(tokens, n*sizeof(int));
		tokens[n-1] = token;

		outputs[i] = token;
		i++;
		if (i == 32) {
			break;
		}
		if (i >= cap) {
			cap *= 2;
			outputs = realloc(outputs, cap*sizeof(int));
		}
	}
	printf("\n");

	char *output = decode(t, outputs, i);
	printf("complete: %s%s\n", s, output);

	free(output);
	free(outputs);
	Qwen25_05B_free(model);
	free(config);
	free(tokens);
	free_tokenizer(t);
	return 0;
}
