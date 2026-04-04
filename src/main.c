#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "qwen25.h"
#include "tokenizer.h"

int main() {
	Tokenizer t = new_tokenizer("./Qwen25/vocab.json", "./Qwen25/merges.txt");

	const char* s = u8"Hello, 我是";
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

	ModelConfig* config = read_config("./Qwen25/config.json");
	void* model = Qwen25_05B(config, "./Qwen25/model.safetensors");
	int token = Qwen25_05B_inference(model, tokens, n);
	printf("next token: %d\n", token);
	printf("output: %s\n", get_word(t, token));

	Qwen25_05B_free(model);
	free(config);
	free(tokens);
	free_tokenizer(t);
	return 0;
}
