#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "qwen25.h"
#include "tokenizer.h"

#define MODEL "./Qwen2.5-0.5B"

int main(int argc, char* argv[]) {
	Tokenizer t = new_tokenizer(MODEL "/vocab.json", MODEL "/merges.txt");

	char* s;
	if (argc >= 2) {
		s = argv[1];
	} else {
		s = u8"The last man on Earth sat alone in a room.";
	}
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

	ModelConfig* config = read_config(MODEL "/config.json");
	void* model = Qwen25_05B(config, MODEL "/model.safetensors");

	int outputs[4];
	int outputs_i = 0;
	int i = 0;
	while (1) {
		// printf("\r %d / %d", i, 16);
		int token = Qwen25_05B_inference(model, tokens, n);
		if (token == config->eos_token_id) {
			printf("<|eos|>\n");
			break;
		}

		n++;
		tokens = realloc(tokens, n * sizeof(int));
		tokens[n - 1] = token;

		outputs[outputs_i] = token;
		outputs_i++;

		while (outputs_i > 0) {
			const char* output = decode_stream(t, outputs, &outputs_i);
			if (output) {
				printf("%s", output);
			}
		}

		fflush(stdout);
		i++;
		// if (i == 64) {
		// 	break;
		// }
	}

	Qwen25_05B_free(model);
	free(config);
	free(tokens);
	free_tokenizer(t);
	return 0;
}
