#include <stdio.h>
#include "tokenizer.h"

int main() {
	Tokenizer t = new_tokenizer("./Qwen25/vocab.json", "./Qwen25/merges.txt");

	const char *s = u8"Hello, 世界！";
	int n;
	int *tokens = tokenize(t, s, &n);
	printf("input: %s\n", s);
	printf("output: [ ");
	for (int i = 0; i < n; i++) {
		printf("%d", tokens[i]);
		if (i != n - 1) {
			printf(", ");
		}
	}
	printf(" ]\n");

	return 0;
}
