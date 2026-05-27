#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* read_file(const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) {
		printf("ERROR: no file %s\n", filename);
	}

	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	rewind(fp);

	char* buffer = malloc(size + 1);
	fread(buffer, 1, size, fp);
	buffer[size] = '\0';
	fclose(fp);

	return buffer;
}

static VocabItem* read_vocabs(const char* vocab_json, int* out_n) {
	int cap = 256;
	VocabItem* vocabs = malloc(cap * sizeof(VocabItem));
	int n_vocabs = 0;

	char* content = read_file(vocab_json);
	char* p = content;
	while (*p) {
		while (*p && *p != '"') {
			p++;
		}
		if (!*p) {
			break;
		}

		char* token_start = ++p;
		while (*p != '"') {
			if (*p == '\\') {
				p++;
			}
			p++;
		}
		char* const token = calloc(p - token_start + 1, 1);
		char* q = token;
		while (*token_start != '"') {
			if (*token_start == '\\') {
				token_start++;
			}
			*q++ = *token_start++;
		}
		*q = '\0';

		while (*p != ':') {
			p++;
		}
		p++;
		while (*p == ' ' || *p == '\t') {
			p++;
		}
		int id = 0;
		while (*p >= '0' && *p <= '9') {
			id = id * 10 + (*p - '0');
			p++;
		}

		vocabs[n_vocabs] = (VocabItem){token, id};
		n_vocabs++;
		if (n_vocabs >= cap) {
			cap *= 2;
			vocabs = realloc(vocabs, cap * sizeof(VocabItem));
		}
	}
	free(content);

	*out_n = n_vocabs;
	return vocabs;
}

static MergeRule* read_merges(const char* merges_txt, int* out_n) {
	int cap = 256;
	MergeRule* rules = malloc(cap * sizeof(MergeRule));
	int n_rules = 0;

	char* content = read_file(merges_txt);
	char* p = content;
	while (*p) {
		while (*p && (*p == '\n' || *p == '\r')) {
			p++;
		}
		if (!*p) {
			break;
		}

		const char* start1 = p;
		while (*p != ' ' && *p != '\t') {
			p++;
		}
		int len1 = p - start1;
		char* token1 = calloc(len1 + 1, 1);
		memcpy(token1, start1, len1);

		while (*p == ' ' || *p == '\t') {
			p++;
		}
		const char* start2 = p;
		while (*p != ' ' && *p != '\t' && *p != '\n') {
			p++;
		}
		int len2 = p - start2;
		char* token2 = calloc(len2 + 1, 1);
		memcpy(token2, start2, len2);

		rules[n_rules] = (MergeRule){token1, token2};
		n_rules++;
		if (n_rules >= cap) {
			cap *= 2;
			rules = realloc(rules, cap * sizeof(MergeRule));
		}
	}
	free(content);

	*out_n = n_rules;
	return rules;
}

Tokenizer new_tokenizer(const char* vocab_json, const char* merges_txt) {
	Tokenizer t;
	t.vocabs = read_vocabs(vocab_json, &t.n_vocabs);
	t.merges = read_merges(merges_txt, &t.n_merges);
	return t;
}

void free_tokenizer(Tokenizer t) {
	for (int i = 0; i < t.n_vocabs; i++) {
		free(t.vocabs[i].s);
	}
	for (int i = 0; i < t.n_merges; i++) {
		free(t.merges[i].s1);
		free(t.merges[i].s2);
	}
	free(t.vocabs);
	free(t.merges);
}

static const char* bytes_to_unicode(char b_char) {
	static char u[3];
	memset(u, 0, 3);

	int b = (int)(unsigned char)b_char;
	int c;

	if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
		c = b;
	} else {
		if (b < 33) {
			c = 256 + b;  // 0 256, 32 288
		} else if (b < 161) {
			c = 256 + 32 + b - 126;	 // 127 289, 160 322
		} else {
			c = 256 + 32 + 161 - 126;  // 173 323
		}
	}

	if (c < 0b10000000) {
		// 0b 01111111
		u[0] = c;
	} else if (c < 2048) {
		// 0b 11011111 10111111
		u[0] = 0b11000000 | (c >> 6);
		u[1] = 0b10000000 | (c & 0b00111111);
	}

	return u;
}

// create a cute cat
static char* new_cat(const char* s1, const char* s2) {
	char* s = malloc(strlen(s1) + strlen(s2) + 1);
	memcpy(s, s1, strlen(s1));
	memcpy(s + strlen(s1), s2, strlen(s2));
	s[strlen(s1) + strlen(s2)] = '\0';
	return s;
}

int* tokenize(Tokenizer tokenizer, const char* input, int* out_n) {
	int n = strlen(input);
	char** tokens = malloc(n * sizeof(char*));

	// input raw bytes as id, map to string
	// raw bytes of input -> each byte to a unicode/utf-8 string of one unicode
	for (int i = 0; i < n; i++) {
		// this is wrong
		// const char* p =
		//     id_to_s(tokenizer.vocabs, tokenizer.n_vocabs, (unsigned char)input[i]);

		// this is right
		const char* p = bytes_to_unicode(input[i]);
		tokens[i] = strdup(p);
	}

	// merge token string
	for (int m = 0; m < tokenizer.n_merges; m++) {
		MergeRule* rule = &tokenizer.merges[m];
		while (1) {
			int has_combine = 0;
			int i = 0;
			while (i < n - 1) {
				if ((strcmp(tokens[i], rule->s1) == 0) && (strcmp(tokens[i + 1], rule->s2) == 0)) {
					free(tokens[i]);
					free(tokens[i + 1]);
					tokens[i] = new_cat(rule->s1, rule->s2);
					for (int j = i + 1; j < n - 1; j++) {
						tokens[j] = tokens[j + 1];
					}
					has_combine = 1;
					n--;
				} else {
					i++;
				}
			}
			if (!has_combine) {
				break;
			}
		}
	}

	int* ids = malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		ids[i] = -1;
		for (int j = 0; j < tokenizer.n_vocabs; j++) {
			if (strcmp(tokens[i], tokenizer.vocabs[j].s) == 0) {
				ids[i] = tokenizer.vocabs[j].id;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		free(tokens[i]);
	}
	free(tokens);

	*out_n = n;
	return ids;
}

const char* get_word(Tokenizer tokenizer, int token) {
	for (int i = 0; i < tokenizer.n_vocabs; i++) {
		if (tokenizer.vocabs[i].id == token) {
			return tokenizer.vocabs[i].s;
		}
	}
	return "unknown";
}

int unicode_to_byte(int u) {
	if ((u >= 33 && u <= 126) || (u >= 161 && u <= 172) || (u >= 174 && u <= 255)) {
		return u;
	}
	if (u >= 256 && u <= 288) {
		return u - 256;
	}
	if (u >= 289 && u <= 322) {
		return u - 162;
	}
	return 173;
}

char* decode(Tokenizer tokenizer, int* tokens, int n) {
	int cap = 32;
	char* p = calloc(cap, sizeof(char));
	for (int i = 0; i < n; i++) {
		const char* s = get_word(tokenizer, tokens[i]);
		while (strlen(s) + strlen(p) >= cap) {
			cap *= 2;
			p = realloc(p, cap);
		}
		strcat(p, s);
	}

	char* s = calloc(strlen(p), sizeof(char));
	char* q = s;
	char* u = p;
	while (*u != '\0') {
		if ((*u >> 7) == 0) {
			*q = *u;
			q++;
			u++;
		} else {
			// no more than 2 byte
			int a = ((*u) & 0b00011111);
			u++;
			int b = ((*u) & 0b00111111);
			int v = (a << 6) + b;
			*q = unicode_to_byte(v);
			q++;
			u++;
		}
	}

	free(p);
	return s;
}

static int utf8_len(unsigned char c) {
	if ((c >> 7) == 0) return 1;	    // 0xxxxxxx
	if ((c >> 5) == 0b110) return 2;    // 110xxxxx
	if ((c >> 4) == 0b1110) return 3;   // 1110xxxx
	if ((c >> 3) == 0b11110) return 4;  // 11110xxx
	return 0;
}

#define MAX_TOKEN_LEN 128

// decode some utf8 chars
// consume 1 to n tokens, save remaining n
// usage:
// int tokens[n];
// while (n > 0) {
// 	const char *s = decode_stream(tokenizer, tokens, &n);
// 	if (s) printf(s);
// }
const char* decode_stream(Tokenizer tokenizer, int* tokens, int* n) {
	/*
	 * token map to utf8 string, split by byte, map to origin bytes, to utf8 string
	 * 1 token -> [1, MAX_TOKEN_LEN] bytes -> [1, MAX_TOKEN_LEN] bytes -> [0, MAX_TOKEN_LEN] utf chars
	 */
	static char buffer[MAX_TOKEN_LEN + 8] = {0};  // TODO: can use a ring buffer
	static char utf8_buffer[MAX_TOKEN_LEN + 8] = {0};

	while (strlen(buffer) < 8 && *n > 0) {
		const char* s = get_word(tokenizer, *tokens);
		for (int i = 0; i < *n - 1; i++) {
			tokens[i] = tokens[i + 1];
		}
		(*n)--;
		strcat(buffer, s);
	}

	memset(utf8_buffer, 0, MAX_TOKEN_LEN + 8);
	// 1 utf8 char -> [1, 4] bytes -> map to unicodes -> [2, 8] converted bytes
	while (strlen(buffer) >= 8) {
		char out[5];
		int used[4];
		char* u = buffer;
		for (int i = 0; i < 4; i++) {
			if ((*u >> 7) == 0) {
				out[i] = *u++;
				used[i] = 1;
			} else {
				int a = ((*u++) & 0b00011111);
				int b = ((*u++) & 0b00111111);
				int v = (a << 6) + b;
				out[i] = unicode_to_byte(v);
				used[i] = 2;
			}
		};
		int len = utf8_len(out[0]);
		out[len] = '\0';
		strcat(utf8_buffer, out);

		int used_len = 0;
		for (int i = 0; i < len; i++) {
			used_len += used[i];
		}
		int buffer_len = strlen(buffer);
		memmove(buffer, buffer + used_len, buffer_len - used_len);
		buffer[buffer_len - used_len] = '\0';
		return utf8_buffer;
	}

	return NULL;
}
