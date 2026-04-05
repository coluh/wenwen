#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

typedef struct VocabItem {
	char* s;
	int id;
} VocabItem;

typedef struct MergeRule {
	char* s1;
	char* s2;
} MergeRule;

typedef struct Tokenizer {
	VocabItem* vocabs;
	MergeRule* merges;
	int n_vocabs;
	int n_merges;
} Tokenizer;

Tokenizer new_tokenizer(const char* vocab_json, const char* merges_txt);
void free_tokenizer(Tokenizer t);

int* tokenize(Tokenizer tokenizer, const char* input, int* out_n);
char *decode(Tokenizer tokenizer, int *tokens, int n);

#endif
