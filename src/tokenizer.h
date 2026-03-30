#ifndef _TOKENIZER_H_
#define _TOKENIZER_H_

typedef struct VocabItem {
	char *s;
	int id;
} VocabItem;

typedef struct MergeRule {
	char *s1;
	char *s2;
} MergeRule;

typedef struct Tokenizer {
	VocabItem *vocabs;
	MergeRule *merges;
	int n_vocabs;
	int n_merges;
} Tokenizer;

Tokenizer new_tokenizer(const char *vocab_json, const char *merges_txt);

// return tokens
int *tokenize(Tokenizer tokenizer, const char *input, int * out_n);

#endif
