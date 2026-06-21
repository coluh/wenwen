#include "data.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool endswith(const char* s1, const char* s2) {
	int len1 = strlen(s1);
	int len2 = strlen(s2);
	if (len2 > len1) return false;
	return strcmp(s1 + len1 - len2, s2) == 0;
}

static int get_token_count(const int* tokens, int n, int token) {
	int count = 0;
	for (int i = 0; i < n; i++) {
		if (tokens[i] == token) {
			count++;
		}
	}
	return count;
}

static void get_seq_offset(const int* tokens, int n, int (*items)[2], int max_count, int eos_token_id) {
	int count = 0;
	items[count][0] = 0;
	items[count][1] = 0;
	for (int i = 0; i < n; i++) {
		items[count][1]++;
		if (tokens[i] == eos_token_id) {
			count++;
			if (count == max_count) {
				return;
			}
			items[count][0] = i + 1;
			items[count][1] = 0;
		}
	}
}

Dataset* new_dataset(const char* bin_path, int eos_token_id) {
	if (!endswith(bin_path, ".bin")) {
		printf("warning: data_path not end with '.bin'\n");
	}

	int fd = open(bin_path, O_RDONLY);
	if (fd < 0) {
		printf("ERROR: no data file %s\n", bin_path);
	}
	struct stat st;
	fstat(fd, &st);
	int size = st.st_size;

	void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (data == MAP_FAILED) {
		printf("ERROR: fail to mmap\n");
	}
	close(fd);

	Dataset* ds = malloc(sizeof(Dataset));
	ds->tokens = data;
	ds->size = size / sizeof(int);
	ds->splited = false;
	ds->eos_token_id = eos_token_id;

	if (ds->tokens[ds->size - 1] != eos_token_id) {
		printf("bad data: last %d is not eos %d\n", ds->tokens[ds->size - 1], eos_token_id);
		exit(1);
	}
	ds->count = get_token_count(ds->tokens, ds->size, eos_token_id);
	ds->items = calloc(ds->count, 2 * sizeof(int));
	get_seq_offset(ds->tokens, ds->size, ds->items, ds->count, eos_token_id);
	return ds;
}

Dataset* dataset_split(Dataset* ds, int from, int to) {
	if (from < 0 || to > ds->count) {
		printf("dataset_split: not enough items, have %d, want %d ~ %d\n", ds->count, from, to);
		return NULL;
	}

	Dataset* ns = calloc(1, sizeof(Dataset));
	int offset = ds->items[from][0];
	ns->tokens = ds->tokens + offset;
	ns->size = (ds->size * sizeof(int) - offset) / sizeof(int);
	ns->splited = true;
	ns->eos_token_id = ds->eos_token_id;

	ns->count = to - from;
	ns->items = calloc(ns->count, 2*sizeof(int));
	get_seq_offset(ns->tokens, ns->size, ns->items, ns->count, ns->eos_token_id);
	return ns;
}

void free_dataset(Dataset* ds) {
	if (!ds->splited) {
		munmap(ds->tokens, ds->size * sizeof(int));
	}
	free(ds->items);
	free(ds);
}

DataLoader* new_dataloader(Dataset* ds, int batch_size, int seq_len, int pad_id, bool shuffle) {
	DataLoader* dl = calloc(1, sizeof(DataLoader));
	dl->ds = ds;
	dl->batch_size = batch_size;
	dl->max_seq_len = seq_len;
	dl->pad_id = pad_id;
	dl->shuffle = shuffle;

	dl->num_batch = ds->count / batch_size;
	dl->indices = malloc(dl->num_batch * batch_size * sizeof(int));
	dl->batch_idx = 0;

	dl->x = malloc(batch_size * seq_len * sizeof(int));
	dl->y = malloc(batch_size * seq_len * sizeof(int));

	return dl;
}

void free_dataloader(DataLoader* dl) {
	free(dl->indices);
	free(dl->x);
	free(dl->y);
	free(dl);
}

void shuffle(int arr[], int n) {
	for (int i = n - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		int tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

void dataloader_next(DataLoader* dl) {
	if (dl->batch_idx == 0) {  // new epoch, prepare indices
		for (int i = 0; i < dl->num_batch * dl->batch_size; i++) {
			dl->indices[i] = i;
		}
		if (dl->shuffle) {
			shuffle(dl->indices, dl->num_batch * dl->batch_size);
		}
	}

	// one batch, copy <batch_size> sequences
	int* indices = dl->indices + dl->batch_idx * dl->batch_size;
	for (int s = 0; s < dl->batch_size; s++) {
		int indice = indices[s];
		int offset = dl->ds->items[indice][0];
		int length = dl->ds->items[indice][1];
		if (length > dl->max_seq_len) length = dl->max_seq_len;
		int* x = dl->x + s * dl->max_seq_len;
		int* y = dl->y + s * dl->max_seq_len;
		int* seq = dl->ds->tokens + offset;  // seq with eos
		memcpy(x, seq, length * sizeof(int));
		memcpy(y, seq + 1, (length - 1) * sizeof(int));
		for (int i = length; i < dl->max_seq_len; i++) {
			x[i] = dl->pad_id;
		}
		for (int i = length - 1; i < dl->max_seq_len; i++) {
			y[i] = dl->pad_id;
		}

		// x: 114514, 514114, 151643, -100, -100
		// y: 514114, 151643, -100, -100, -100
	}

	dl->batch_idx++;
	if (dl->batch_idx == dl->num_batch) {  // done
		dl->batch_idx = 0;
	}
	return;
}
