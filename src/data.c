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

void shuffle(int arr[], int n) {
	for (int i = n - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		int tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

Dataset* new_dataset(const char* bin_path) {
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
	ds->count = size / 4;
	return ds;
}

Dataset dataset_split(Dataset* ds, int begin, int end) {
	return (Dataset){
	    .tokens = ds->tokens + begin,
	    .count = end - begin,
	};
}

void free_dataset(Dataset* ds) {
	munmap(ds->tokens, ds->count * 4);
	free(ds);
}

void dataloader_init(DataLoader* dl, Dataset* ds, int batch_size, int seq_len, bool shuffle) {
	dl->ds = ds;
	dl->batch_size = batch_size;
	dl->seq_len = seq_len;

	dl->batch_count = ds->count / seq_len / batch_size;
	dl->shuffle = shuffle;
	dl->indices = malloc(dl->batch_count * batch_size * sizeof(int));
	dl->batch_idx = 0;

	dl->x = malloc(batch_size * seq_len * sizeof(int));
	dl->y = malloc(batch_size * seq_len * sizeof(int));
}

void dataloader_deinit(DataLoader* dl) {
	free(dl->indices);
	free(dl->x);
	free(dl->y);
}

void dataloader_next(DataLoader* dl) {
	if (dl->batch_idx == 0) {
		for (int i = 0; i < dl->batch_count * dl->batch_size; i++) {
			dl->indices[i] = i;  // tokens视作[size, seq_len], size=batch_count*batch_size
		}
		if (dl->shuffle) {
			shuffle(dl->indices, dl->batch_count * dl->batch_size);
		}
	}

	for (int s = 0; s < dl->batch_size; s++) {
		// copy one sequence a time
		int offset = dl->indices[dl->batch_idx * dl->batch_size + s] * dl->seq_len;
		memcpy(dl->x + s * dl->seq_len + 1, dl->ds->tokens + offset, (dl->seq_len - 1) * sizeof(int));
		memcpy(dl->y + s * dl->seq_len, dl->ds->tokens + offset, (dl->seq_len - 1) * sizeof(int));
		dl->x[s * dl->seq_len] = 151643;  // TODO: config
		dl->y[s * dl->seq_len + dl->seq_len - 1] = 151643;
		// now: x [151643, ...], y [..., 151643]
		// should be:
		// x: 151643, 114514, 114514, 151643, -100, -100
		// y: 114514, 114514, 151643, -100, -100, -100
	}

	dl->batch_idx++;
	if (dl->batch_idx == dl->batch_count) {
		dl->batch_idx = 0;
	}
	return;
}
