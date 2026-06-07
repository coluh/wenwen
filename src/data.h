#ifndef __DATA_H__
#define __DATA_H__

#include <stdbool.h>

// FIXME: store/load <|EOS|> token
typedef struct Dataset {
	int* tokens;  // [item_size, seq_len]
	int count;
} Dataset;

Dataset* new_dataset(const char* bin_path);
Dataset dataset_split(Dataset* ds, int begin, int end);
void free_dataset(Dataset* ds);

typedef struct DataLoader {
	Dataset* ds;
	int batch_size;
	int seq_len;

	int batch_count;
	bool shuffle;
	int* indices;
	int batch_idx;	// [0, batch_count]

	int* x;	 // [batch_size, seq_len]
	int* y;
} DataLoader;

void dataloader_init(DataLoader* dl, Dataset* ds, int batch_size, int seq_len, bool shuffle);
void dataloader_deinit(DataLoader* dl);
void dataloader_next(DataLoader* dl);

#endif
