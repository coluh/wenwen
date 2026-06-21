#ifndef __DATA_H__
#define __DATA_H__

#include <stdbool.h>

typedef struct Dataset {
	int* tokens;  // seen as [item_count, seq_len]
	int size;     // int count of tokens
	// content: ..., EOS, ... EOS, ..., ..., ... , EOS
	// <count> sequences
	int splited;
	int eos_token_id;

	int (*items)[2];  // offset and length of each sequence, length include eos
	int count;
} Dataset;

Dataset* new_dataset(const char* bin_path, int eos_token_id);
Dataset* dataset_split(Dataset* ds, int from, int to);
void free_dataset(Dataset* ds);

typedef struct DataLoader {
	Dataset* ds;
	int batch_size;
	int max_seq_len;
	int pad_id;
	bool shuffle;

	int num_batch;	// ds.count / batch_size
	int* indices;	// indexes of sequences, this epoch
	int batch_idx;	// increase every iteration

	int* x;	 // Tensor[batch_size, max_seq_len]
	int* y;	 // Tensor[batch_size, max_seq_len]
} DataLoader;

DataLoader* new_dataloader(Dataset* ds, int batch_size, int seq_len, int pad_id, bool shuffle);
void free_dataloader(DataLoader* dl);
void dataloader_next(DataLoader* dl);

#endif
