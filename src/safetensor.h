#ifndef __SAFETENSOR_H__
#define __SAFETENSOR_H__

typedef enum DType {
	DType_BF16,
} DType;

typedef struct SafeTensor {
	char* name;
	DType dtype;
	int shape[2];
	int offset;
	int length;
	void* data;
} SafeTensor;

typedef struct SafeTensors {
	SafeTensor* tensors;
	int n_tensors;
	void* mapped;
	int file_size;
} SafeTensors;

SafeTensors* load_safetensors(const char* model_safetensors);
void free_safetensors(SafeTensors *sf);
void *get_tensor(SafeTensors *sf, const char *name);

#endif
