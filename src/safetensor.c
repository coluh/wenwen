#include "safetensor.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cJSON.h"

SafeTensors* load_safetensors(const char* model_safetensors) {
	SafeTensors* sf = calloc(1, sizeof(SafeTensors));

	int fd = open(model_safetensors, O_RDONLY);
	if (fd < 0) {
		return NULL;
	}
	long file_size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);
	sf->file_size = file_size;

	void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (mapped == MAP_FAILED) {
		close(fd);
		return NULL;
	}
	close(fd);
	sf->mapped = mapped;

	uint64_t header_len = *(uint64_t*)mapped;
	cJSON* json = cJSON_ParseWithLength((char*)mapped + sizeof(header_len), header_len);
	if (!json) {
		munmap(mapped, file_size);
		return NULL;
	}

	sf->n_tensors = cJSON_GetArraySize(json) - 1; // except "__metadata__"
	sf->tensors = calloc(sf->n_tensors, sizeof(SafeTensor));
	uint8_t* start = (uint8_t*)mapped + sizeof(header_len) + header_len;

	int idx = 0;
	cJSON* item;
	cJSON_ArrayForEach(item, json) {
		if (strcmp(item->string, "__metadata__") == 0) {
			continue;
		}
		SafeTensor* t = &sf->tensors[idx];
		t->name = strdup(item->string);

		cJSON* dtype = cJSON_GetObjectItem(item, "dtype");
		cJSON* shape = cJSON_GetObjectItem(item, "shape");
		cJSON* offsets = cJSON_GetObjectItem(item, "data_offsets");
		if (strcmp(dtype->valuestring, "BF16") == 0) {
			t->dtype = DType_BF16;
		}
		int ndim = cJSON_GetArraySize(shape);
		for (int i = 0; i < ndim && i < 2; i++) {
			t->shape[i] = cJSON_GetArrayItem(shape, i)->valueint;
		}
		t->offset = cJSON_GetArrayItem(offsets, 0)->valueint;
		t->length = cJSON_GetArrayItem(offsets, 1)->valueint;
		t->data = start + t->offset;
		idx++;
	}

	cJSON_Delete(json);
	return sf;
}

void* get_tensor(SafeTensors* sf, const char* name) {
	for (int i = 0; i < sf->n_tensors; i++) {
		if (strcmp(sf->tensors[i].name, name) == 0) {
			return sf->tensors[i].data;
		}
	}
	printf("ERROR: no such tensor: %s\n", name);
	return NULL;
}

void free_safetensors(SafeTensors* sf) {
	for (int i = 0; i < sf->n_tensors; i++) {
		free(sf->tensors[i].name);
	}
	free(sf->tensors);
	munmap(sf->mapped, sf->file_size);
	free(sf);
}
