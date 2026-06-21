#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "data.h"
#include "optimize.h"
#include "qwen25.h"
#include "tokenizer.h"

void train_model() {
	ModelConfig* config = calloc(1, sizeof(ModelConfig));
	*config = (ModelConfig){
	    .vocab_size = 151936,
	    .eos_token_id = 151643,
	    .hidden_size = 256,
	    .num_hidden_layers = 4,
	    .num_attention_heads = 14,
	    .num_key_value_heads = 2,
	    .rms_norm_eps = 1e-6,
	    .rope_theta = 1e6,
	    .intermediate_size = 1024,
	};
	const int batch_size = 32;
	const int max_seq_len = 256;
	const int pad_id = -100;
	const int num_epochs = 10;

	Dataset* dataset = new_dataset("./data/tokens.bin", config->eos_token_id);
	dataset->count *= 0.001;
	Dataset* train_set = dataset_split(dataset, 0, dataset->count * 0.8);
	Dataset* val_set = dataset_split(dataset, dataset->count * 0.8, dataset->count);
	DataLoader* train_loader = new_dataloader(train_set, batch_size, max_seq_len, pad_id, true);
	DataLoader* val_loader = new_dataloader(val_set, batch_size, max_seq_len, pad_id, false);

	Model* model = new_model(config);
	ModelRunner* runner = new_modelrunner(model, batch_size, max_seq_len);

	AdamW* optimizer = new_optimizer(runner, 3e-4);
	Scheduler scheduler = cosine_scheduler(num_epochs * train_loader->num_batch, &optimizer->lr);

	for (int epoch = 1; epoch <= num_epochs; epoch++) {
		float train_loss = 0.0f;
		float val_loss = 0.0f;

		for (int batch = 0; batch < train_loader->num_batch; batch++) {
			dataloader_next(train_loader);
			int* inputs = train_loader->x;
			int* labels = train_loader->y;

			float* logits = model_forward(runner, inputs, true);
			float* dlogits = runner->ctx.dlogits;
			float loss = criterion(logits, labels, batch_size, max_seq_len, config->vocab_size, dlogits);
			free(logits);
			train_loss += loss;

			zero_grad(runner);
			model_backward(runner, inputs, dlogits);
			optimizer_step(optimizer);
			scheduler_step(&scheduler);
			break;
		}
		train_loss /= train_loader->num_batch;

		// validate
		for (int batch = 0; batch < val_loader->num_batch; batch++) {
			dataloader_next(val_loader);
			int* inputs = val_loader->x;
			int* labels = val_loader->y;

			float* logits = model_forward(runner, inputs, false);
			float* dlogits = runner->ctx.dlogits;
			float loss = criterion(logits, labels, batch_size, max_seq_len, config->vocab_size, dlogits);
			free(logits);
			val_loss += loss;
			break;
		}
		val_loss /= val_loader->num_batch;

		printf("Epoch %2d/%2d | Train Loss: %8.4f | Val Loss: %8.4f | LR: %.8e \n", epoch, num_epochs,
		       train_loss, val_loss, optimizer->lr);
		break;
	}

	// TODO: export model.parameters

	free_optimizer(optimizer);
	free_dataloader(train_loader);
	free_dataloader(val_loader);
	free_dataset(dataset);
	free_dataset(train_set);
	free_dataset(val_set);
	free_modelrunner(runner);
	free_model(model);
	free(config);
}

void load_infer(char* s);

int main(int argc, char* argv[]) {
	srand((unsigned)time(NULL));

	if (argc < 2) {
		printf("\nUsage:\n\n");
		printf("\twen run [input]\n");
		printf("\twen train\n");
		return -1;
	}

	if (strcmp(argv[1], "run") == 0) {
		if (argc < 3) {
			load_infer(NULL);
		} else {
			load_infer(argv[2]);
		}
	} else if (strcmp(argv[1], "train") == 0) {
		train_model();
	} else {
		printf("unknown command: %s\n", argv[1]);
		return -1;
	}

	return 0;
}

#define MODEL "./Qwen2.5-0.5B"

void load_infer(char* s) {
	Tokenizer t = new_tokenizer(MODEL "/vocab.json", MODEL "/merges.txt");

	if (s == NULL) {
		s = u8"The last man on Earth sat alone in a room.";
	}

	int n;
	int* tokens = tokenize(t, s, &n);
	printf("tokens: [ ");
	for (int i = 0; i < n; i++) {
		printf("%d", tokens[i]);
		if (i != n - 1) {
			printf(", ");
		}
	}
	printf(" ]\n");
	printf("\x1B[1;32m%s\x1B[0m", s);
	fflush(stdout);

	// ModelConfig* config = read_config(MODEL "/config.json");
	// void* model = new_model(config, MODEL "/model.safetensors");
	//
	// int outputs[4];
	// int outputs_i = 0;
	// while (1) {
	// 	// printf("\r %d / %d", i, 16);
	// 	int token = inference(model, tokens, n);
	// 	if (token == config->eos_token_id) {
	// 		printf("<|eos|>");
	// 		break;
	// 	}
	//
	// 	outputs[outputs_i] = token;
	// 	outputs_i++;
	//
	// 	while (outputs_i > 0) {
	// 		const char* output = decode_stream(t, outputs, &outputs_i);
	// 		if (output) {
	// 			printf("%s", output);
	// 		}
	// 	}
	// 	fflush(stdout);
	//
	// 	n++;
	// 	tokens = realloc(tokens, n * sizeof(int));
	// 	tokens[n - 1] = token;
	// 	if (n >= 4096) {
	// 		break;
	// 	}
	// }
	//
	// printf("\n");
	// free_model(model);
	// free(config);
	free(tokens);
	free_tokenizer(t);
}
