#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/config.h"
#include "../src/data.h"
#include "../src/optimize.h"
#include "../src/qwen25.h"

// void export_bin(const void* data, size_t n, size_t size, const char* filename) {
// 	FILE* fp = fopen(filename, "wb");
// 	if (!fp) return;
// 	fwrite(data, size, n, fp);
// 	fclose(fp);
// }
//
// float forward_loss(ModelRunner* runner, int* inputs, int* labels) {
// 	float* logits = model_forward(runner, inputs, false);
// 	float loss = criterion(logits, labels, runner->B, runner->S, runner->V, NULL);
// 	free(logits);
// 	return loss;
// }
//
// void test_grad(float* th, float grad_analytic, ModelRunner* runner, int* inputs, int* labels) {
// 	float eps = 1e-5f;
// 	float origin = *th;
// 	*th = origin + eps;
// 	float loss_plus = forward_loss(runner, inputs, labels);
// 	*th = origin - eps;
// 	float loss_minus = forward_loss(runner, inputs, labels);
// 	*th = origin;
// 	float grad_numeric = (loss_plus - loss_minus) / (2 * eps);
// 	float diff = fabsf(grad_analytic - grad_numeric) / (fabsf(grad_analytic) + fabsf(grad_numeric) + 1e-8f);
// 	if (diff > 1e-4) {
// 		printf("not done 😈\ndiff = %.4f\n", diff);
// 	}
// }
//
// void test_grad_line(float* th, const float* grad, int count, ModelRunner* runner, int* inputs, int* labels) {
// 	for (int i = 0; i < count; i++) {
// 		test_grad(th + i, grad[i], runner, inputs, labels);
// 	}
// }

int main() {
	ModelConfig* config = calloc(1, sizeof(ModelConfig));
	*config = (ModelConfig){
	    .vocab_size = 151936,
	    .eos_token_id = 151643,
	    .hidden_size = 128,
	    .num_hidden_layers = 2,
	    .num_attention_heads = 14,
	    .num_key_value_heads = 2,
	    .rms_norm_eps = 1e-6,
	    .rope_theta = 1e6,
	    .intermediate_size = 256,
	};
	const int batch_size = 1;    // 32
	const int max_seq_len = 32;  // 256
	const int pad_id = -100;
	const int num_epochs = 1000;

	Dataset* dataset = new_dataset("./data/tokens.bin", config->eos_token_id);
	// dataset->count = 1,000,000
	dataset->count *= 0.001 * 0.01 * 0.2;
	// Dataset* train_set = dataset_split(dataset, 0, dataset->count * 0.8);
	// Dataset* val_set = dataset_split(dataset, dataset->count * 0.8, dataset->count);
	Dataset* train_set = dataset_split(dataset, 0, dataset->count * 0.5);		  // 1
	Dataset* val_set = dataset_split(dataset, dataset->count * 0.5, dataset->count);  // 1
	DataLoader* train_loader = new_dataloader(train_set, batch_size, max_seq_len, pad_id, true);
	DataLoader* val_loader = new_dataloader(val_set, batch_size, max_seq_len, pad_id, false);

	Model* model = new_model(config);
	ModelRunner* runner = new_modelrunner(model, batch_size, max_seq_len);

	// AdamW* optimizer = new_optimizer(runner, 3e-4);
	// Scheduler scheduler = cosine_scheduler(num_epochs * train_loader->num_batch, &optimizer->lr);
	AdamW* optimizer = new_optimizer(runner, 1e-3);

	float train_loss = 0.0f;
	for (int epoch = 1; epoch <= num_epochs; epoch++) {
		train_loss = 0.0f;
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
			// scheduler_step(&scheduler);
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
		}
		val_loss /= val_loader->num_batch;

		printf("Epoch %2d/%2d | Train Loss: %8.4f | Val Loss: %8.4f | LR: %.8e \n", epoch, num_epochs,
		       train_loss, val_loss, optimizer->lr);
	}

	free_optimizer(optimizer);
	free_dataloader(train_loader);
	free_dataloader(val_loader);
	free_dataset(dataset);
	free_dataset(train_set);
	free_dataset(val_set);
	free_modelrunner(runner);
	free_model(model);
	free(config);

	if (train_loss > 0.01f) {
		printf("train loss = %e > 0.01, you fail😈\n", train_loss);
		exit(1);
	}

	return 0;
}
