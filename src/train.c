#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "qwen25.h"

// logits: [B, S, v]
// labels: [B, S]
float criterion(float* logits, int* labels, int B, int S, int V, float* grad) {
	float loss = 0.0f;
	int count = 0;

	for (int i = 0; i < B * S; i++) {
		int token = labels[i];
		if (token == -100) continue;

		// combine softmax and cross entropy
		// L = -log(softmax(z)_y)
		//   = -(z_y - log(sum(e^z)))
		float max_val = logits[i * V];
		for (int v = 0; v < V; v++) {
			if (logits[i * V + v] > max_val) {
				max_val = logits[i * V + v];
			}
		}
		float sum_exp = 0.0f;
		for (int v = 0; v < V; v++) {
			sum_exp += expf(logits[i * V + v] - max_val);
		}

		if (grad) {
			float* dlogits = grad + i * V;
			for (int v = 0; v < V; v++) {
				dlogits[v] = expf(logits[i * V + v] - max_val) / sum_exp;
			}
			dlogits[token] -= 1.0f;
		}

		loss += -(logits[i * V + token] - max_val) - logf(sum_exp);
		count++;
	}

	if (count == 0) return 0.0f;
	return loss / count;
}

void train(Model* model, Dataset* dataset, int num_epochs, int batch_size, int max_seq) {
	if (!batch_size) batch_size = 32;
	if (!max_seq) max_seq = 512;
	const int vocab_size = 151936;
	DataLoader train_loader, val_loader;
	Dataset train_data = dataset_split(dataset, 0, dataset->count * 0.8);
	Dataset val_data = dataset_split(dataset, dataset->count * 0.8, dataset->count);
	dataloader_init(&train_loader, &train_data, batch_size, max_seq, true);
	dataloader_init(&val_loader, &val_data, batch_size, max_seq, false);

	ModelRunner* runner = new_modelrunner(model, batch_size, max_seq);

	for (int epoch = 1; epoch <= num_epochs; epoch++) {
		// train
		float train_loss = 0.0f;

		// batch loop
		for (int batch = 0; batch < train_loader.batch_count; batch++) {
			dataloader_next(&train_loader);
			int* inputs = train_loader.x;  // [batch_size, seq_len]
			int* labels = train_loader.y;  // [batch_size, seq_len]

			float* logits = model_forward(runner, inputs, true);
			float* dlogits = malloc(batch_size * max_seq * vocab_size * sizeof(float));
			float loss = criterion(logits, labels, batch_size, max_seq, vocab_size, dlogits);

			zero_grad(runner);
			model_backward(runner, inputs, dlogits);
			/// optimizer.step()

			train_loss += loss;
		}

		float epoch_train_loss = train_loss / train_loader.batch_count;

		// validate
		float val_loss = 0.0f;
		for (int batch = 0; batch < val_loader.batch_count; batch++) {
			dataloader_next(&val_loader);
			int* inputs = val_loader.x;  // [batch_size, seq_len]
			int* labels = val_loader.y;  // [batch_size, seq_len]
			float* logits = model_forward(runner, inputs, false);
			float loss = criterion(logits, labels, batch_size, max_seq, vocab_size, NULL);

			val_loss += loss;
		}

		float epoch_val_loss = val_loss / val_loader.batch_count;

		/// scheduler.step(epoch_val_loss)

		printf("Epoch %2d/%2d | Train Loss: %8.4f | Val Loss: %8.4f | LR: %.8e \n", epoch, num_epochs,
		       epoch_train_loss, epoch_val_loss, 1e-5);

		/// if epoch_val_loss < best_val_loss
		/// 	save
		/// else
		/// 	patience_counter += 1
	}

	// store best model

	dataloader_deinit(&train_loader);
	dataloader_deinit(&val_loader);
	// return history
}
