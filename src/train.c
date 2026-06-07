#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "data.h"
#include "qwen25.h"

float criterion(float* logits, int* labels) { return 0; }

void train(Qwen25_05B_Model* model, Dataset* dataset, int num_epochs, int batch_size, int max_seq) {
	if (!batch_size) batch_size = 32;
	if (!max_seq) max_seq = 512;
	DataLoader train_loader, val_loader;
	Dataset train_data = dataset_split(dataset, 0, dataset->count * 0.8);
	Dataset val_data = dataset_split(dataset, dataset->count * 0.8, dataset->count);
	dataloader_init(&train_loader, &train_data, batch_size, max_seq, true);
	dataloader_init(&val_loader, &val_data, batch_size, max_seq, false);

	for (int epoch = 1; epoch <= num_epochs; epoch++) {
		// train
		float train_loss = 0.0f;

		// batch loop
		for (int batch = 0; batch < train_loader.batch_count; batch++) {
			dataloader_next(&train_loader);
			int* inputs = train_loader.x;  // [batch_size, seq_len]
			int* labels = train_loader.y;  // [batch_size, seq_len]

			/// optimizer.zero_grad()
			float* logits = Qwen25_05B_forward(model, inputs, batch, max_seq, true);
			float loss = criterion(logits, labels);
			/// backward()
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
			float* logits = Qwen25_05B_forward(model, inputs, batch_size, max_seq, false);
			float loss = criterion(logits, labels);

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
