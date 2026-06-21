#ifndef __OPTIMIZE_H__
#define __OPTIMIZE_H__

#include "qwen25.h"
typedef struct Parameters {
	float *th; // array
	float *dth; // array
	int count;
	float (*states)[2]; // m, v for each argument
	struct Parameters *next;
} Parameters;

typedef struct {
	Parameters *parameters;
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;

	int step;
} AdamW;

AdamW* new_optimizer(ModelRunner* mr, float lr);
void free_optimizer(AdamW *optimizer);

void optimizer_step(AdamW *optimizer);

typedef struct {
	int total_steps;
	int warmup_steps;
	float start_lr;
	float peak_lr;
	float end_lr;
	int step;
	float *lr;
} Scheduler;

Scheduler cosine_scheduler(int total_steps, float* lr);
void scheduler_step(Scheduler *s);

#endif
