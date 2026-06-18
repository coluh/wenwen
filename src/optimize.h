#ifndef __OPTIMIZE_H__
#define __OPTIMIZE_H__

#include "qwen25.h"
typedef struct Arguments {
	float *th; // array
	float *dth; // array
	int count;
	float *state; // each argument one m, v
	struct Arguments *next;
} Arguments;

typedef struct {
	Arguments *arguments;
	float lr;
	float beta1;
	float beta2;
	float weight_decay;


	int step;
} AdamW;

AdamW* new_optimizer(ModelRunner* mr, float lr);
void free_optimizer(AdamW *optimizer);

void optimizer_step(AdamW *optimizer);

typedef struct {
	int total_steps;
	int step;
	float *lr;
} Scheduler;

void scheduler_step(Scheduler *s);

#endif
