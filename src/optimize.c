#include "optimize.h"

#include <stdlib.h>

AdamW* new_optimizer(ModelRunner* mr, float lr) {}
void free_optimizer(AdamW* optimizer) {}

static void step1(AdamW* optimizer, Arguments* args) {
	// m_t = beta ... m_{t-1} beta ... grad
	// theta -= lr * (m / v ... + weight_decay * theta)
}

void optimizer_step(AdamW* optimizer) {
	optimizer->step++;
	for (Arguments* a = optimizer->arguments; a != NULL; a = a->next) {
		step1(optimizer, a);
	}
}

void scheduler_step(Scheduler* s) {
	// linear, cosine...
	*s->lr *= 0.5f;
	;
}
