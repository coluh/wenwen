#include "optimize.h"

#include <math.h>
#include <stdlib.h>

#include "qwen25.h"

void add_param(AdamW* optimizer, float* th, float* grad, int count) {
	Parameters* p = calloc(1, sizeof(Parameters));
	p->th = th;
	p->dth = grad;
	p->count = count;
	p->states = calloc(count, 2 * sizeof(float));

	Parameters** pp = &optimizer->parameters;
	while (*pp != NULL) {
		pp = &(*pp)->next;
	}
	*pp = p;
}

AdamW* new_optimizer(ModelRunner* mr, float lr) {
	AdamW* aw = calloc(1, sizeof(AdamW));
	aw->lr = lr;
	aw->beta1 = 0.9f;
	aw->beta2 = 0.95f;
	aw->weight_decay = 0.1f;

	const int V = mr->V, D = mr->D, L = mr->L, Dkv = mr->Dkv, Df = mr->Df;
	add_param(aw, mr->model->embedding.table, mr->grad.embed, V * D);
	for (int l = 0; l < L; l++) {
		struct Layer* layer = &mr->model->layers[l];
		struct LayerGrad* grad = &mr->grad.layer[l];
		add_param(aw, layer->norm.weight, grad->norm, D);
		add_param(aw, layer->attention.q.weight, grad->Wq, D * D);
		add_param(aw, layer->attention.k.weight, grad->Wk, D * Dkv);
		add_param(aw, layer->attention.v.weight, grad->Wv, D * Dkv);
		add_param(aw, layer->attention.o.weight, grad->Wo, D * D);
		add_param(aw, layer->post_norm.weight, grad->post_norm, D);
		add_param(aw, layer->mlp.gate.weight, grad->gate, D * Df);
		add_param(aw, layer->mlp.up.weight, grad->up, D * Df);
		add_param(aw, layer->mlp.down.weight, grad->down, Df * D);
	}
	add_param(aw, mr->model->norm.weight, mr->grad.norm, D);

	aw->step = 0;

	return aw;
}

void free_optimizer(AdamW* optimizer) {
	for (Parameters* p = optimizer->parameters; p != NULL;) {
		Parameters* next = p->next;
		free(p->states);
		free(p);
		p = next;
	}
	free(optimizer);
}

static void step1(Parameters* args, float beta1, float beta2, float step, float lr, float weight_decay) {
	// m_t = beta ... m_{t-1} + (1 - beta) ... grad
	// theta -= lr * (m / v ... + weight_decay * theta)

	for (int i = 0; i < args->count; i++) {
		float grad = args->dth[i];
		args->states[i][0] = beta1 * args->states[i][0] + (1 - beta1) * grad;
		args->states[i][1] = beta2 * args->states[i][1] + (1 - beta2) * grad * grad;

		float m_hat = args->states[i][0] / (1 - powf(beta1, step));
		float v_hat = args->states[i][1] / (1 - powf(beta2, step));

		float th = args->th[i];
		args->th[i] += -lr * (m_hat / (sqrtf(v_hat) + 1e-8));
		args->th[i] += -lr * weight_decay * th;
	}
}

void optimizer_step(AdamW* aw) {
	aw->step++;
	for (Parameters* p = aw->parameters; p != NULL; p = p->next) {
		step1(p, aw->beta1, aw->beta2, aw->step, aw->lr, aw->weight_decay);
	}
}

float linear_cosine_schedule(int step, int total_steps, int warmup_steps, float start_lr, float peak_lr, float end_lr) {
	if (step < warmup_steps) {
		float progress = (float)step / warmup_steps;
		return start_lr + progress * (peak_lr - start_lr);
	} else {
		float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
		float cosine_decay = 0.5f * (1.0f + cosf(M_PI * progress));
		return end_lr + cosine_decay * (peak_lr - end_lr);
	}
}

Scheduler cosine_scheduler(int total_steps, float* lr) {
	return (Scheduler){
	    .total_steps = total_steps,
	    .warmup_steps = total_steps * 0.05f,
	    .start_lr = 0.0f,
	    .peak_lr = *lr,
	    .end_lr = 0.0f,
	    .step = 0,
	    .lr = lr,
	};
}

void scheduler_step(Scheduler* s) {
	*s->lr = linear_cosine_schedule(s->step, s->total_steps, s->warmup_steps, s->start_lr, s->peak_lr, s->end_lr);
	s->step++;
}
