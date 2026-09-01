#include "plasticity.h"
#include <math.h>

/* Traces decay first, so a weight update sees only spikes strictly earlier
 * than the current step. Coincident pre/post spikes therefore contribute
 * nothing to each other, which is the usual discrete-time convention. */
void plasticity_decay(struct plasticity_state *ps, size_t n, double dt)
{
    if (!ps || ps->type == PLASTICITY_NONE || !ps->params)
        return;

    double tp = ps->params[PL_TAU_PLUS];
    double tm = ps->params[PL_TAU_MINUS];
    double ti = ps->params[PL_TAU_I];

    double dp = (tp > 0.0) ? exp(-dt / tp) : 0.0;
    double dm = (tm > 0.0) ? exp(-dt / tm) : 0.0;
    double di = (ti > 0.0) ? exp(-dt / ti) : 0.0;

    for (size_t k = 0; k < n; k++) {
        ps->x_pre_e[k]  *= dp;
        ps->x_post_e[k] *= dm;
        ps->x_pre_i[k]  *= di;
        ps->x_post_i[k] *= di;
    }
}

/* One spike vector: in a recurrent reservoir a neuron's spike is a
 * presynaptic event for its outgoing synapses and a postsynaptic event for
 * its incoming ones, so both trace families advance from the same vector. */
void plasticity_accumulate(struct plasticity_state *ps, const double *spikes, size_t n)
{
    if (!ps || ps->type == PLASTICITY_NONE || !spikes)
        return;

    for (size_t k = 0; k < n; k++) {
        double s = spikes[k];
        if (s == 0.0)
            continue;
        ps->x_pre_e[k]  += s;
        ps->x_post_e[k] += s;
        ps->x_pre_i[k]  += s;
        ps->x_post_i[k] += s;
    }
}
