#ifndef PLASTICITY_H
#define PLASTICITY_H

#include <stdlib.h>

/* Runtime weight plasticity; see spires_plasticity_type. */
enum plasticity_type {
    PLASTICITY_NONE,
    PLASTICITY_STDP
};

/* Everything a plasticity rule needs beyond the weights themselves.
 *
 * Four traces rather than two: the excitatory rule runs on tau_plus/tau_minus
 * and the inhibitory rule on its own tau, and since a neuron receives from
 * both populations one postsynaptic trace cannot carry both time constants.
 * Traces are per-neuron, not per-synapse, which is what keeps the update at
 * 4n extra state instead of 2*nnz.
 *
 * neuron_sign is the presynaptic neuron's identity and selects the rule for a
 * synapse. It is required, not optional: the weight cannot stand in for it,
 * since a weight depressed to zero no longer carries a sign. */
struct plasticity_state {
    enum plasticity_type type;
    const double *params;
    double *x_pre_e;                  /* decays with tau_plus  */
    double *x_post_e;                 /* decays with tau_minus */
    double *x_pre_i;                  /* decays with tau_i     */
    double *x_post_i;                 /* decays with tau_i     */
    const signed char *neuron_sign;   /* length n, or NULL */
};

/* params indices, mirroring the layout documented in spires.h */
enum {
    PL_TAU_PLUS = 0, PL_TAU_MINUS, PL_LAMBDA, PL_ALPHA_E,
    PL_MU_PLUS, PL_MU_MINUS,
    PL_TAU_I, PL_ETA, PL_RHO_0,
    PL_W_MAX,
    PL_N_PARAMS
};

void plasticity_decay(struct plasticity_state *ps, size_t n, double dt);
void plasticity_accumulate(struct plasticity_state *ps, const double *spikes, size_t n);

#endif // PLASTICITY_H
