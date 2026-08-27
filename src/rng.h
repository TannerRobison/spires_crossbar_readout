#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* xoshiro256++. Per-reservoir state, so reservoir construction is
 * reproducible from a seed and safe to run on several threads at once --
 * neither of which holds for the global rand(). */
struct spires_rng {
    uint64_t s[4];
};

void   spires_rng_seed(struct spires_rng *r, uint64_t seed);
uint64_t spires_rng_next(struct spires_rng *r);
double spires_rng_double(struct spires_rng *r);   /* uniform on [0, 1) */

#endif // RNG_H
