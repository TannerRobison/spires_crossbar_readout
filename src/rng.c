#include "rng.h"

static inline uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

/* splitmix64: expands one seed word into the four xoshiro needs, so that
 * adjacent seeds (0, 1, 2, ...) still give well-separated streams. */
static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void spires_rng_seed(struct spires_rng *r, uint64_t seed)
{
    if (!r)
        return;
    uint64_t x = seed;
    for (int i = 0; i < 4; i++)
        r->s[i] = splitmix64(&x);
}

uint64_t spires_rng_next(struct spires_rng *r)
{
    const uint64_t result = rotl(r->s[0] + r->s[3], 23) + r->s[0];
    const uint64_t t = r->s[1] << 17;

    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl(r->s[3], 45);

    return result;
}

/* Top 53 bits, the number a double can represent exactly. */
double spires_rng_double(struct spires_rng *r)
{
    return (double)(spires_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}
