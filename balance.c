/*
 * balance.c
 * CS231P - Parallel and Distributed Computing - Homework 8
 *
 * Simulation of a simple give-only load balancing scheme on a ring of
 * K processors. I'm adding lots of small, human/student-style comments
 * so you can follow every step — what each variable means and why we do
 * each operation. The original behaviour is preserved.
 *
 * Short gist (student-speak): every processor has some integer load.
 * Processors wake up at random times and give away some load to their
 * lower neighbors. They never take load. Two give-only rules are here:
 * STRICT: try to equalize the triple (left, self, right) by giving
 *         down to neighbors that are below the triple average.
 * VARIANT: if a neighbor is strictly lower give at least one unit
 *         (roughly half the gap, but at least 1). This avoids dumb
 *         stuck patterns and converges nicely.
 *
 * Build/run: same as original. See the usage notes at the bottom of
 * this file for running experiments and timeseries mode.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TIME_LIMIT   60000000L   /* hard cap on time-intervals per run     */
#define QUIET_WIN      20000L    /* observation window for steady detection */

typedef enum { STRICT = 0, VARIANT = 1 } strategy_t;

static long *load     = NULL;    /* load[i] = ||P_i||                       */
static long *nextfire = NULL;    /* next interval at which P_i activates    */
static int   K        = 0;

/* Uniform integer in [lo, hi] inclusive. */
static long urand(long lo, long hi) {
    /*
     * We want a uniform integer in the closed interval [lo, hi]. If the
     * range is empty or invalid (hi <= lo) just return lo — that's a
     * small defensive shortcut used elsewhere.
     *
     * rand() gives us a 15-bit-ish value; combine two calls to get a
     * larger pseudo-random integer and reduce modulo the span. This is
     * not cryptographically strong but totally fine for simulations.
     */
    if (hi <= lo) return lo;
    long span = hi - lo + 1;
    unsigned long r = ((unsigned long)rand() << 15) ^ (unsigned long)rand();
    return lo + (long)(r % (unsigned long)span);
}

static int left(int i)  { return (i - 1 + K) % K; }
static int right(int i) { return (i + 1) % K; }

/* STRICT: give surplus down to below-average neighbors; never take. */
static long activate_strict(int i) {
    int  l = left(i), r = right(i);
    /*
     * We compute the integer (floor) average of the triple: left, me,
     * right. The active processor will not drop below this avg — it
     * only gives away any excess above avg to neighbors that are below
     * avg. This is the "strict" rule from the assignment.
     */
    long avg = (load[l] + load[i] + load[r]) / 3;   /* floor target */
    long moved = 0; /* total units moved out of i during this activation */

    /* Only do anything if we have more than the local target. */
    if (load[i] > avg) {
        /* Give to left neighbor if it's below the target. */
        if (load[l] < avg) {
            long g = avg - load[l];         /* how much left wants */
            long surplus = load[i] - avg;   /* how much we can spare */
            if (g > surplus) g = surplus;   /* can't give more than we have */
            load[i] -= g; load[l] += g; moved += g;
        }
        /* Then try the right neighbor similarly. Note we recompute
         * surplus from the possibly-reduced load[i] so we don't over-give.
         */
        if (load[r] < avg) {
            long g = avg - load[r];
            long surplus = load[i] - avg;
            if (g > surplus) g = surplus;
            load[i] -= g; load[r] += g; moved += g;
        }
    }
    return moved; /* caller may use this to detect "transfer quiet" */
}

/* VARIANT: give >=1 unit (floor half the gap) to any strictly-lower
 * neighbor; never take. Breaks the staircase fixed point. */
static long activate_variant(int i) {
    int  l = left(i), r = right(i);
    long moved = 0;
    /* nb is just a tiny convenience so we can loop over the two
     * neighbors in order (left then right) with the same code path.
     */
    int  nb[2] = { l, r };
    for (int n = 0; n < 2; n++) {
        int j = nb[n];
        if (load[i] > load[j]) {
            /* gap is how much larger we are than neighbor j */
            long gap = load[i] - load[j];
            long g = gap / 2;      /* give roughly half the gap */
            if (g < 1) g = 1;      /* but always give at least one unit */
            if (g > gap) g = gap;  /* defensive: don't give more than gap */
            load[i] -= g; load[j] += g; moved += g;
        }
    }
    return moved; /* used by caller to determine if any transfer happened */
}

/* Current spread (max - min) of the system. */
static long spread(void) {
    /* Simple pass to compute current spread = max - min. We return
     * that difference as a measure of how balanced the ring is.
     */
    long mn = load[0], mx = load[0];
    for (int i = 0; i < K; i++) {
        if (load[i] < mn) mn = load[i];
        if (load[i] > mx) mx = load[i];
    }
    return mx - mn;
}

/*
 * Run a single simulation. Returns the time-interval at which steady
 * state was detected, or -1 if the time limit was reached.
 * *spread_out = final spread; *balanced_out = 1 iff max-min <= 1.
 */
static long run(int k, long Lmin, long Lmax, long Dmin, long Dmax,
                strategy_t strat, long *spread_out, int *balanced_out) {
    K = k;
    /* allocate arrays sized by k */
    load     = malloc(sizeof(long) * k);
    nextfire = malloc(sizeof(long) * k);

    /* Initialize state: random load per processor and a random next
     * activation time in the interval [Dmin, Dmax] for each.
     * We also keep `total` to sanity-check conservation later.
     */
    long total = 0;
    for (int i = 0; i < k; i++) {
        load[i]     = urand(Lmin, Lmax);
        nextfire[i] = urand(Dmin, Dmax);
        total      += load[i];
    }

    /* t is the current time-interval. win counts how long the desired
     * steady condition has held (we look for QUIET_WIN consecutive
     * intervals of "quiet"). steady_t will store the time we detected
     * steady state or remain -1 if we timed out.
     */
    long t = 0, win = 0, steady_t = -1;

    while (t < TIME_LIMIT) {
        long moved = 0; /* total units moved in this time step */

        /* Walk all processors and activate those whose timer equals t. */
        for (int i = 0; i < k; i++) {
            if (nextfire[i] == t) {
                /* Call the appropriate rule and count how many units
                 * were transferred away from i during this activation.
                 */
                moved += (strat == STRICT) ? activate_strict(i)
                                           : activate_variant(i);
                /* schedule the next activation for this processor */
                nextfire[i] = t + urand(Dmin, Dmax);
            }
        }

        /* Check our two flavors of "steady":
         * - STRICT: steady means "transfer-quiet" i.e., no units moved
         *   during this interval. We need that to hold for QUIET_WIN
         *   consecutive intervals.
         * - VARIANT: steady means the global spread (max-min) is <= 1
         *   and that this condition holds for QUIET_WIN intervals.
         */
        if (strat == STRICT) {
            win = (moved == 0) ? win + 1 : 0;
        } else {
            win = (spread() <= 1) ? win + 1 : 0;
        }
        if (win >= QUIET_WIN) { steady_t = t; break; }
        t++;
    }

    /* report final spread and whether the configuration is "balanced" */
    *spread_out   = spread();
    *balanced_out = (*spread_out <= 1) ? 1 : 0;

    /* Sanity check: load should be conserved (we only moved units). */
    long check = 0;
    for (int i = 0; i < k; i++) check += load[i];
    if (check != total)
        fprintf(stderr, "ERROR: load not conserved (%ld != %ld)\n", check, total);

    /* Clean up dynamic memory. Caller gets steady_t as return value. */
    free(load);     load = NULL;
    free(nextfire); nextfire = NULL;
    return steady_t;
}

/*
 * Timeseries mode: run one trial and print "t spread\n" every SAMPLE_EVERY
 * intervals so Python can plot convergence curves.
 */
#define TIMESERIES_SAMPLE 100L

static void run_timeseries(int k, long Lmin, long Lmax, long Dmin, long Dmax,
                            strategy_t strat) {
    K = k;
    /* Similar setup to run() but we print (time, spread) samples so the
     * Python plotting script can produce convergence curves. We sample
     * every TIMESERIES_SAMPLE intervals.
     */
    load     = malloc(sizeof(long) * k);
    nextfire = malloc(sizeof(long) * k);

    for (int i = 0; i < k; i++) {
        load[i]     = urand(Lmin, Lmax);
        nextfire[i] = urand(Dmin, Dmax);
    }

    long t = 0, win = 0, next_sample = 0;

    while (t < TIME_LIMIT) {
        if (t >= next_sample) {
            /* print a sample line: time spread */
            printf("%ld %ld\n", t, spread());
            next_sample += TIMESERIES_SAMPLE;
        }

        long moved = 0;
        for (int i = 0; i < k; i++) {
            if (nextfire[i] == t) {
                moved += (strat == STRICT) ? activate_strict(i)
                                           : activate_variant(i);
                nextfire[i] = t + urand(Dmin, Dmax);
            }
        }

        if (strat == STRICT) {
            win = (moved == 0) ? win + 1 : 0;
        } else {
            win = (spread() <= 1) ? win + 1 : 0;
        }
        if (win >= QUIET_WIN) break;
        t++;
    }
    printf("%ld %ld\n", t, spread());   /* final point */

    free(load);     load = NULL;
    free(nextfire); nextfire = NULL;
}

/* Average several independent trials for one (k, strategy). */
static void experiment(int k, long Lmin, long Lmax, long Dmin, long Dmax,
                       strategy_t strat, int trials) {
    long  ok_cycles = 0, sum_spread = 0, worst_spread = 0;
    int   steady_count = 0, balanced_count = 0;

    for (int tr = 0; tr < trials; tr++) {
        long sp; int bal;
        /* Seed the RNG in a repeatable but varied way for each trial so
         * runs are independent but reproducible. Then run one trial and
         * collect stats about how long it took to steady and the final
         * spread.
         */
        srand((unsigned)(2024u + tr * 1009u + k * 31u + (unsigned)strat * 7u));
        long st = run(k, Lmin, Lmax, Dmin, Dmax, strat, &sp, &bal);
        if (st >= 0) { steady_count++; ok_cycles += st; }
        if (bal) balanced_count++;
        sum_spread += sp;
        if (sp > worst_spread) worst_spread = sp;
    }

    double avg_cycles = steady_count ? (double)ok_cycles / steady_count : 0.0;
    printf("%-7s k=%-4d | steady:%2d/%-2d  balanced:%2d/%-2d  "
           "avg_cycles=%9.0f  avg_spread=%6.2f  worst_spread=%ld\n",
           strat == STRICT ? "STRICT" : "VARIANT", k,
           steady_count, trials, balanced_count, trials,
           avg_cycles, (double)sum_spread / trials, worst_spread);
}

int main(int argc, char **argv) {
    if (argc == 9 && strcmp(argv[8], "timeseries") == 0) {
        int  k    = atoi(argv[1]);
        long Lmin = atol(argv[2]), Lmax = atol(argv[3]);
        long Dmin = atol(argv[4]), Dmax = atol(argv[5]);
        srand((unsigned)atol(argv[6]));
        strategy_t strat = (atoi(argv[7]) == 0) ? STRICT : VARIANT;
        run_timeseries(k, Lmin, Lmax, Dmin, Dmax, strat);
        return 0;
    }

    if (argc == 8) {
        int  k    = atoi(argv[1]);
        long Lmin = atol(argv[2]), Lmax = atol(argv[3]);
        long Dmin = atol(argv[4]), Dmax = atol(argv[5]);
        srand((unsigned)atol(argv[6]));
        strategy_t strat = (atoi(argv[7]) == 0) ? STRICT : VARIANT;
        long sp; int bal;
        long st = run(k, Lmin, Lmax, Dmin, Dmax, strat, &sp, &bal);
        printf("strategy=%s k=%d steady_at=%ld final_spread=%ld balanced=%d\n",
               strat == STRICT ? "STRICT" : "VARIANT", k, st, sp, bal);
        return 0;
    }

    long Lmin = 10, Lmax = 1000, Dmin = 100, Dmax = 1000;
    int  trials = 20, ks[] = {5, 10, 100};

    printf("CS231P HW8 - Ring load-balancing simulation\n");
    printf("Load ~ U(%ld,%ld), Activity ~ U(%ld,%ld), trials=%d per cell\n",
           Lmin, Lmax, Dmin, Dmax, trials);
    printf("Balanced := (max-min <= 1)\n");
    printf("STRICT steady := no transfers for %ld intervals; "
           "VARIANT steady := spread<=1 held for %ld intervals\n\n",
           QUIET_WIN, QUIET_WIN);

    printf("--- STRICT strategy (rule exactly as specified) ---\n");
    for (size_t i = 0; i < sizeof(ks)/sizeof(ks[0]); i++)
        experiment(ks[i], Lmin, Lmax, Dmin, Dmax, STRICT, trials);

    printf("\n--- VARIANT strategy (convergent give-only fix) ---\n");
    for (size_t i = 0; i < sizeof(ks)/sizeof(ks[0]); i++)
        experiment(ks[i], Lmin, Lmax, Dmin, Dmax, VARIANT, trials);

    return 0;
}
