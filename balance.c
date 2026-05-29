/*
 * balance.c
 * CS231P - Parallel and Distributed Computing - Homework 8
 *
 * Simulation of a local-knowledge based diffusion load-balancing
 * strategy on a ring of k processors.
 *
 * Each processor P_i holds ||P_i|| in N (non-negative integers) of
 * never-ending load units. Time is discrete (time-intervals). Each
 * processor independently schedules a balancing activity at random
 * intervals B_t ~ U(Dmin, Dmax). When a processor's activity fires it
 * inspects its two ring neighbors and GIVES load units so that the
 * local neighborhood is balanced. A processor never TAKES units.
 *
 * Two strategies are implemented:
 *
 *   STRICT  (strategy 0): the rule exactly as the assignment states it.
 *           The active processor computes the average of the triple
 *           (left, self, right) and gives its surplus down to whichever
 *           neighbor(s) are below that average, never pulling itself
 *           below the average and never taking. Transfers stop when no
 *           processor has a local surplus -> the run goes "transfer
 *           quiet". This is our STEADY definition for STRICT.
 *
 *   VARIANT (strategy 1): the convergent fix for question 8. The active
 *           processor gives at least one unit (floor(diff/2), min 1)
 *           toward ANY strictly-lower neighbor. This removes the
 *           staircase fixed point and drives max-min -> {0,1}, i.e. a
 *           globally balanced state, for every k. Because a single
 *           boundary unit can keep hopping, "steady" for VARIANT is
 *           defined on the load DISTRIBUTION: the spread (max-min) stays
 *           <= 1 for a long observation window.
 *
 * Usage:
 *   ./balance                     run the full experiment suite
 *   ./balance k Lmin Lmax Dmin Dmax seed strategy
 *
 * Author: see team.txt
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
    long avg = (load[l] + load[i] + load[r]) / 3;   /* floor target */
    long moved = 0;
    if (load[i] > avg) {
        if (load[l] < avg) {
            long g = avg - load[l];
            long surplus = load[i] - avg;
            if (g > surplus) g = surplus;
            load[i] -= g; load[l] += g; moved += g;
        }
        if (load[r] < avg) {
            long g = avg - load[r];
            long surplus = load[i] - avg;
            if (g > surplus) g = surplus;
            load[i] -= g; load[r] += g; moved += g;
        }
    }
    return moved;
}

/* VARIANT: give >=1 unit (floor half the gap) to any strictly-lower
 * neighbor; never take. Breaks the staircase fixed point. */
static long activate_variant(int i) {
    int  l = left(i), r = right(i);
    long moved = 0;
    int  nb[2] = { l, r };
    for (int n = 0; n < 2; n++) {
        int j = nb[n];
        if (load[i] > load[j]) {
            long gap = load[i] - load[j];
            long g = gap / 2;
            if (g < 1) g = 1;        /* always move at least one unit */
            if (g > gap) g = gap;
            load[i] -= g; load[j] += g; moved += g;
        }
    }
    return moved;
}

/* Current spread (max - min) of the system. */
static long spread(void) {
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
    load     = malloc(sizeof(long) * k);
    nextfire = malloc(sizeof(long) * k);

    long total = 0;
    for (int i = 0; i < k; i++) {
        load[i]     = urand(Lmin, Lmax);
        nextfire[i] = urand(Dmin, Dmax);
        total      += load[i];
    }

    long t = 0, win = 0, steady_t = -1;

    while (t < TIME_LIMIT) {
        long moved = 0;
        for (int i = 0; i < k; i++) {
            if (nextfire[i] == t) {
                moved += (strat == STRICT) ? activate_strict(i)
                                           : activate_variant(i);
                nextfire[i] = t + urand(Dmin, Dmax);
            }
        }

        if (strat == STRICT) {
            /* steady == transfer-quiet (no processor has local surplus) */
            win = (moved == 0) ? win + 1 : 0;
        } else {
            /* steady == load distribution balanced and holding */
            win = (spread() <= 1) ? win + 1 : 0;
        }
        if (win >= QUIET_WIN) { steady_t = t; break; }
        t++;
    }

    *spread_out   = spread();
    *balanced_out = (*spread_out <= 1) ? 1 : 0;

    long check = 0;
    for (int i = 0; i < k; i++) check += load[i];
    if (check != total)
        fprintf(stderr, "ERROR: load not conserved (%ld != %ld)\n", check, total);

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
    load     = malloc(sizeof(long) * k);
    nextfire = malloc(sizeof(long) * k);

    for (int i = 0; i < k; i++) {
        load[i]     = urand(Lmin, Lmax);
        nextfire[i] = urand(Dmin, Dmax);
    }

    long t = 0, win = 0, next_sample = 0;

    while (t < TIME_LIMIT) {
        if (t >= next_sample) {
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
