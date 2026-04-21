import numpy as np
import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from data import N_ACTIVITIES
from fitness import score_population
from ga import init_population, next_generation
from output import (print_generation_stats, print_schedule,
                    save_schedule, plot_fitness, print_violations)

# ── Hyperparameters ────────────────────────────────────────────────────
POP_SIZE        = 500
MIN_GENERATIONS = 100
IMPROVE_THRESH  = 0.01      # stop when avg improvement < 1%
INIT_MUTATION   = 0.01
HALVE_AFTER     = 20        # halve mutation rate every N gens of low improvement
PRINT_EVERY     = 10        # print stats every N generations


def run():
    print(f"Initializing population of {POP_SIZE}...")
    pop     = init_population(POP_SIZE)
    fitness = score_population(pop)

    mutation_rate  = INIT_MUTATION
    prev_avg       = float(np.mean(fitness))
    low_improve_ct = 0

    history = {"best": [], "avg": [], "worst": []}

    gen = 0
    while True:
        gen += 1

        # ── Produce next generation ──────────────────────────────────
        combined, offspring = next_generation(pop, fitness, mutation_rate)
        off_fitness = score_population(offspring)

        # Combine and select best POP_SIZE
        combined_fitness = np.concatenate([fitness, off_fitness])
        top_idx  = np.argsort(combined_fitness)[-POP_SIZE:]
        pop      = combined[top_idx]
        fitness  = combined_fitness[top_idx]

        # ── Metrics ──────────────────────────────────────────────────
        best_f  = float(np.max(fitness))
        avg_f   = float(np.mean(fitness))
        worst_f = float(np.min(fitness))

        history["best"].append(best_f)
        history["avg"].append(avg_f)
        history["worst"].append(worst_f)

        improvement = ((avg_f - prev_avg) / (abs(prev_avg) + 1e-9)) * 100

        if gen % PRINT_EVERY == 0 or gen == 1:
            print_generation_stats(gen, best_f, avg_f, worst_f, improvement)

        # ── Mutation rate halving ────────────────────────────────────
        if abs(improvement) < 0.5:
            low_improve_ct += 1
        else:
            low_improve_ct = 0

        if low_improve_ct >= HALVE_AFTER and mutation_rate > 1e-6:
            mutation_rate /= 2
            low_improve_ct = 0
            print(f"  >> Mutation rate halved to {mutation_rate:.6f}")

        # ── Stopping condition ────────────────────────────────────────
        if gen >= MIN_GENERATIONS and abs(improvement) < IMPROVE_THRESH:
            print(f"\nStopped: avg improvement {improvement:+.4f}% < {IMPROVE_THRESH}% "
                  f"at generation {gen}")
            break

        prev_avg = avg_f

        if gen >= 2000:   # hard cap
            print(f"\nReached max generations ({gen})")
            break

    # ── Final output ─────────────────────────────────────────────────
    best_idx      = int(np.argmax(fitness))
    best_schedule = pop[best_idx]
    best_fitness  = fitness[best_idx]

    print_schedule(best_schedule, best_fitness)
    print_violations(best_schedule)
    save_schedule(best_schedule, best_fitness, "schedule.txt")

    print(f"\nFitness chart:")
    plot_fitness(history)


if __name__ == "__main__":
    run()
