import numpy as np
from scipy.special import softmax
from data import N_ACTIVITIES, N_ROOMS, N_TIMES, N_FACILITATORS

rng = np.random.default_rng(np.random.PCG64DXSM())


def init_population(pop_size: int) -> np.ndarray:
    """Create random population. Shape: (pop_size, N_ACTIVITIES, 3)"""
    pop = np.empty((pop_size, N_ACTIVITIES, 3), dtype=np.int32)
    pop[:, :, 0] = rng.integers(0, N_ROOMS,        size=(pop_size, N_ACTIVITIES))
    pop[:, :, 1] = rng.integers(0, N_TIMES,        size=(pop_size, N_ACTIVITIES))
    pop[:, :, 2] = rng.integers(0, N_FACILITATORS, size=(pop_size, N_ACTIVITIES))
    return pop


def select_parents(pop: np.ndarray, fitness: np.ndarray, n_pairs: int):
    """Select parent pairs using softmax-normalized probabilities."""
    probs = softmax(fitness)
    idx_a = rng.choice(len(pop), size=n_pairs, p=probs)
    idx_b = rng.choice(len(pop), size=n_pairs, p=probs)
    return idx_a, idx_b


def crossover(pop: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Single-point crossover on activity list."""
    n = len(idx_a)
    offspring = np.empty((n, N_ACTIVITIES, 3), dtype=np.int32)
    points = rng.integers(1, N_ACTIVITIES, size=n)
    for i, (a, b, pt) in enumerate(zip(idx_a, idx_b, points)):
        offspring[i, :pt]  = pop[a, :pt]
        offspring[i, pt:]  = pop[b, pt:]
    return offspring


def mutate(offspring: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Randomly mutate each gene with probability mutation_rate."""
    mask = rng.random(offspring.shape[:2]) < mutation_rate   # (P, A)
    n_mut = int(mask.sum())
    if n_mut == 0:
        return offspring

    result = offspring.copy()
    p_idx, a_idx = np.where(mask)

    # Randomly choose which variable (room=0, time=1, fac=2) to mutate
    which = rng.integers(0, 3, size=n_mut)
    new_room = rng.integers(0, N_ROOMS,        size=n_mut)
    new_time = rng.integers(0, N_TIMES,        size=n_mut)
    new_fac  = rng.integers(0, N_FACILITATORS, size=n_mut)

    for k in range(n_mut):
        pi, ai, w = p_idx[k], a_idx[k], which[k]
        if w == 0:
            result[pi, ai, 0] = new_room[k]
        elif w == 1:
            result[pi, ai, 1] = new_time[k]
        else:
            result[pi, ai, 2] = new_fac[k]

    return result


def next_generation(pop: np.ndarray, fitness: np.ndarray,
                    mutation_rate: float) -> np.ndarray:
    """Produce one new generation (parents untouched)."""
    pop_size = len(pop)
    n_pairs  = pop_size // 2

    idx_a, idx_b = select_parents(pop, fitness, n_pairs)
    children_ab  = crossover(pop, idx_a, idx_b)
    children_ba  = crossover(pop, idx_b, idx_a)   # both possible offspring
    offspring    = np.concatenate([children_ab, children_ba], axis=0)[:pop_size]
    offspring    = mutate(offspring, mutation_rate)

    # Combine parents + offspring, keep top pop_size
    combined     = np.concatenate([pop, offspring], axis=0)
    return combined, offspring   # caller will re-score offspring separately
