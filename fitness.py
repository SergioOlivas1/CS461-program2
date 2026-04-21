import numpy as np
from data import (
    N_ACTIVITIES, N_ROOMS, N_TIMES, N_FACILITATORS,
    ROOM_CAPS, ACT_ENROLL, ACT_PREF_IDX, ACT_OTHER_IDX,
    TYLER_IDX, SLA101, SLA191, ROOM_NAMES, CLOSE_BUILDINGS, TIMES
)

ROOM_CAPS_ARR = np.array(ROOM_CAPS, dtype=np.float32)
ACT_ENROLL_ARR = np.array(ACT_ENROLL, dtype=np.float32)


def score_population(pop: np.ndarray) -> np.ndarray:
    """
    Score all schedules in the population at once.
    pop shape: (P, N_ACTIVITIES, 3)  — last dim: [room_idx, time_idx, fac_idx]
    Returns fitness array of shape (P,).
    """
    P = pop.shape[0]
    fitness = np.zeros(P, dtype=np.float64)

    rooms = pop[:, :, 0]   # (P, A)
    times = pop[:, :, 1]   # (P, A)
    facs  = pop[:, :, 2]   # (P, A)

    # ── Room size scoring ──────────────────────────────────────────────
    caps = ROOM_CAPS_ARR[rooms]          # (P, A)
    enrl = ACT_ENROLL_ARR[None, :]      # (1, A)

    too_small  = caps < enrl
    too_big_3x = caps > 3 * enrl
    too_big_15 = (~too_big_3x) & (caps > 1.5 * enrl)
    just_right = ~(too_small | too_big_15 | too_big_3x)

    fitness += np.sum(-0.5 * too_small, axis=1)
    fitness += np.sum(-0.4 * too_big_3x, axis=1)
    fitness += np.sum(-0.2 * too_big_15, axis=1)
    fitness += np.sum( 0.3 * just_right, axis=1)

    # ── Room conflicts (same room & time) ──────────────────────────────
    for p in range(P):
        rt = times[p] * N_ROOMS + rooms[p]   # unique room-time key
        _, counts = np.unique(rt, return_counts=True)
        conflicts = np.sum(counts[counts > 1] - 1)
        fitness[p] -= 0.5 * conflicts

    # ── Facilitator scoring ────────────────────────────────────────────
    for ai in range(N_ACTIVITIES):
        f = facs[:, ai]   # (P,)
        # preferred / other / else
        pref_mask = np.zeros(P, dtype=bool)
        othr_mask = np.zeros(P, dtype=bool)
        for fi in ACT_PREF_IDX[ai]:
            pref_mask |= (f == fi)
        for fi in ACT_OTHER_IDX[ai]:
            othr_mask |= (f == fi)
        else_mask = ~(pref_mask | othr_mask)
        fitness +=  0.5 * pref_mask
        fitness +=  0.2 * othr_mask
        fitness += -0.1 * else_mask

    # ── Facilitator load per schedule ──────────────────────────────────
    for p in range(P):
        t = times[p]    # (A,)
        f = facs[p]     # (A,)

        # Count per facilitator per time slot
        for fi in range(N_FACILITATORS):
            act_mask = (f == fi)
            if not np.any(act_mask):
                continue
            n_total = int(np.sum(act_mask))

            # Overload / underload total
            if n_total > 4:
                fitness[p] -= 0.5
            elif n_total < 3:
                if fi == TYLER_IDX:
                    if n_total < 2:
                        fitness[p] -= 0.4
                else:
                    fitness[p] -= 0.4

            # Per-time-slot double booking
            fac_times = t[act_mask]
            _, slot_counts = np.unique(fac_times, return_counts=True)
            for sc in slot_counts:
                if sc == 1:
                    fitness[p] += 0.2
                else:
                    fitness[p] -= 0.2 * sc

            # Consecutive time slots for facilitators
            unique_slots = np.unique(fac_times)
            for s in unique_slots:
                if s + 1 in unique_slots:
                    fitness[p] -= 0.5   # penalty same as 101/191 consecutive

    # ── SLA 101 section spacing ────────────────────────────────────────
    t101 = times[:, SLA101[0]], times[:, SLA101[1]]
    same_101 = t101[0] == t101[1]
    far_101  = np.abs(t101[0].astype(int) - t101[1].astype(int)) > 4
    fitness -= 0.5 * same_101
    fitness += 0.5 * far_101

    # ── SLA 191 section spacing ────────────────────────────────────────
    t191 = times[:, SLA191[0]], times[:, SLA191[1]]
    same_191 = t191[0] == t191[1]
    far_191  = np.abs(t191[0].astype(int) - t191[1].astype(int)) > 4
    fitness -= 0.5 * same_191
    fitness += 0.5 * far_191

    # ── SLA 101 vs SLA 191 cross-section rules ─────────────────────────
    for i101 in SLA101:
        for i191 in SLA191:
            ti = times[:, i101].astype(int)
            tj = times[:, i191].astype(int)
            diff = np.abs(ti - tj)

            consec = diff == 1
            sep1   = diff == 2
            same   = diff == 0

            fitness += 0.5  * consec
            fitness += 0.25 * sep1
            fitness -= 0.25 * same

            # Building proximity penalty for consecutive slots
            for p in range(P):
                if consec[p]:
                    r_i = ROOM_NAMES[rooms[p, i101]]
                    r_j = ROOM_NAMES[rooms[p, i191]]
                    in_close_i = r_i in CLOSE_BUILDINGS
                    in_close_j = r_j in CLOSE_BUILDINGS
                    # Penalty if one is in close building and the other isn't
                    if in_close_i != in_close_j:
                        fitness[p] -= 0.4

    return fitness
