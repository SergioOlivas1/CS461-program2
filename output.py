import plotext as plt
import numpy as np
from data import ACT_NAMES, ROOM_NAMES, TIMES, FACILITATORS, N_ACTIVITIES


def format_schedule(schedule: np.ndarray) -> str:
    """Return schedule as formatted string."""
    lines = []
    lines.append(f"\n{'Activity':<12} {'Room':<14} {'Time':<8} {'Facilitator'}")
    lines.append("-" * 55)
    # Sort by time, then activity name
    order = np.argsort(schedule[:, 1])
    for ai in order:
        room = ROOM_NAMES[schedule[ai, 0]]
        time = TIMES[schedule[ai, 1]]
        fac  = FACILITATORS[schedule[ai, 2]]
        lines.append(f"{ACT_NAMES[ai]:<12} {room:<14} {time:<8} {fac}")
    return "\n".join(lines)


def print_schedule(schedule: np.ndarray, fitness: float):
    print(f"\n{'='*55}")
    print(f"  BEST SCHEDULE   (fitness = {fitness:.4f})")
    print(f"{'='*55}")
    print(format_schedule(schedule))


def save_schedule(schedule: np.ndarray, fitness: float, path: str = "schedule.txt"):
    with open(path, "w") as f:
        f.write(f"Best Schedule  (fitness = {fitness:.4f})\n")
        f.write(format_schedule(schedule))
        f.write("\n")
    print(f"\nSchedule saved to {path}")


def print_generation_stats(gen: int, best: float, avg: float, worst: float, improvement: float):
    print(f"Gen {gen:4d} | best={best:7.3f}  avg={avg:7.3f}  worst={worst:7.3f}  "
          f"Δavg={improvement:+.2f}%")


def plot_fitness(history: dict):
    """Plot best/avg/worst fitness over generations using plotext."""
    gens = list(range(1, len(history["best"]) + 1))
    plt.clf()
    plt.plot(gens, history["best"],  label="Best")
    plt.plot(gens, history["avg"],   label="Average")
    plt.plot(gens, history["worst"], label="Worst")
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()


def compute_violations(schedule: np.ndarray) -> dict:
    """Compute constraint violation counts for reporting."""
    from data import ROOM_CAPS, ACT_ENROLL, FACILITATORS, N_FACILITATORS, TIMES
    from data import SLA101, SLA191, ROOM_NAMES, CLOSE_BUILDINGS

    violations = {
        "room_conflicts": 0,
        "room_too_small": 0,
        "facilitator_double_booked": 0,
        "facilitator_overloaded": 0,
        "sla101_same_time": 0,
        "sla191_same_time": 0,
        "sla_191_101_same_time": 0,
    }

    rooms = schedule[:, 0]
    times = schedule[:, 1]
    facs  = schedule[:, 2]

    # Room conflicts
    rt = times * len(ROOM_NAMES) + rooms
    _, counts = np.unique(rt, return_counts=True)
    violations["room_conflicts"] = int(np.sum(counts[counts > 1] - 1))

    # Room too small
    for ai in range(N_ACTIVITIES):
        cap = ROOM_CAPS[rooms[ai]]
        enrl = ACT_ENROLL[ai]
        if cap < enrl:
            violations["room_too_small"] += 1

    # Facilitator issues
    for fi in range(N_FACILITATORS):
        mask = facs == fi
        if not np.any(mask):
            continue
        n_total = int(np.sum(mask))
        if n_total > 4:
            violations["facilitator_overloaded"] += 1
        fac_times = times[mask]
        _, sc = np.unique(fac_times, return_counts=True)
        if np.any(sc > 1):
            violations["facilitator_double_booked"] += 1

    # SLA 101/191 same time
    if times[SLA101[0]] == times[SLA101[1]]:
        violations["sla101_same_time"] = 1
    if times[SLA191[0]] == times[SLA191[1]]:
        violations["sla191_same_time"] = 1
    for i101 in SLA101:
        for i191 in SLA191:
            if times[i101] == times[i191]:
                violations["sla_191_101_same_time"] += 1

    return violations


def print_violations(schedule: np.ndarray):
    v = compute_violations(schedule)
    print("\nConstraint Violations:")
    for k, val in v.items():
        print(f"  {k.replace('_',' '):<35}: {val}")
