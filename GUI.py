"""
Interactive GA Scheduler GUI using PySimpleGUI + Matplotlib
Supports runtime parameter tuning, live plotting, and real-time schedule display.
"""

import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import queue
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data import N_ACTIVITIES
from fitness import score_population
from ga import init_population, next_generation
from output import format_schedule, compute_violations

# ── Configuration ──────────────────────────────────────────────────────
sg.theme("DarkBlue2")

# Default hyperparameters
DEFAULT_PARAMS = {
    "pop_size": 500,
    "min_generations": 100,
    "improve_thresh": 0.01,
    "init_mutation": 0.01,
    "halve_after": 20,
    "print_every": 10,
}


class GARunner:
    """Encapsulates GA execution with pause/resume/stop capability."""

    def __init__(self, params: dict):
        """Initialize GA with given parameters."""
        self.params = params
        self.running = False
        self.paused = False
        self.population = None
        self.fitness = None
        self.generation = 0
        self.history = {"best": [], "avg": [], "worst": []}
        self.mutation_rate = params["init_mutation"]
        self.output_queue = queue.Queue()

    def initialize(self):
        """Create initial population."""
        self.population = init_population(self.params["pop_size"])
        self.fitness = score_population(self.population)
        self.generation = 0
        self.history = {"best": [], "avg": [], "worst": []}
        self.mutation_rate = self.params["init_mutation"]

    def step(self, mutation_rate_override: float = None) -> dict:
        """Perform one generation step. Return metrics dict."""
        if mutation_rate_override is not None:
            self.mutation_rate = mutation_rate_override

        self.generation += 1
        combined, offspring = next_generation(
            self.population, self.fitness, self.mutation_rate
        )
        off_fitness = score_population(offspring)

        combined_fitness = np.concatenate([self.fitness, off_fitness])
        top_idx = np.argsort(combined_fitness)[-self.params["pop_size"] :]
        self.population = combined[top_idx]
        self.fitness = combined_fitness[top_idx]

        best_f = float(np.max(self.fitness))
        avg_f = float(np.mean(self.fitness))
        worst_f = float(np.min(self.fitness))

        self.history["best"].append(best_f)
        self.history["avg"].append(avg_f)
        self.history["worst"].append(worst_f)

        return {
            "gen": self.generation,
            "best": best_f,
            "avg": avg_f,
            "worst": worst_f,
            "mutation_rate": self.mutation_rate,
        }

    def get_best_schedule(self):
        """Return (schedule, fitness)."""
        best_idx = int(np.argmax(self.fitness))
        return self.population[best_idx], self.fitness[best_idx]

    def should_stop(self, prev_avg: float) -> bool:
        """Check stopping criteria."""
        if self.generation < self.params["min_generations"]:
            return False
        if self.generation >= 2000:
            return True
        avg_f = float(np.mean(self.fitness))
        improvement = ((avg_f - prev_avg) / (abs(prev_avg) + 1e-9)) * 100
        return abs(improvement) < self.params["improve_thresh"]


def ga_thread_worker(runner: GARunner, param_queue: queue.Queue, output_queue: queue.Queue):
    """Run GA in background thread, listen for parameter updates."""
    runner.initialize()
    prev_avg = float(np.mean(runner.fitness))
    low_improve_ct = 0

    while runner.running:
        # Check for parameter updates
        try:
            new_mutation = param_queue.get_nowait()
            if new_mutation is not None:
                runner.mutation_rate = new_mutation
        except queue.Empty:
            pass

        # Perform one step
        metrics = runner.step()
        avg_f = metrics["avg"]
        improvement = ((avg_f - prev_avg) / (abs(prev_avg) + 1e-9)) * 100

        # Adaptive mutation halving
        if abs(improvement) < 0.5:
            low_improve_ct += 1
        else:
            low_improve_ct = 0

        if low_improve_ct >= runner.params["halve_after"] and runner.mutation_rate > 1e-6:
            runner.mutation_rate /= 2
            low_improve_ct = 0

        metrics["improvement"] = improvement
        output_queue.put(metrics)

        # Check stopping condition
        if runner.should_stop(prev_avg):
            break

        prev_avg = avg_f
        time.sleep(0.01)  # Small delay to keep UI responsive

    runner.running = False
    output_queue.put(None)  # Signal completion


def create_window(runner: GARunner):
    """Build PySimpleGUI window layout."""

    # --- Control Panel ---
    control_col = [
        [sg.Text("GA Scheduler Control Panel", font=("Arial", 14, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Text("Population Size:", size=(20, 1)), sg.InputText("500", key="pop_size", size=(10, 1))],
        [sg.Text("Mutation Rate:", size=(20, 1)), sg.Slider((0.001, 0.1), default_value=0.01, resolution=0.001, key="mutation_rate", orientation="h")],
        [sg.Text("Initial Mutation:", size=(20, 1)), sg.InputText("0.01", key="init_mutation", size=(10, 1))],
        [sg.Text("Improve Threshold (%):", size=(20, 1)), sg.InputText("0.01", key="improve_thresh", size=(10, 1))],
        [sg.Text("Min Generations:", size=(20, 1)), sg.InputText("100", key="min_generations", size=(10, 1))],
        [sg.HorizontalSeparator()],
        [
            sg.Button("Start", size=(10, 1), key="start_btn"),
            sg.Button("Pause", size=(10, 1), key="pause_btn", disabled=True),
            sg.Button("Resume", size=(10, 1), key="resume_btn", disabled=True),
            sg.Button("Stop", size=(10, 1), key="stop_btn", disabled=True),
        ],
        [sg.HorizontalSeparator()],
        [sg.Text("Status:", font=("Arial", 10, "bold"))],
        [sg.Multiline(size=(45, 5), key="status_output", disabled=True, autoscroll=True)],
    ]

    # --- Metrics Panel ---
    metrics_col = [
        [sg.Text("Current Metrics", font=("Arial", 14, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Text("Generation:", size=(20, 1)), sg.Text("0", key="gen_text", size=(15, 1))],
        [sg.Text("Best Fitness:", size=(20, 1)), sg.Text("--", key="best_text", size=(15, 1))],
        [sg.Text("Avg Fitness:", size=(20, 1)), sg.Text("--", key="avg_text", size=(15, 1))],
        [sg.Text("Worst Fitness:", size=(20, 1)), sg.Text("--", key="worst_text", size=(15, 1))],
        [sg.Text("Improvement:", size=(20, 1)), sg.Text("--", key="improvement_text", size=(15, 1))],
        [sg.Text("Mutation Rate:", size=(20, 1)), sg.Text("--", key="mut_rate_text", size=(15, 1))],
        [sg.HorizontalSeparator()],
        [sg.Text("Best Schedule", font=("Arial", 10, "bold"))],
        [sg.Multiline(size=(40, 10), key="schedule_output", disabled=True)],
    ]

    # --- Plot Panel ---
    plot_col = [[sg.Canvas(key="plot_canvas", size=(500, 400))]]

    # --- Main Layout ---
    layout = [
        [
            sg.Column(control_col, vertical_alignment="top"),
            sg.Column(metrics_col, vertical_alignment="top"),
            sg.Column(plot_col, vertical_alignment="top"),
        ]
    ]

    return sg.Window("GA Scheduler GUI", layout, finalize=True, size=(1400, 700))


def draw_fitness_plot(canvas_elem, history: dict, figure: Figure = None):
    """Draw fitness plot on matplotlib canvas."""
    if figure is None:
        figure = Figure(figsize=(5, 4), dpi=100)

    figure.clear()
    ax = figure.add_subplot(111)

    if history["best"]:
        gens = list(range(1, len(history["best"]) + 1))
        ax.plot(gens, history["best"], label="Best", linewidth=2)
        ax.plot(gens, history["avg"], label="Average", linewidth=2)
        ax.plot(gens, history["worst"], label="Worst", linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Over Generations")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if canvas_elem and hasattr(canvas_elem, "get_tk_widget"):
        canvas_elem.get_tk_widget().forget()

    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas_elem.TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)

    return figure_canvas_agg, figure


def main():
    """Main GUI event loop."""
    window = create_window(None)

    runner = None
    ga_thread = None
    output_queue = queue.Queue()
    param_queue = queue.Queue()
    figure = None
    figure_agg = None
    paused = False
    status_log = []

    def log_status(msg: str):
        """Add message to status log."""
        status_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        window["status_output"].update("\n".join(status_log[-20:]))  # Keep last 20 lines

    log_status("Ready to start GA")

    while True:
        event, values = window.read(timeout=100)

        if event == sg.WINDOW_CLOSED:
            if runner and runner.running:
                runner.running = False
                if ga_thread:
                    ga_thread.join(timeout=2)
            break

        # ━━━ Start Button ━━━
        if event == "start_btn":
            try:
                params = {
                    "pop_size": int(values["pop_size"]),
                    "init_mutation": float(values["init_mutation"]),
                    "improve_thresh": float(values["improve_thresh"]),
                    "min_generations": int(values["min_generations"]),
                    "halve_after": 20,
                    "print_every": 10,
                }
                runner = GARunner(params)
                runner.running = True
                log_status(f"Started GA (pop={params['pop_size']}, mut={params['init_mutation']})")

                # Start GA thread
                ga_thread = threading.Thread(target=ga_thread_worker, args=(runner, param_queue, output_queue), daemon=True)
                ga_thread.start()

                # Update UI state
                window["start_btn"].update(disabled=True)
                window["pause_btn"].update(disabled=False)
                window["stop_btn"].update(disabled=False)
                window["pop_size"].update(disabled=True)
                window["init_mutation"].update(disabled=True)
                paused = False

            except ValueError as e:
                log_status(f"Error: Invalid parameters - {e}")

        # ━━━ Pause Button ━━━
        if event == "pause_btn" and runner:
            runner.running = False
            paused = True
            log_status("Paused GA")
            window["pause_btn"].update(disabled=True)
            window["resume_btn"].update(disabled=False)

        # ━━━ Resume Button ━━━
        if event == "resume_btn" and runner:
            runner.running = True
            paused = False
            log_status("Resumed GA")
            window["resume_btn"].update(disabled=True)
            window["pause_btn"].update(disabled=False)

        # ━━━ Stop Button ━━━
        if event == "stop_btn" and runner:
            runner.running = False
            log_status("Stopped GA")
            window["pause_btn"].update(disabled=True)
            window["resume_btn"].update(disabled=True)
            window["stop_btn"].update(disabled=True)
            window["start_btn"].update(disabled=False)
            window["pop_size"].update(disabled=False)
            window["init_mutation"].update(disabled=False)

        # ━━━ Mutation Rate Slider (Real-time) ━━━
        if runner and runner.running:
            new_mut = float(values["mutation_rate"])
            param_queue.put(new_mut)

        # ━━━ Process GA Output ━━━
        try:
            metrics = output_queue.get_nowait()
            if metrics is None:  # Completion signal
                log_status("GA completed!")
                window["pause_btn"].update(disabled=True)
                window["resume_btn"].update(disabled=True)
                window["stop_btn"].update(disabled=True)
                window["start_btn"].update(disabled=False)
                window["pop_size"].update(disabled=False)
                window["init_mutation"].update(disabled=False)
                runner.running = False
            else:
                # Update metrics display
                window["gen_text"].update(f"{metrics['gen']}")
                window["best_text"].update(f"{metrics['best']:.4f}")
                window["avg_text"].update(f"{metrics['avg']:.4f}")
                window["worst_text"].update(f"{metrics['worst']:.4f}")
                window["improvement_text"].update(f"{metrics['improvement']:+.2f}%")
                window["mutation_rate"].update(metrics["mutation_rate"])
                window["mut_rate_text"].update(f"{metrics['mutation_rate']:.6f}")

                # Update best schedule display
                schedule, fitness = runner.get_best_schedule()
                schedule_text = format_schedule(schedule)
                window["schedule_output"].update(schedule_text)

                # Update plot
                if metrics["gen"] % 5 == 0:  # Update plot every 5 generations
                    figure_agg, figure = draw_fitness_plot(window["plot_canvas"], runner.history, figure)

        except queue.Empty:
            pass

    window.close()


if __name__ == "__main__":
    main()
