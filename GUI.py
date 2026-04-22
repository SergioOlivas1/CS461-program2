"""
Interactive GA Scheduler GUI using PySimpleGUI + Matplotlib
Supports runtime parameter tuning, live plotting, and real-time schedule display.
"""
 
try:
    import PySimpleGUI as sg
except ImportError:
    import FreeSimpleGUI as sg
 
import matplotlib
matplotlib.use("TkAgg")
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
import ga as ga_module
from ga import init_population, next_generation
from output import format_schedule, compute_violations
 
# ── Configuration ──────────────────────────────────────────────────────
sg.theme("DarkBlue2")
 
DEFAULT_PARAMS = {
    "pop_size": 500,
    "min_generations": 100,
    "improve_thresh": 0.01,
    "init_mutation": 0.01,
    "halve_after": 20,
}
 
 
class GARunner:
    """Encapsulates GA execution with pause/resume/stop capability."""
 
    def __init__(self, params: dict):
        self.params = params
        self.running = False
        self.population = None
        self.fitness = None
        self.generation = 0
        self.history = {"best": [], "avg": [], "worst": []}
        self.mutation_rate = params["init_mutation"]
 
    def initialize(self):
        """Create initial population with a fresh RNG each run."""
        ga_module.rng = np.random.default_rng(np.random.PCG64DXSM())
        self.population = init_population(self.params["pop_size"])
        self.fitness = score_population(self.population)
        self.generation = 0
        self.history = {"best": [], "avg": [], "worst": []}
        self.mutation_rate = self.params["init_mutation"]
 
    def step(self, mutation_rate_override: float = None) -> dict:
        """Perform one generation step. Returns metrics dict."""
        if mutation_rate_override is not None:
            self.mutation_rate = mutation_rate_override
 
        self.generation += 1
        combined, offspring = next_generation(
            self.population, self.fitness, self.mutation_rate
        )
        off_fitness = score_population(offspring)
 
        combined_fitness = np.concatenate([self.fitness, off_fitness])
        top_idx = np.argsort(combined_fitness)[-self.params["pop_size"]:]
        self.population = combined[top_idx]
        self.fitness = combined_fitness[top_idx]
 
        best_f  = float(np.max(self.fitness))
        avg_f   = float(np.mean(self.fitness))
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
        """Return (schedule, fitness) for best individual."""
        best_idx = int(np.argmax(self.fitness))
        return self.population[best_idx], self.fitness[best_idx]
 
    def should_stop(self, improvement: float) -> bool:
        """Check stopping criteria."""
        if self.generation < self.params["min_generations"]:
            return False
        if self.generation >= 2000:
            return True
        return abs(improvement) < self.params["improve_thresh"]
 
 
def ga_thread_worker(runner: GARunner, param_queue: queue.Queue, output_queue: queue.Queue):
    """Run GA in background thread; listen for live parameter updates."""
    runner.initialize()
    prev_avg = float(np.mean(runner.fitness))
    low_improve_ct = 0
 
    while runner.running:
        # Drain param queue — take only the most recent value
        new_mutation = None
        while True:
            try:
                new_mutation = param_queue.get_nowait()
            except queue.Empty:
                break
        if new_mutation is not None:
            runner.mutation_rate = new_mutation
 
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
 
        if runner.should_stop(improvement):
            break
 
        prev_avg = avg_f
        time.sleep(0.01)
 
    runner.running = False
    output_queue.put(None)  # Signal completion
 
 
# ── Violation summary helper ────────────────────────────────────────────
def violations_text(schedule: np.ndarray) -> str:
    v = compute_violations(schedule)
    lines = []
    for k, val in v.items():
        label = k.replace("_", " ").title()
        status = "✓" if val == 0 else f"✗ {val}"
        lines.append(f"{label:<35} {status}")
    return "\n".join(lines)
 
 
# ── Plot drawing ────────────────────────────────────────────────────────
def draw_fitness_plot(canvas_elem, history: dict, figure_agg=None, figure: Figure = None):
    """Redraw fitness chart. Reuses existing figure to avoid memory leaks."""
    if figure is None:
        figure = Figure(figsize=(5, 4), dpi=100)
 
    figure.clear()
    ax = figure.add_subplot(111)
 
    if history["best"]:
        gens = list(range(1, len(history["best"]) + 1))
        ax.plot(gens, history["best"],  label="Best",    linewidth=2, color="#4CAF50")
        ax.plot(gens, history["avg"],   label="Average", linewidth=2, color="#2196F3")
        ax.plot(gens, history["worst"], label="Worst",   linewidth=2, color="#F44336")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Over Generations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        figure.tight_layout()
 
    # Only create a new FigureCanvasTkAgg on first call
    if figure_agg is None:
        figure_agg = FigureCanvasTkAgg(figure, master=canvas_elem.TKCanvas)
        figure_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
 
    figure_agg.draw()
    return figure_agg, figure
 
 
# ── Window layout ───────────────────────────────────────────────────────
def create_window():
    control_col = [
        [sg.Text("GA Scheduler", font=("Arial", 14, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Text("Population Size:",      size=(22, 1)), sg.InputText("500",  key="pop_size",        size=(10, 1))],
        [sg.Text("Initial Mutation Rate:",size=(22, 1)), sg.InputText("0.01", key="init_mutation",   size=(10, 1))],
        [sg.Text("Improve Threshold (%):",size=(22, 1)), sg.InputText("0.01", key="improve_thresh",  size=(10, 1))],
        [sg.Text("Min Generations:",      size=(22, 1)), sg.InputText("100",  key="min_generations", size=(10, 1))],
        [sg.HorizontalSeparator()],
        [sg.Text("Live Mutation Rate:", size=(22, 1))],
        [sg.Slider((0.0001, 0.1), default_value=0.01, resolution=0.0001,
                   key="mutation_slider", orientation="h", size=(30, 15),
                   enable_events=True, disabled=True)],
        [sg.HorizontalSeparator()],
        [
            sg.Button("Start",  size=(9, 1), key="start_btn"),
            sg.Button("Pause",  size=(9, 1), key="pause_btn",  disabled=True),
            sg.Button("Resume", size=(9, 1), key="resume_btn", disabled=True),
            sg.Button("Stop",   size=(9, 1), key="stop_btn",   disabled=True),
        ],
        [sg.Button("Save Schedule", size=(20, 1), key="save_btn", disabled=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("Status:", font=("Arial", 10, "bold"))],
        [sg.Multiline(size=(42, 6), key="status_output", disabled=True,
                      autoscroll=True, font=("Courier", 9))],
    ]
 
    metrics_col = [
        [sg.Text("Current Metrics", font=("Arial", 14, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Text("Generation:",   size=(20, 1)), sg.Text("0",  key="gen_text",         size=(15, 1), font=("Courier", 10))],
        [sg.Text("Best Fitness:", size=(20, 1)), sg.Text("--", key="best_text",         size=(15, 1), font=("Courier", 10))],
        [sg.Text("Avg Fitness:",  size=(20, 1)), sg.Text("--", key="avg_text",          size=(15, 1), font=("Courier", 10))],
        [sg.Text("Worst Fitness:",size=(20, 1)), sg.Text("--", key="worst_text",        size=(15, 1), font=("Courier", 10))],
        [sg.Text("Improvement:",  size=(20, 1)), sg.Text("--", key="improvement_text",  size=(15, 1), font=("Courier", 10))],
        [sg.Text("Mutation Rate:",size=(20, 1)), sg.Text("--", key="mut_rate_text",     size=(15, 1), font=("Courier", 10))],
        [sg.HorizontalSeparator()],
        [sg.Text("Best Schedule", font=("Arial", 10, "bold"))],
        [sg.Multiline(size=(42, 10), key="schedule_output", disabled=True,
                      font=("Courier", 9))],
        [sg.HorizontalSeparator()],
        [sg.Text("Constraint Violations", font=("Arial", 10, "bold"))],
        [sg.Multiline(size=(42, 8), key="violations_output", disabled=True,
                      font=("Courier", 9))],
    ]
 
    plot_col = [
        [sg.Canvas(key="plot_canvas", size=(520, 420))],
    ]
 
    layout = [
        [
            sg.Column(control_col, vertical_alignment="top"),
            sg.VSeparator(),
            sg.Column(metrics_col, vertical_alignment="top", scrollable=False),
            sg.VSeparator(),
            sg.Column(plot_col, vertical_alignment="top"),
        ]
    ]
 
    return sg.Window("GA Scheduler GUI", layout, finalize=True, size=(1500, 780))
 
 
# ── Main event loop ─────────────────────────────────────────────────────
def main():
    window = create_window()
 
    runner      = None
    ga_thread   = None
    output_queue = queue.Queue()
    param_queue  = queue.Queue()
    figure      = None
    figure_agg  = None
    status_log  = []
 
    def log_status(msg: str):
        status_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        window["status_output"].update("\n".join(status_log[-30:]))
 
    def reset_ui_to_idle():
        window["start_btn"].update(disabled=False)
        window["pause_btn"].update(disabled=True)
        window["resume_btn"].update(disabled=True)
        window["stop_btn"].update(disabled=True)
        window["mutation_slider"].update(disabled=True)
        window["pop_size"].update(disabled=False)
        window["init_mutation"].update(disabled=False)
 
    log_status("Ready — configure parameters and press Start.")
 
    # Draw empty plot on launch
    figure_agg, figure = draw_fitness_plot(window["plot_canvas"], {"best": [], "avg": [], "worst": []})
 
    while True:
        event, values = window.read(timeout=100)
 
        if event == sg.WINDOW_CLOSED:
            if runner and runner.running:
                runner.running = False
                if ga_thread:
                    ga_thread.join(timeout=2)
            break
 
        # ── Start ──────────────────────────────────────────────────────
        if event == "start_btn":
            try:
                params = {
                    "pop_size":        int(values["pop_size"]),
                    "init_mutation":   float(values["init_mutation"]),
                    "improve_thresh":  float(values["improve_thresh"]),
                    "min_generations": int(values["min_generations"]),
                    "halve_after":     20,
                }
                # Validate
                if params["pop_size"] < 10:
                    raise ValueError("Population size must be at least 10")
                if not (0 < params["init_mutation"] <= 1):
                    raise ValueError("Mutation rate must be between 0 and 1")
 
                # Clear queues from any previous run
                while not output_queue.empty():
                    output_queue.get_nowait()
                while not param_queue.empty():
                    param_queue.get_nowait()
 
                # Reset plot
                figure_agg, figure = draw_fitness_plot(
                    window["plot_canvas"],
                    {"best": [], "avg": [], "worst": []},
                    figure_agg, figure
                )
 
                runner = GARunner(params)
                runner.running = True
                window["mutation_slider"].update(
                    value=params["init_mutation"], disabled=False
                )
                log_status(f"Started — pop={params['pop_size']}, mut={params['init_mutation']}")
 
                ga_thread = threading.Thread(
                    target=ga_thread_worker,
                    args=(runner, param_queue, output_queue),
                    daemon=True
                )
                ga_thread.start()
 
                window["start_btn"].update(disabled=True)
                window["pause_btn"].update(disabled=False)
                window["stop_btn"].update(disabled=False)
                window["save_btn"].update(disabled=True)
                window["pop_size"].update(disabled=True)
                window["init_mutation"].update(disabled=True)
 
            except ValueError as e:
                log_status(f"Error: {e}")
 
        # ── Pause ──────────────────────────────────────────────────────
        if event == "pause_btn" and runner:
            runner.running = False
            log_status("Paused")
            window["pause_btn"].update(disabled=True)
            window["resume_btn"].update(disabled=False)
            window["mutation_slider"].update(disabled=True)
 
        # ── Resume ─────────────────────────────────────────────────────
        if event == "resume_btn" and runner:
            runner.running = True
            log_status("Resumed")
            # Restart thread since old one exited on pause
            ga_thread = threading.Thread(
                target=ga_thread_worker,
                args=(runner, param_queue, output_queue),
                daemon=True
            )
            ga_thread.start()
            window["resume_btn"].update(disabled=True)
            window["pause_btn"].update(disabled=False)
            window["mutation_slider"].update(disabled=False)
 
        # ── Stop ───────────────────────────────────────────────────────
        if event == "stop_btn" and runner:
            runner.running = False
            log_status("Stopped by user")
            reset_ui_to_idle()
            if runner.fitness is not None:
                window["save_btn"].update(disabled=False)
 
        # ── Save Schedule ──────────────────────────────────────────────
        if event == "save_btn" and runner:
            schedule, fit = runner.get_best_schedule()
            path = "schedule.txt"
            from output import save_schedule
            save_schedule(schedule, fit, path)
            log_status(f"Schedule saved to {path}")
 
        # ── Live mutation slider → push to GA thread ───────────────────
        if event == "mutation_slider" and runner and runner.running:
            param_queue.put(float(values["mutation_slider"]))
 
        # ── Process GA output ──────────────────────────────────────────
        try:
            metrics = output_queue.get_nowait()
 
            if metrics is None:  # GA finished naturally
                log_status(f"GA completed at generation {runner.generation}!")
                reset_ui_to_idle()
                if runner.fitness is not None:
                    window["save_btn"].update(disabled=False)
                # Final plot update
                figure_agg, figure = draw_fitness_plot(
                    window["plot_canvas"], runner.history, figure_agg, figure
                )
 
            else:
                # Update metric labels
                window["gen_text"].update(str(metrics["gen"]))
                window["best_text"].update(f"{metrics['best']:.4f}")
                window["avg_text"].update(f"{metrics['avg']:.4f}")
                window["worst_text"].update(f"{metrics['worst']:.4f}")
                window["improvement_text"].update(f"{metrics['improvement']:+.3f}%")
                window["mut_rate_text"].update(f"{metrics['mutation_rate']:.6f}")
 
                # Sync slider to auto-adjusted mutation rate (without triggering event)
                window["mutation_slider"].update(value=metrics["mutation_rate"])
 
                # Update schedule + violations every 5 gens (expensive text ops)
                if metrics["gen"] % 5 == 0 or metrics["gen"] == 1:
                    schedule, fit = runner.get_best_schedule()
                    window["schedule_output"].update(format_schedule(schedule))
                    window["violations_output"].update(violations_text(schedule))
 
                # Update plot every 5 gens
                if metrics["gen"] % 5 == 0:
                    figure_agg, figure = draw_fitness_plot(
                        window["plot_canvas"], runner.history, figure_agg, figure
                    )
 
                # Log mutation halving events
                if len(runner.history["best"]) > 1:
                    prev_mut = runner.history["best"][-2] if len(runner.history["best"]) > 1 else None
                    if metrics["mutation_rate"] < (runner.params["init_mutation"] * 0.99):
                        pass  # Already logged via adaptive logic in thread
 
        except queue.Empty:
            pass
 
    window.close()
 
 
if __name__ == "__main__":
    main()
