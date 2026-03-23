"""Evaluation tab UI and diagram rendering utilities."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


def create_evaluation_tab(notebook: ttk.Notebook) -> dict[str, object]:
	"""Create the Evaluation tab and return rendering widgets."""
	eval_tab = ttk.Frame(notebook)
	notebook.add(eval_tab, text="Evaluation")

	eval_split = ttk.PanedWindow(eval_tab, orient=tk.HORIZONTAL)
	eval_split.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

	table_frame = ttk.Frame(eval_split)
	chart_frame = ttk.Frame(eval_split)
	eval_split.add(table_frame, weight=3)
	eval_split.add(chart_frame, weight=4)

	columns = ("algo", "runtime", "best", "success", "residual")
	eval_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
	eval_tree.heading("algo", text="Algorithm")
	eval_tree.heading("runtime", text="Runtime (s)")
	eval_tree.heading("best", text="Best Cost")
	eval_tree.heading("success", text="Success %")
	eval_tree.heading("residual", text="Residual")
	eval_tree.column("algo", width=180, anchor=tk.W)
	eval_tree.column("runtime", width=110, anchor=tk.E)
	eval_tree.column("best", width=110, anchor=tk.E)
	eval_tree.column("success", width=90, anchor=tk.E)
	eval_tree.column("residual", width=90, anchor=tk.E)
	eval_tree.pack(fill=tk.BOTH, expand=True)

	chart_canvas = tk.Canvas(chart_frame, bg="#0f172a", highlightthickness=0)
	chart_scroll = ttk.Scrollbar(chart_frame, orient=tk.VERTICAL, command=chart_canvas.yview)
	chart_canvas.configure(yscrollcommand=chart_scroll.set)
	chart_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	chart_scroll.pack(side=tk.RIGHT, fill=tk.Y)

	eval_widgets: dict[str, object] = {"tree": eval_tree, "canvas": chart_canvas, "metrics": None}

	def _on_resize(_event: tk.Event) -> None:
		metrics = eval_widgets.get("metrics")
		if isinstance(metrics, dict) and metrics:
			_draw_chart(chart_canvas, metrics)

	chart_canvas.bind("<Configure>", _on_resize)

	def _on_mousewheel(event: tk.Event) -> None:
		if event.delta:
			chart_canvas.yview_scroll(int(-event.delta / 120), "units")

	chart_canvas.bind("<Button-4>", lambda _e: chart_canvas.yview_scroll(-1, "units"))
	chart_canvas.bind("<Button-5>", lambda _e: chart_canvas.yview_scroll(1, "units"))
	chart_canvas.bind("<MouseWheel>", _on_mousewheel)

	return eval_widgets


def clear_evaluation_tab(eval_widgets: dict[str, object]) -> None:
	"""Clear table and chart content."""
	tree = eval_widgets.get("tree")
	canvas = eval_widgets.get("canvas")

	if isinstance(tree, ttk.Treeview):
		for item in tree.get_children():
			tree.delete(item)

	if isinstance(canvas, tk.Canvas):
		canvas.delete("all")
		canvas.configure(scrollregion=(0, 0, 0, 0))

	eval_widgets["metrics"] = None


def _draw_metric_panel(
	canvas: tk.Canvas,
	panel_x: int,
	panel_y: int,
	panel_w: int,
	panel_h: int,
	label: str,
	key: str,
	metrics: dict[str, dict[str, float]],
	algos: list[str],
	colors: dict[str, str],
) -> None:
	"""Draw one metric comparison panel with horizontal bars."""
	canvas.create_rectangle(panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, outline="#334155")
	canvas.create_text(
		panel_x + 10,
		panel_y + 12,
		anchor=tk.W,
		fill="#e2e8f0",
		font=("Segoe UI", 9, "bold"),
		text=label,
	)

	values = np.array([float(metrics[a][key]) for a in algos], dtype=float)
	vmax = float(np.max(values)) if values.size else 1.0
	if vmax <= 0:
		vmax = 1.0

	left_pad = 108
	right_pad = 16
	top_pad = 32
	row_h = max(18, (panel_h - top_pad - 8) // max(1, len(algos)))
	bar_h = max(8, row_h - 8)
	usable_w = max(40, panel_w - left_pad - right_pad)

	for i, algo in enumerate(algos):
		y = panel_y + top_pad + i * row_h
		v = float(metrics[algo][key])
		bar_w = max(1, int((v / vmax) * usable_w))

		canvas.create_text(
			panel_x + 8,
			y + bar_h / 2,
			anchor=tk.W,
			fill="#cbd5e1",
			font=("Segoe UI", 8),
			text=algo,
		)
		canvas.create_rectangle(
			panel_x + left_pad,
			y,
			panel_x + left_pad + bar_w,
			y + bar_h,
			fill=colors.get(algo, "#cbd5e1"),
			outline="",
		)
		canvas.create_text(
			panel_x + left_pad + bar_w + 4,
			y + bar_h / 2,
			anchor=tk.W,
			fill="#e2e8f0",
			font=("Segoe UI", 8),
			text=f"{v:.4g}",
		)


def _draw_chart(canvas: tk.Canvas, metrics: dict[str, dict[str, float]]) -> None:
	"""Draw all evaluation metric charts as vertically stacked panels."""
	canvas.delete("all")

	w = max(760, canvas.winfo_width())

	canvas.create_text(
		14,
		14,
		anchor=tk.W,
		fill="#e2e8f0",
		font=("Segoe UI", 10, "bold"),
		text="Evaluation Charts (Comparison Diagram)",
	)

	algos = list(metrics.keys())
	colors = {
		"Hamiltonian Energy": "#60a5fa",
		"Simulated Annealing": "#34d399",
		"Genetic Algorithm": "#f59e0b",
		"Tabu Search": "#f87171",
	}

	panels = [
		("Runtime (s)", "time_taken"),
		("Computational Cost", "computational_cost"),
		("Success Probability", "success_probability"),
		("Standard Deviation", "standard_deviation"),
	]

	margin = 12
	top = 34
	gap = 10
	panel_w = w - 2 * margin
	panel_h = 148

	for idx, (label, key) in enumerate(panels):
		px = margin
		py = top + idx * (panel_h + gap)
		_draw_metric_panel(
			canvas=canvas,
			panel_x=px,
			panel_y=py,
			panel_w=panel_w,
			panel_h=panel_h,
			label=label,
			key=key,
			metrics=metrics,
			algos=algos,
			colors=colors,
		)

	content_h = top + len(panels) * (panel_h + gap) + margin
	canvas.configure(scrollregion=(0, 0, w, content_h))


def render_evaluation(metrics: dict[str, dict[str, float]], eval_widgets: dict[str, object]) -> None:
	"""Render evaluation table and grouped bar-chart diagram."""
	tree = eval_widgets.get("tree")
	canvas = eval_widgets.get("canvas")
	if not isinstance(tree, ttk.Treeview) or not isinstance(canvas, tk.Canvas):
		return

	clear_evaluation_tab(eval_widgets)

	for algo, m in metrics.items():
		tree.insert(
			"",
			tk.END,
			values=(
				algo,
				f"{m['time_taken']:.4f}",
				f"{m['best_energy']:.4f}",
				f"{100.0 * m['success_probability']:.1f}%",
				f"{m['residual_energy']:.4f}",
			),
		)

	eval_widgets["metrics"] = metrics
	_draw_chart(canvas, metrics)
