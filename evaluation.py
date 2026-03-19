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
	eval_tree.heading("best", text="Best Energy")
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
