"""
Ising Model GUI
===============
Interactive visualiser for the Ising model Hamiltonian.
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np

from hemiltonian_energy import hamiltonian_vectorized
from calculation import run_all_calculations_bundle
from evaluation import create_evaluation_tab, clear_evaluation_tab, render_evaluation

BG_CANVAS = "#0f172a"
COL_UP = "#ef4444"
COL_DOWN = "#3b82f6"
COL_LINE = "#64748b"
COL_LABEL = "#94a3b8"
COL_VAL = "#f1f5f9"


class IsingGUI:

    RADIUS_MAX = 32
    RADIUS_MIN = 8

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Ising Model — Hamiltonian Visualiser")
        self.root.configure(bg="#1e293b")

        self.N = 6
        self.spins: list[int] = [1] * self.N
        self._custom_J: np.ndarray | None = None
        self._custom_h: np.ndarray | None = None

        self._build_ui()
        self.root.after(50, self._refresh)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabelframe", background="#1e293b", foreground="#e2e8f0")
        style.configure("TLabelframe.Label", background="#1e293b", foreground="#7dd3fc", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#1e293b", foreground="#e2e8f0")
        style.configure("TButton", padding=4)
        style.configure("TEntry", fieldbackground="#334155", foreground="#f8fafc")
        style.configure("TSpinbox", fieldbackground="#334155", foreground="#f8fafc")

        top_wrap = ttk.Frame(self.root)
        top_wrap.pack(fill=tk.X, padx=10, pady=(10, 4))

        top = ttk.LabelFrame(top_wrap, text="Parameters", padding=8)
        top.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        conn = ttk.LabelFrame(top_wrap, text="Connection of Spins", padding=8)
        conn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(top, text="Spins  N:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self._n_var = tk.IntVar(value=self.N)
        ttk.Spinbox(top, from_=2, to=30, textvariable=self._n_var, width=5, command=self._apply_n).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Apply N", command=self._apply_n).grid(row=0, column=2, padx=8)

        ttk.Label(top, text="Coupling  J:").grid(row=1, column=0, sticky=tk.W, padx=4)
        self._j_var = tk.StringVar(value="1.0")
        ttk.Entry(top, textvariable=self._j_var, width=8).grid(row=1, column=1, padx=4)
        ttk.Button(top, text="Custom J matrix…", command=self._open_custom_J).grid(row=1, column=2, padx=8)

        ttk.Label(top, text="Field  h:").grid(row=2, column=0, sticky=tk.W, padx=4)
        self._h_var = tk.StringVar(value="0.5")
        ttk.Entry(top, textvariable=self._h_var, width=8).grid(row=2, column=1, padx=4)
        ttk.Button(top, text="Custom h vector…", command=self._open_custom_h).grid(row=2, column=2, padx=8)

        self._topology_var = tk.StringVar(value="Custom")
        self._topology_var.trace_add("write", lambda *_: self._on_topology_change())
        ttk.Label(conn, text="Topology:").grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Radiobutton(conn, text="Custom", value="Custom", variable=self._topology_var).grid(row=0, column=1, sticky=tk.W, padx=2)
        ttk.Radiobutton(conn, text="None", value="None (disconnected)", variable=self._topology_var).grid(row=0, column=2, sticky=tk.W, padx=2)
        ttk.Radiobutton(conn, text="Chain", value="Chain (1D)", variable=self._topology_var).grid(row=0, column=3, sticky=tk.W, padx=2)
        ttk.Radiobutton(conn, text="Full", value="Fully Connected", variable=self._topology_var).grid(row=0, column=4, sticky=tk.W, padx=2)

        ttk.Label(conn, text="sᵢ:").grid(row=1, column=0, sticky=tk.W, padx=(2, 2), pady=(8, 0))
        self._pair_i_var = tk.IntVar(value=1)
        self._pair_i_spin = ttk.Spinbox(conn, from_=1, to=30, textvariable=self._pair_i_var, width=5)
        self._pair_i_spin.grid(row=1, column=1, sticky=tk.W, padx=2, pady=(8, 0))
        ttk.Label(conn, text="sⱼ:").grid(row=1, column=2, sticky=tk.W, padx=(6, 2), pady=(8, 0))
        self._pair_j_var = tk.IntVar(value=5)
        self._pair_j_spin = ttk.Spinbox(conn, from_=1, to=30, textvariable=self._pair_j_var, width=5)
        self._pair_j_spin.grid(row=1, column=3, sticky=tk.W, padx=2, pady=(8, 0))
        ttk.Label(conn, text="J:").grid(row=2, column=0, sticky=tk.W, padx=(2, 2), pady=(8, 0))
        self._pair_j_strength = tk.StringVar(value="1.0")
        self._pair_j_entry = ttk.Entry(conn, textvariable=self._pair_j_strength, width=8)
        self._pair_j_entry.grid(row=2, column=1, sticky=tk.W, padx=2, pady=(8, 0))
        self._connect_btn = ttk.Button(conn, text="Connect", command=self._connect_pair)
        self._connect_btn.grid(row=2, column=2, sticky=tk.W, padx=4, pady=(8, 0))
        self._disconnect_btn = ttk.Button(conn, text="Disconnect", command=self._disconnect_pair)
        self._disconnect_btn.grid(row=2, column=3, sticky=tk.W, padx=4, pady=(8, 0))

        self._manual_controls = [
            self._pair_i_spin,
            self._pair_j_spin,
            self._pair_j_entry,
            self._connect_btn,
            self._disconnect_btn,
        ]

        btn_row = ttk.Frame(top)
        btn_row.grid(row=4, column=0, columnspan=6, pady=(6, 0))
        ttk.Button(btn_row, text="All  +1", command=lambda: self._set_all(1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="All  −1", command=lambda: self._set_all(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Alternate", command=self._set_alternating).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Random", command=self._set_random).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Calculations", command=self._open_calculations_window).pack(side=tk.LEFT, padx=8)

        self._update_manual_connection_state()

        cf = ttk.LabelFrame(self.root, text="Spin Configuration  —  click a spin to flip it", padding=4)
        cf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._canvas = tk.Canvas(cf, bg=BG_CANVAS, height=300, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Configure>", lambda _e: self._refresh())

    def _set_all(self, val: int) -> None:
        self.spins = [val] * self.N
        self._refresh()

    def _set_alternating(self) -> None:
        self.spins = [1 if i % 2 == 0 else -1 for i in range(self.N)]
        self._refresh()

    def _set_random(self) -> None:
        self.spins = list(np.random.choice([1, -1], size=self.N))
        self._refresh()

    def _apply_n(self) -> None:
        n = int(self._n_var.get())
        if n < 2:
            messagebox.showerror("Invalid N", "N must be at least 2.")
            return
        if n > self.N:
            self.spins += [1] * (n - self.N)
        else:
            self.spins = self.spins[:n]
        self.N = n
        self._custom_J = None
        self._custom_h = None
        self._refresh()

    def _on_topology_change(self) -> None:
        self._update_manual_connection_state()
        self._refresh()

    def _update_manual_connection_state(self) -> None:
        enabled = self._topology_var.get() == "Custom"
        for widget in self._manual_controls:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])

    def _connect_pair(self) -> None:
        if self._topology_var.get() != "Custom":
            messagebox.showinfo("Manual edit disabled", "Switch topology to 'Custom' to edit pair connections manually.")
            return
        try:
            i = int(self._pair_i_var.get()) - 1
            j = int(self._pair_j_var.get()) - 1
            jval = float(self._pair_j_strength.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        if not (0 <= i < self.N and 0 <= j < self.N):
            messagebox.showerror("Out of range", f"Spin numbers must be between 1 and {self.N}.")
            return
        if i == j:
            messagebox.showerror("Invalid pair", "Please choose two different spins.")
            return

        if self._custom_J is None:
            self._custom_J = self._build_J().copy()

        lo, hi = (i, j) if i < j else (j, i)
        self._custom_J[lo, hi] = jval
        self._refresh()
