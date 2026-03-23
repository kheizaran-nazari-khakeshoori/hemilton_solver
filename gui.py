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

    def _disconnect_pair(self) -> None:
        if self._topology_var.get() != "Custom":
            messagebox.showinfo("Manual edit disabled", "Switch topology to 'Custom' to edit pair connections manually.")
            return
        try:
            i = int(self._pair_i_var.get()) - 1
            j = int(self._pair_j_var.get()) - 1
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        if not (0 <= i < self.N and 0 <= j < self.N) or i == j:
            messagebox.showerror("Invalid pair", "Choose two valid different spins.")
            return

        if self._custom_J is None:
            self._custom_J = self._build_J().copy()

        lo, hi = (i, j) if i < j else (j, i)
        self._custom_J[lo, hi] = 0.0
        self._refresh()

    def _open_custom_J(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Custom Coupling Matrix J  (upper triangle, i < j)")
        dlg.configure(bg="#1e293b")
        dlg.resizable(False, False)

        instruction = (
            f"Enter the {self.N}×{self.N} coupling matrix J.\n"
            "Each row is space-separated. Only the upper triangle (i<j) is used.\n"
            "Example for N=3:\n"
            "  0  1.0  0.5\n"
            "  0  0    2.0\n"
            "  0  0    0"
        )
        ttk.Label(dlg, text=instruction, font=("Courier New", 9), padding=8).pack()

        default = self._matrix_to_str(self._build_J())
        text = scrolledtext.ScrolledText(
            dlg,
            width=40,
            height=self.N + 2,
            font=("Courier New", 10),
            bg="#334155",
            fg="#f8fafc",
            insertbackground="white",
        )
        text.insert("1.0", default)
        text.pack(padx=10, pady=4)

        def _apply() -> None:
            try:
                rows = text.get("1.0", tk.END).strip().splitlines()
                mat = np.array([[float(v) for v in r.split()] for r in rows])
                if mat.shape != (self.N, self.N):
                    raise ValueError(f"Expected {self.N}×{self.N}, got {mat.shape}")
                self._custom_J = mat
                dlg.destroy()
                self._refresh()
            except Exception as exc:
                messagebox.showerror("Invalid input", str(exc), parent=dlg)

        ttk.Button(dlg, text="Apply", command=_apply).pack(pady=6)

    def _open_custom_h(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Custom Field Vector h")
        dlg.configure(bg="#1e293b")
        dlg.resizable(False, False)

        ttk.Label(dlg, text=f"Enter {self.N} space-separated values for h:", padding=8).pack()

        default = " ".join(str(v) for v in self._build_h())
        entry = ttk.Entry(dlg, width=40, font=("Courier New", 10))
        entry.insert(0, default)
        entry.pack(padx=10, pady=4)

        def _apply() -> None:
            try:
                vec = np.array([float(v) for v in entry.get().split()])
                if vec.shape != (self.N,):
                    raise ValueError(f"Expected {self.N} values, got {len(vec)}")
                self._custom_h = vec
                dlg.destroy()
                self._refresh()
            except Exception as exc:
                messagebox.showerror("Invalid input", str(exc), parent=dlg)

        ttk.Button(dlg, text="Apply", command=_apply).pack(pady=6)

    def _open_calculations_window(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Calculations")
        dlg.configure(bg="#1e293b")
        try:
            dlg.state("zoomed")
        except tk.TclError:
            sw = dlg.winfo_screenwidth()
            sh = dlg.winfo_screenheight()
            dlg.geometry(f"{sw}x{sh}+0+0")
        dlg.minsize(640, 360)

        toolbar = ttk.Frame(dlg, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(
            toolbar,
            text="Run all algorithms with current GUI parameters (s, J, h)",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)

        notebook = ttk.Notebook(dlg)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        tab_widgets: dict[str, scrolledtext.ScrolledText] = {}
        for name in ["Hamiltonian Energy", "Simulated Annealing", "Genetic Algorithm", "Tabu Search"]:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            txt = scrolledtext.ScrolledText(
                tab,
                wrap=tk.WORD,
                font=("Courier New", 10),
                bg="#0f172a",
                fg="#e2e8f0",
                insertbackground="#e2e8f0",
            )
            txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            txt.insert("1.0", "Running calculations...\n")
            txt.configure(state=tk.DISABLED)
            tab_widgets[name] = txt

        eval_widgets = create_evaluation_tab(notebook)

        def _run_and_display() -> None:
            self._populate_calculation_tabs(tab_widgets, eval_widgets)

        _run_and_display()

    def _populate_calculation_tabs(
        self,
        tab_widgets: dict[str, scrolledtext.ScrolledText],
        eval_widgets: dict[str, object],
    ) -> None:
        try:
            s = np.array(self.spins, dtype=float)
            J = self._build_J()
            h = self._build_h()
            bundle = run_all_calculations_bundle(s, J, h)
            results = bundle["texts"]
            metrics = bundle["metrics"]
        except Exception as exc:
            err_text = f"Calculation error:\n{exc}\n"
            for txt in tab_widgets.values():
                txt.configure(state=tk.NORMAL)
                txt.delete("1.0", tk.END)
                txt.insert("1.0", err_text)
                txt.configure(state=tk.DISABLED)
            clear_evaluation_tab(eval_widgets)
            return

        for tab_name, txt in tab_widgets.items():
            body = results.get(tab_name, "No result available.\n")
            txt.configure(state=tk.NORMAL)
            txt.delete("1.0", tk.END)
            txt.insert("1.0", body)
            txt.configure(state=tk.DISABLED)

        render_evaluation(metrics, eval_widgets)

    def _build_J(self) -> np.ndarray:
        topo = self._topology_var.get()
        J = np.zeros((self.N, self.N))
        if topo == "Custom":
            return self._custom_J if self._custom_J is not None else J
        if topo == "None (disconnected)":
            return J
        try:
            val = float(self._j_var.get())
        except ValueError:
            val = 1.0
        if topo == "Fully Connected":
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    J[i, j] = val
        else:
            for i in range(self.N - 1):
                J[i, i + 1] = val
        return J

    def _build_h(self) -> np.ndarray:
        if hasattr(self, "_custom_h") and self._custom_h is not None:
            return self._custom_h
        try:
            val = float(self._h_var.get())
        except ValueError:
            val = 0.0
        return np.full(self.N, val)

    @staticmethod
    def _matrix_to_str(mat: np.ndarray) -> str:
        return "\n".join("  ".join(f"{v:6.2f}" for v in row) for row in mat)

    def _refresh(self) -> None:
        self._draw_spins()

    def _draw_spins(self) -> None:
        c = self._canvas
        c.delete("all")

        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10:
            W = 700
        if H < 10:
            H = 300

        N = self.N
        J = self._build_J()

        if N == 2:
            r = self.RADIUS_MAX
            positions: list[tuple[float, float]] = [(W / 3, H / 2), (2 * W / 3, H / 2)]
        else:
            margin = 55
            ring_r = min(W, H) / 2 - margin
            r = min(self.RADIUS_MAX, max(self.RADIUS_MIN, int(ring_r / (N + 1) * 2)))
            ring_r = max(ring_r, r * 3)
            cx0, cy0 = W / 2, H / 2
            positions = []
            for i in range(N):
                angle = 2 * math.pi * i / N - math.pi / 2
                positions.append((cx0 + ring_r * math.cos(angle), cy0 + ring_r * math.sin(angle)))

        self._hit_boxes: list[tuple[int, int, int]] = []

        i_idx, j_idx = np.triu_indices(N, k=1)
        nonzero_pairs = [(i, j) for i, j in zip(i_idx, j_idx) if J[i, j] != 0]
        max_j_abs = max((abs(J[i, j]) for i, j in nonzero_pairs), default=1.0)

        for i, j in nonzero_pairs:
            jval = J[i, j]
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            line_w = max(1, round(abs(jval) / max_j_abs * 4))
            line_col = "#22c55e" if jval > 0 else "#f97316"
            dash: tuple = () if abs(jval) >= max_j_abs * 0.5 else (4, 4)
            c.create_line(x1, y1, x2, y2, fill=line_col, width=line_w, dash=dash)
            if N <= 8:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                c.create_text(mx, my - 6, text=f"{jval:.1f}", fill="#475569", font=("Arial", 7))

        for i, spin in enumerate(self.spins):
            cx, cy = int(positions[i][0]), int(positions[i][1])
            color = COL_UP if spin == 1 else COL_DOWN

            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=COL_VAL, width=2)

            shaft = max(4, r - 6)
            if spin == 1:
                c.create_line(cx, cy + shaft // 2, cx, cy - shaft // 2, fill="white", width=2, arrow=tk.LAST, arrowshape=(6, 8, 3))
            else:
                c.create_line(cx, cy - shaft // 2, cx, cy + shaft // 2, fill="white", width=2, arrow=tk.LAST, arrowshape=(6, 8, 3))

            val_text = "+1" if spin == 1 else "−1"
            c.create_text(cx, cy - r - 12, text=val_text, fill=color, font=("Arial", 8, "bold"))
            c.create_text(cx, cy + r + 12, text=f"s{i + 1}", fill=COL_LABEL, font=("Arial", 8))

            self._hit_boxes.append((cx, cy, r))

        c.create_oval(6, H - 22, 18, H - 10, fill=COL_UP, outline="")
        c.create_text(22, H - 16, text="+1 spin up", anchor=tk.W, fill=COL_LABEL, font=("Arial", 8))
        c.create_oval(80, H - 22, 92, H - 10, fill=COL_DOWN, outline="")
        c.create_text(96, H - 16, text="−1 spin down", anchor=tk.W, fill=COL_LABEL, font=("Arial", 8))
        c.create_line(160, H - 16, 185, H - 16, fill="#22c55e", width=2)
        c.create_text(188, H - 16, text="J>0 ferro", anchor=tk.W, fill=COL_LABEL, font=("Arial", 8))
        c.create_line(258, H - 16, 283, H - 16, fill="#f97316", width=2)
        c.create_text(286, H - 16, text="J<0 antiferro", anchor=tk.W, fill=COL_LABEL, font=("Arial", 8))
