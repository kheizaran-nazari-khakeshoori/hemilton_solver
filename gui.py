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
