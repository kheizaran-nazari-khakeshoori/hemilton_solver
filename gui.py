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
