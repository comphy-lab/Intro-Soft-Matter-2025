#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:47:49 2025

@author: aidannicholas
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ODE system:
# y = [h, h', h'']
# y' = [h', h'', h''']
# h''' = -0.01 / (h^2 + h)
# ----------------------------------------------------------

def ode_system(x, y):
    h, h1, h2 = y
    # Prevent division by zero
    denom = h*h + h
    if abs(denom) < 1e-10:
        denom = np.sign(denom) * 1e-10
    h3 = -0.01 / denom
    return [h1, h2, h3]

# ----------------------------------------------------------
# Shooting method: adjust h''(0) = s such that h''(xmax) = 0
# ----------------------------------------------------------

xmax = 200

def shoot(s):
    """Return h''(xmax) for a given guess s = h''(0)."""
    y0 = [0.0, 1.0, s]   # h(0)=0, h'(0)=1, h''(0)=s
    sol = solve_ivp(ode_system, [0, xmax], y0, max_step=0.5, rtol=1e-6, atol=1e-9)
    return sol.y[2, -1]  # h''(xmax)

# Find s such that h''(xmax) ≈ 0
sol_root = root_scalar(shoot, bracket=[-1, 1], method="bisect")
s_opt = sol_root.root
print("Optimal h''(0) =", s_opt)

# ----------------------------------------------------------
# Solve the ODE with the optimal initial condition
# ----------------------------------------------------------

y0 = [0.0, 1.0, s_opt]
sol = solve_ivp(
    ode_system,
    [0, xmax],
    y0,
    max_step=0.5,
    dense_output=True,
    rtol=1e-6,
    atol=1e-9
)

x = sol.t
h = sol.y[0]
h1 = sol.y[1]
h2 = sol.y[2]

# ----------------------------------------------------------
# Plot results
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, h1, 'b')
plt.xlabel("x")
plt.ylabel("h'(x)")
plt.title("First Derivative h'(x)")

plt.subplot(2, 1, 2)
plt.plot(x, h2, 'r')
plt.xlabel("x")
plt.ylabel("h''(x)")
plt.title("Second Derivative h''(x)")

plt.tight_layout()
plt.show()