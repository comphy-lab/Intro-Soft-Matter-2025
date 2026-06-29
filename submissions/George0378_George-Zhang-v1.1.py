#!/usr/bin/env python3
"""
Solve the third-order ODE

    d^3 h / dx^3 = -0.01 * 1 / (h^2 + h)

with boundary conditions

    h(0) = 0
    h'(0) = 1
    h''(∞) = 0   (approximated by h''(L) = 0 for large L)

and plot h'(x) and h''(x).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
L = 50.0        # "infinity" for the numerical BVP  (CHANGED: added L)
eps = 1e-6      # small regularisation to avoid division by zero at h=0


# ---------------------------------------------------------------------
# ODE system in first-order form
# ---------------------------------------------------------------------
# CHANGED: Previously we had a 1st-order ODE for h only.
#          Now we write a 3rd-order ODE as a system:
#          y[0] = h,  y[1] = h',  y[2] = h''
#
#          Then:
#              y0' = y1
#              y1' = y2
#              y2' = -0.01 / (h^2 + h)
#
#          We add a small eps to the denominator so h=0 is not singular.
def ode_system(x, y):
    h = y[0]
    denom = h**2 + h + eps              # CHANGED: regularised denominator
    d3h_dx3 = -0.01 / denom
    return np.vstack((y[1],            # y0' = y1
                      y[2],            # y1' = y2
                      d3h_dx3))        # y2' = d^3h/dx^3


# ---------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------
# At x = 0:   h(0) = 0,  h'(0) = 1
# At x = L:   h''(L) = 0  (approximate h''(∞) = 0)
def bc(ya, yb):
    return np.array([
        ya[0] - 0.0,   # h(0)   = 0
        ya[1] - 1.0,   # h'(0)  = 1
        yb[2] - 0.0    # h''(L) = 0
    ])


# ---------------------------------------------------------------------
# Initial mesh and initial guess for the solution
# ---------------------------------------------------------------------
x_init = np.linspace(0.0, L, 200)

# CHANGED: we now need an initial guess for h, h', h'' over the whole domain.
y_init = np.zeros((3, x_init.size))
y_init[0, :] = x_init           # crude guess: h(x) ≈ x
y_init[1, :] = 1.0              # crude guess: h'(x) ≈ 1
y_init[2, :] = 0.0              # crude guess: h''(x) ≈ 0


# ---------------------------------------------------------------------
# Solve the BVP
# ---------------------------------------------------------------------
sol = solve_bvp(ode_system, bc, x_init, y_init)

if sol.status != 0:
    print("WARNING: BVP solver did not converge. Message:")
    print(sol.message)

# Create a fine grid for plotting
x_plot = np.linspace(0.0, L, 1000)
y_plot = sol.sol(x_plot)
h_vals = y_plot[0]
h_prime_vals = y_plot[1]    # h'(x)
h_2prime_vals = y_plot[2]   # h''(x)


# ---------------------------------------------------------------------
# Plot h'(x) and h''(x)
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# First subplot: h'(x)
ax1.plot(x_plot, h_prime_vals)
ax1.set_ylabel("h'(x)")
ax1.set_title("Solution of third-order ODE")
ax1.grid(True)

# Second subplot: h''(x)
ax2.plot(x_plot, h_2prime_vals)
ax2.set_xlabel("x")
ax2.set_ylabel("h''(x)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("hprime_h2prime_linear.png", dpi=300)
plt.show()

