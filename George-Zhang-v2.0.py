#!/usr/bin/env python3
"""
Solve the third-order ODE:

    d^3 h / dx^3 = -0.01 / (h^2 + h)

with boundary conditions:

    h(0) = 0
    h'(0) = 1
    h''(∞) = 0    (numerically approximated by h''(L) = 0)

and produce:
1. A log–log plot of h'(x) and h''(x)
2. (optional) A linear-scale plot, currently commented out

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# ---------------------------------------------------------------------
# Numerical setup
# ---------------------------------------------------------------------
L = 500.0
eps = 1e-6

# ---------------------------------------------------------------------
# Third-order ODE written as first-order system
# ---------------------------------------------------------------------
def ode_system(x, y):
    h = y[0]
    denom = h**2 + h + eps
    d3h = -0.01 / denom
    return np.vstack((y[1],    # h'
                      y[2],    # h''
                      d3h))    # h'''


# ---------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------
def bc(ya, yb):
    return np.array([
        ya[0] - 0.0,   # h(0)
        ya[1] - 1.0,   # h'(0)
        yb[2] - 0.0    # h''(L)
    ])


# ---------------------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------------------
x_init = np.linspace(0, L, 200)
y_init = np.zeros((3, x_init.size))
y_init[0] = x_init
y_init[1] = 1.0
y_init[2] = 0.0

# ---------------------------------------------------------------------
# Solve BVP
# ---------------------------------------------------------------------
sol = solve_bvp(ode_system, bc, x_init, y_init)
if sol.status != 0:
    print("WARNING:", sol.message)

x_plot = np.linspace(0, L, 1000)
h      = sol.sol(x_plot)[0]
h_p    = sol.sol(x_plot)[1]
h_pp   = sol.sol(x_plot)[2]


# ---------------------------------------------------------------------
# (OPTIONAL) Linear plots — COMMENTED OUT
# ---------------------------------------------------------------------


# Uncomment this block if you want linear-scale figures.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(x_plot, h_p)
ax1.set_ylabel("h'(x)")
ax1.set_title("First Derivative (Linear Scale)")
#ax1.grid(True)

ax2.plot(x_plot, h_pp)
ax2.set_xlabel("x")
ax2.set_ylabel("h''(x)")
ax2.set_title("Second Derivative (Linear Scale)")
#ax2.grid(True)

plt.tight_layout()
fig.savefig("linear.png", dpi=300)
plt.show()


# ---------------------------------------------------------------------
# Log–log plots (ACTIVE)
# ---------------------------------------------------------------------

# Avoid log(0)
h_p_safe  = np.clip(np.abs(h_p), 1e-20, None)
h_pp_safe = np.clip(np.abs(h_pp), 1e-20, None)

fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# First derivative: h'(x)
bx1.loglog(x_plot, h_p_safe)
bx1.set_ylabel("log(|h'(x)|)")
bx1.set_title("First derivative on log–log scale")
#bx1.grid(True, which='both')

# Second derivative: h''(x)
bx2.loglog(x_plot, h_pp_safe)
bx2.set_xlabel("log(x)")
bx2.set_ylabel("log(|h''(x)|)")
bx2.set_title("Second derivative on log–log scale")
#bx2.grid(True, which='both')

plt.tight_layout()
fig2.savefig("loglogplot.png", dpi=300)
plt.show()

