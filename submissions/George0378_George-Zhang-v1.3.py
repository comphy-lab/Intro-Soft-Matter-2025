#!/usr/bin/env python3
"""
Solve the third-order ODE:

    d^3 h / dx^3 = -0.01 / (h^2 + h)

with boundary conditions:

    h(0) = 0
    h'(0) = 1
    h''(∞) = 0    (approximated numerically by h''(L) = 0)

and produce:
1. A linear–linear plot of h'(x) and h''(x)
2. A log–log plot of h'(x) and h''(x)

Log–log plots are included because:
    "For asymptotic behaviours (e.g. algebraic or exponential decay), 
     a log–log axis allows us to clearly visualise the slope, 
     revealing whether h', h'' decay with a power-law trend. 
     Linear axes often compress small values near large x, 
     hiding the behaviour of interest."

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os

# ---------------------------------------------------------------------
# Numerical domain
# ---------------------------------------------------------------------
L = 5000.0   # Approximation of x = ∞
eps = 1e-6 # Regularisation so denominator doesn't blow up at h=0


# ---------------------------------------------------------------------
# Third-order ODE written as a system of first-order equations
# ---------------------------------------------------------------------
# y[0] = h
# y[1] = h'
# y[2] = h''
#
# System:
#   y0' = y1
#   y1' = y2
#   y2' = -0.01 / (h^2 + h + eps)
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
        ya[0] - 0.0,   # h(0)  = 0
        ya[1] - 1.0,   # h'(0) = 1
        yb[2] - 0.0    # h''(L) = 0  ≈ h''(∞) = 0
    ])


# ---------------------------------------------------------------------
# Initial mesh and crude initial guess
# ---------------------------------------------------------------------
x_init = np.linspace(0, L, 200)

y_init = np.zeros((3, x_init.size))
y_init[0] = x_init         # guess: h(x) ≈ x
y_init[1] = 1.0            # guess: h'(x) ≈ constant
y_init[2] = 0.0            # guess: h''(x) ≈ 0


# ---------------------------------------------------------------------
# Solve the BVP
# ---------------------------------------------------------------------
sol = solve_bvp(ode_system, bc, x_init, y_init)

if sol.status != 0:
    print("WARNING: solve_bvp did not fully converge:")
    print(sol.message)

# Evaluated solution
x_plot = np.linspace(0, L, 1000)
h      = sol.sol(x_plot)[0]
h_p    = sol.sol(x_plot)[1]
h_pp   = sol.sol(x_plot)[2]


# ---------------------------------------------------------------------
# Linear plots
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(x_plot, h_p)
ax1.set_ylabel("h'(x)")
ax1.set_title("Solution of Third-Order ODE (Linear Scale)")
ax1.grid(True)

ax2.plot(x_plot, h_pp)
ax2.set_xlabel("x")
ax2.set_ylabel("h''(x)")
ax2.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# Log–log plots
# ---------------------------------------------------------------------
"""
"Why log–log scale?

The tail behaviour of the solution is asymptotic.
On a linear scale, small values of h' and h'' collapse toward 0,
making it visually impossible to judge the decay rate.

Log–log scaling spreads out the values and turns power-laws into
straight lines. This allows us to check whether h', h'' follow

    h' ~ x^-α    or    h'' ~ x^-β

and makes exponential or algebraic decay immediately visible."

"""

# To avoid log(0), clip extremely small |h'|, |h''|
h_p_safe  = np.clip(np.abs(h_p), 1e-20, None)
h_pp_safe = np.clip(np.abs(h_pp), 1e-20, None)

fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

bx1.loglog(x_plot, h_p_safe)
bx1.set_ylabel("|h'(x)| (log scale)")
bx1.set_title("h'(x) on Log–Log Axes")
bx1.grid(True, which='both')

bx2.loglog(x_plot, h_pp_safe)
bx2.set_xlabel("x (log scale)")
bx2.set_ylabel("|h''(x)| (log scale)")
bx2.grid(True, which='both')

plt.tight_layout()
plt.savefig("hprime_h2prime_loglog.png", dpi=300)
plt.show()

