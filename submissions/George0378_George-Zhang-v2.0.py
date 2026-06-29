#!/usr/bin/env python3
"""
Final working solution for the slip-regularised contact-line problem.

Solve
    h''' = -0.01 / (h^2 + h)

with boundary conditions
    h(0) = 0, h'(0) = 1, h''(infinity) = 0.

The singular left boundary is approximated by x = 1e-6, and infinity is
approximated by x = 1e5.  The comparison plots use the same asymptotic
relations as the reference solution:
    theta^3 = 1 + 0.03 log(e x)
    dtheta/dx = 0.01 / (x theta^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


def ode_system(x, y):
    h = y[0]
    return np.vstack((y[1], y[2], -0.01 / (h**2 + h)))


def boundary_conditions(ya, yb):
    return np.array([
        ya[0] - x_start,
        ya[1] - 1.0,
        yb[2],
    ])


def theta_cubed_analytical(x):
    return 1.0 + 0.03 * np.log(np.e * x)


def dtheta_dx_analytical(x):
    theta = theta_cubed_analytical(x) ** (1.0 / 3.0)
    return 0.01 / (x * theta**2)


x_start = 1e-6
x_end = 1e5
x_plot_start = 1e-3
x_plot_end = 1e4

x_mesh = np.logspace(np.log10(x_start), np.log10(x_end), 5000)
y_guess = np.zeros((3, x_mesh.size))
y_guess[0] = x_mesh
y_guess[1] = theta_cubed_analytical(x_mesh) ** (1.0 / 3.0)
y_guess[2] = np.linspace(-0.01, 0.0, x_mesh.size)

solution = solve_bvp(
    ode_system,
    boundary_conditions,
    x_mesh,
    y_guess,
    max_nodes=100000,
    tol=1e-5,
)

if not solution.success:
    print("WARNING:", solution.message)

x_plot = np.logspace(np.log10(x_plot_start), np.log10(x_plot_end), 1000)
h, theta, dtheta_dx = solution.sol(x_plot)

theta_cubed = theta**3
theta_cubed_exact = theta_cubed_analytical(x_plot)
dtheta_dx_exact = dtheta_dx_analytical(x_plot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.semilogx(x_plot, theta_cubed, "r-", linewidth=3, label="Numerical")
ax1.semilogx(
    x_plot,
    theta_cubed_exact,
    "k--",
    linewidth=2,
    label=r"$1 + 0.03\log(e x)$",
)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$\theta^3$")
ax1.set_title("Contact Angle Cubed")
ax1.set_xlim(1e-3, 1e4)
ax1.set_ylim(0.75, 1.45)
ax1.grid(True, alpha=0.3, which="both")
ax1.legend()

ax2.loglog(x_plot, np.abs(dtheta_dx), "g-", linewidth=3, label="Numerical")
ax2.loglog(
    x_plot,
    dtheta_dx_exact,
    "k--",
    linewidth=2,
    label=r"$0.01/(x\theta^2)$",
)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$|d\theta/dx|$")
ax2.set_title("Derivative of Contact Angle")
ax2.set_xlim(1e-3, 1e4)
ax2.set_ylim(1e-8, 1e2)
ax2.grid(True, alpha=0.3, which="both")
ax2.legend()

plt.tight_layout()
plt.savefig("George0378_loglogplot.png", dpi=300)

fig_linear, (bx1, bx2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
bx1.plot(x_plot, theta, "r-", linewidth=2)
bx1.set_xscale("log")
bx1.set_ylabel(r"$h'(x)$")
bx1.set_title("First Derivative")
bx1.grid(True, alpha=0.3, which="both")

bx2.plot(x_plot, dtheta_dx, "g-", linewidth=2)
bx2.set_xscale("log")
bx2.set_xlabel(r"$x$")
bx2.set_ylabel(r"$h''(x)$")
bx2.set_title("Second Derivative")
bx2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("George0378_linear.png", dpi=300)
plt.show()
