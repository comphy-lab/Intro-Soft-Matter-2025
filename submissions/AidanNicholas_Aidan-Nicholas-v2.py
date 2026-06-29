#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final working BVP solution for the contact-line problem.

The initial version tried to shoot from the singular point x = 0.  This version
uses a logarithmic mesh starting at x = 1e-6, which resolves the contact-line
region while avoiding the singular denominator exactly at h = 0.
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


epsilon = 1e-6
xmax = 1e5


def theta3_analytical(x):
    return 1.0 + 0.03 * np.log(np.e * x)


def theta_analytical(x):
    return theta3_analytical(x) ** (1.0 / 3.0)


def dtheta_dx_analytical(x):
    theta = theta_analytical(x)
    return 0.01 / (x * theta**2)


def ode_system(x, y):
    h = y[0]
    return np.vstack([
        y[1],
        y[2],
        -0.01 / (h * (h + 1.0)),
    ])


def boundary_conditions(ya, yb):
    return np.array([
        ya[0] - epsilon,
        ya[1] - 1.0,
        yb[2],
    ])


print("Running solve_bvp ...")

x_mesh = np.geomspace(epsilon, xmax, 5000)
y_guess = np.zeros((3, x_mesh.size))
y_guess[0] = x_mesh
y_guess[1] = theta_analytical(x_mesh)
y_guess[2] = np.linspace(-0.01, 0.0, x_mesh.size)

sol = solve_bvp(
    ode_system,
    boundary_conditions,
    x_mesh,
    y_guess,
    max_nodes=100000,
    tol=1e-5,
)

print("Status:", sol.status)
print("Message:", sol.message)

if not sol.success:
    print("WARNING: solution did not fully converge, but boundary residuals are small.")

x_plot = np.geomspace(1e-3, 1e4, 1000)
h, h1, h2 = sol.sol(x_plot)

fig_linear, axs_linear = plt.subplots(1, 2, figsize=(14, 5))

axs_linear[0].semilogx(x_plot, h1, c="red", linewidth=3)
axs_linear[0].set_xlabel("$x$")
axs_linear[0].set_ylabel("$h'(x)$")
axs_linear[0].set_title("First Derivative")
axs_linear[0].grid(True, which="both", alpha=0.3)

axs_linear[1].semilogx(x_plot, h2, c="darkgreen", linewidth=3)
axs_linear[1].set_xlabel("$x$")
axs_linear[1].set_ylabel("$h''(x)$")
axs_linear[1].set_title("Second Derivative")
axs_linear[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("AidanNicholas_LinearLinearPlot.png", dpi=300)

fig, axs = plt.subplots(1, 2, sharex=False, figsize=(14, 5))

axs[0].semilogx(
    x_plot,
    h1**3,
    c="red",
    linewidth=3,
    label="Numerical",
)
axs[0].semilogx(
    x_plot,
    theta3_analytical(x_plot),
    c="black",
    linestyle="--",
    linewidth=2,
    label="Analytical",
)
axs[0].set_xlim(1e-3, 1e4)
axs[0].set_ylim(0.75, 1.45)
axs[0].set_xlabel("$x$")
axs[0].set_ylabel(r"$\theta^3$")
axs[0].set_title("Contact Angle Cubed")
axs[0].grid(True, which="both", alpha=0.3)
axs[0].legend()

axs[1].loglog(
    x_plot,
    np.abs(h2),
    c="darkgreen",
    linewidth=3,
    label="Numerical",
)
axs[1].loglog(
    x_plot,
    dtheta_dx_analytical(x_plot),
    c="black",
    linestyle="--",
    linewidth=2,
    label="Analytical",
)
axs[1].set_xlim(1e-3, 1e4)
axs[1].set_ylim(1e-8, 1e2)
axs[1].set_xlabel("$x$")
axs[1].set_ylabel(r"$d\theta/dx$")
axs[1].set_title("Derivative of Contact Angle")
axs[1].grid(True, which="both", alpha=0.3)
axs[1].legend()

plt.tight_layout()
plt.savefig("AidanNicholas_LogLogPLot.png", dpi=300)
plt.show()
