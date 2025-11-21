"""
Solve the following ODE with boundary conditions:
- ODE: dh/dx = -0.01(1/(h^2+h))
- BCs: h(0) = 0, h'(0) = 1, h''(∞) = 0

Write a [INSERT YOUR LANGUAGE HERE] script that:
1. Solves this ODE numerically
2. Plots h'(x) vs x in the first subplot
3. Plots h''(x) vs x in the second subplot

Please provide complete, runnable code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def ode_rhs(x, h):
    return -0.01 / (h**2 + h)

def h_prime(h):
    return -0.01 / (h**2 + h)

def h_double_prime(h):
    return -0.0001 * (2*h + 1) / (h**2 + h)**3

# Numerical settings
x_start = 0.0
x_end = 50.0
h0 = 1.0

sol = solve_ivp(
    fun=ode_rhs,
    t_span=(x_start, x_end),
    y0=[h0],
    dense_output=True,
    max_step=0.1
)

x_plot = np.linspace(x_start, x_end, 1000)
h_vals = sol.sol(x_plot)[0]
hprime_vals = h_prime(h_vals)
h2prime_vals = h_double_prime(h_vals)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(x_plot, hprime_vals)
ax1.set_ylabel("h'(x)")
ax1.grid(True)

ax2.plot(x_plot, h2prime_vals)
ax2.set_xlabel("x")
ax2.set_ylabel("h''(x)")
ax2.grid(True)

plt.tight_layout()

# ------------------------------------------------
# Save outputs to same directory as this script
# ------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

plot_path = os.path.join(script_dir, "solution_plot.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")

data_path = os.path.join(script_dir, "h_data.csv")
np.savetxt(
    data_path,
    np.column_stack([x_plot, h_vals, hprime_vals, h2prime_vals]),
    delimiter=",",
    header="x, h(x), h'(x), h''(x)"
)
print(f"Data saved to: {data_path}")

plt.show()

