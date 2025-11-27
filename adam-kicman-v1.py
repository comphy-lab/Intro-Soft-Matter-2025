import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ----------------------------
# Define the ODE system
# ----------------------------
# Let y[0] = h
#     y[1] = h'
#     y[2] = h''
# Then y'[2] = h''' = RHS
def ode_system(x, y):
    h = y[0]
    rhs = -0.01 / (h**2 + h)   # h'''(x)
    return np.vstack((y[1],      # h'
                      y[2],      # h''
                      rhs))      # h'''

# ----------------------------
# Boundary conditions
# ----------------------------
def bc(ya, yb):
    return np.array([
        ya[0] - 0.0,   # h(0) = 0
        ya[1] - 1.0,   # h'(0) = 1
        yb[2] - 0.0    # h''(inf) = 0  approx at x_max
    ])

# ----------------------------
# Domain and initial guess
# ----------------------------
x_max = 50              # large-domain approximation for infinity
x = np.linspace(0, x_max, 500)

# initial guess (simple linear growth)
y_init = np.zeros((3, x.size))
y_init[0] = x            # h ~ x
y_init[1] = 1.0          # h' ~ 1
y_init[2] = 0.0          # h'' ~ 0

# ----------------------------
# Solve BVP
# ----------------------------
sol = solve_bvp(ode_system, bc, x, y_init, max_nodes=10000)

if not sol.success:
    print("WARNING: BVP solver did not converge")

# Evaluate solution
x_plot = np.linspace(0, x_max, 1000)
y_plot = sol.sol(x_plot)

h = y_plot[0]
hp = y_plot[1]
hpp = y_plot[2]

# ----------------------------
# Plot results
# ----------------------------
plt.figure(figsize=(10,6))

# h'(x)
plt.subplot(2,1,1)
plt.plot(x_plot, hp)
plt.xlabel("x")
plt.ylabel("h'(x)")
plt.title("h'(x) vs x")
plt.grid()

# h''(x)
plt.subplot(2,1,2)
plt.plot(x_plot, hpp)
plt.xlabel("x")
plt.ylabel("h''(x)")
plt.title("h''(x) vs x")
plt.grid()

plt.tight_layout()
plt.show()