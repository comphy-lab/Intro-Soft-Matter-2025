import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
eps = 1e-10          # regularization for singularity
x_max = 20           # domain approximation for infinity
N = 4000             # initial mesh points

# ----------------------------
# Define ODE system
# ----------------------------
# Let y[0] = h
#     y[1] = h'
#     y[2] = h''
# Then y'[2] = h''' = RHS
def ode_system(x, y):
    h = y[0]
    rhs = -0.01 / (h**2 + h + eps) # h'''(x)
    return np.vstack((y[1],     # h'
                      y[2],     # h''
                      rhs))     # h'''

# ----------------------------
# Boundary conditions
# ----------------------------
def bc(ya, yb):
    return np.array([
        ya[0] - 0.0,    # h(0) = 0
        ya[1] - 1.0,    # h'(0) = 1
        yb[2] - 0.0     # h''(∞) = 0
    ])

# ----------------------------
# Initial mesh and guess
# ----------------------------
x = np.linspace(0, x_max, N)

# Smooth cubic initial guess: h ≈ x - (x/x_max)^3
h_guess = x - (x/x_max)**3
y_init = np.zeros((3, x.size))
y_init[0] = h_guess
y_init[1] = np.gradient(h_guess, x)
y_init[2] = np.gradient(y_init[1], x)

# ----------------------------
# Solves BVP
# ----------------------------
sol = solve_bvp(ode_system, bc, x, y_init, max_nodes=20000, tol=1e-4)

if not sol.success:
    print("\nWARNING: BVP solver did not converge!\n")

# ----------------------------
# Evaluates solutions
# ----------------------------
x_plot = np.linspace(0, x_max, 1000)
y_plot = sol.sol(x_plot)

h = y_plot[0]
hp = y_plot[1]
hpp = y_plot[2]

# ----------------------------
# Plot results for h(x), h'(x), h''(x)
# ----------------------------
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(x_plot, h)
plt.xlabel("x")
plt.ylabel("h(x)")
plt.grid()

plt.subplot(3,1,2)
plt.plot(x_plot, hp)
plt.xlabel("x")
plt.ylabel("h'(x)")
plt.grid()

plt.subplot(3,1,3)
plt.plot(x_plot, hpp)
plt.xlabel("x")
plt.ylabel("h''(x)")
plt.grid()

plt.tight_layout()
plt.show()

# ----------------------------
# Numerical Stability Plot
# ----------------------------

N_array = np.arange(100,10100+1000,1000)

for N in N_array:
    x = np.linspace(0, x_max, N)

    # Smooth cubic initial guess: h ≈ x - (x/x_max)^3
    h_guess = x - (x/x_max)**3
    y_init = np.zeros((3, x.size))
    y_init[0] = h_guess
    y_init[1] = np.gradient(h_guess, x)
    y_init[2] = np.gradient(y_init[1], x)
    
    sol = solve_bvp(ode_system, bc, x, y_init, max_nodes=20000, tol=1e-4)

    if not sol.success:
        print("\nWARNING: BVP solver did not converge!\n")
    
    y_plot = sol.sol(x_plot)
    
    plt.figure(figsize=(10,6))

    h = y_plot[0]
    hp = y_plot[1]
    hpp = y_plot[2]
    
    plt.subplot(3,1,1)
    plt.plot(x_plot, h)
    plt.xlabel("x")
    plt.ylabel("h(x)")
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(x_plot, hp)
    plt.xlabel("x")
    plt.ylabel("h'(x)")
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(x_plot, hpp)
    plt.xlabel("x")
    plt.ylabel("h''(x)")
    plt.grid()