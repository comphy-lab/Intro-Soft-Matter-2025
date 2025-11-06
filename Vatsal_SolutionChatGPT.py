import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of ODEs
def ode_system(x, y):
    h1, h2, h3 = y  # unpack the variables: h1 = h, h2 = h', h3 = h''
    
    # Third derivative equation
    dh3dx = -0.01 * (1 / (h1**2 + h1))
    
    # First and second derivative equations
    dh1dx = h2
    dh2dx = h3
    
    return [dh1dx, dh2dx, dh3dx]

# Initial conditions
h1_0 = 0     # h(0) = 0
h2_0 = 1     # h'(0) = 1
h3_0 = 0     # h''(0) = 0 (starting guess for second derivative)

initial_conditions = [h1_0, h2_0, h3_0]

# Solve the ODE system
x_span = (0, 50)  # Solve from x = 0 to x = 50
x_eval = np.linspace(0, 50, 500)

sol = solve_ivp(ode_system, x_span, initial_conditions, t_eval=x_eval, method='RK45')

# Extract solutions for plotting
h1_sol = sol.y[0]
h2_sol = sol.y[1]
h3_sol = sol.y[2]
x_sol = sol.t

# Plot h'(x)^3 vs x
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_sol, h2_sol**3, label="h'(x)^3")
plt.title("Plot of h'(x)^3 vs x")
plt.xlabel("x")
plt.ylabel("h'(x)^3")
plt.grid(True)

# Plot h''(x) vs x
plt.subplot(1, 2, 2)
plt.plot(x_sol, h3_sol, label="h''(x)")
plt.title("Plot of h''(x) vs x")
plt.xlabel("x")
plt.ylabel("h''(x)")
plt.grid(True)

plt.tight_layout()
plt.show()

