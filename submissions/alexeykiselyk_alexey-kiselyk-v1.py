import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def solve_singular_ode():
    # 1. Define Parameters
    # --------------------
    # We shift the start slightly from 0 to avoid division by zero (singularity at h=0)
    x_start = 1e-4  
    # We approximate infinity with a sufficiently large number
    x_end = 20.0    
    
    # 2. Define the ODE System
    # ------------------------
    # Let y[0] = h, y[1] = h', y[2] = h''
    # dy[0]/dx = y[1]
    # dy[1]/dx = y[2]
    # dy[2]/dx = -0.01 / (y[0]^2 + y[0])
    
    def fun(x, y):
        h = y[0]
        dh = y[1]
        d2h = y[2]
        
        # Calculate third derivative
        # Note: h will be > 0 because we start at x_start where h ~ x_start
        d3h = -0.01 / (h**2 + h)
        
        return np.vstack((dh, d2h, d3h))

    # 3. Define Boundary Conditions
    # -----------------------------
    # ya = values at x_start, yb = values at x_end
    # BCs: h(0)=0, h'(0)=1, h''(inf)=0
    
    def bc(ya, yb):
        # BC 1: h(0) = 0. 
        # Since we start at x_start, h(x_start) approx h(0) + h'(0)*x_start = x_start
        bc1 = ya[0] - x_start 
        
        # BC 2: h'(0) = 1.
        bc2 = ya[1] - 1
        
        # BC 3: h''(inf) = 0.
        bc3 = yb[2]
        
        return np.array([bc1, bc2, bc3])

    # 4. Initial Mesh and Guess
    # -------------------------
    x_plot = np.linspace(x_start, x_end, 100)
    
    # Initial guess for y (3 rows, N columns)
    # Guessing linear growth: h ~ x, h' ~ 1, h'' ~ 0
    y_guess = np.zeros((3, x_plot.size))
    y_guess[0] = x_plot
    y_guess[1] = 1.0
    y_guess[2] = 0.0

    # 5. Solve
    # --------
    print("Solving ODE...")
    res = solve_bvp(fun, bc, x_plot, y_guess, tol=1e-5)

    if res.success:
        print("Optimization successful!")
    else:
        print(f"Optimization failed: {res.message}")
        return

    # 6. Plotting
    # -----------
    x = res.x
    h_prime = res.y[1]
    h_double_prime = res.y[2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Plot h'(x)
    ax1.plot(x, h_prime, 'b-', linewidth=2)
    ax1.set_ylabel(r"$h'(x)$", fontsize=12)
    ax1.set_title(r"Solution for $h'(x)$", fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add annotation for the start value
    ax1.plot(0, 1, 'ro', label="BC: h'(0)=1")
    ax1.legend()

    # Plot h''(x)
    ax2.plot(x, h_double_prime, 'r-', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel(r"$h''(x)$", fontsize=12)
    ax2.set_title(r"Solution for $h''(x)$", fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add annotation for the end value
    ax2.annotate(r"$h''(\infty) \to 0$", xy=(x[-1], h_double_prime[-1]), 
                 xytext=(x[-1]-5, h_double_prime[-1]+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_singular_ode()