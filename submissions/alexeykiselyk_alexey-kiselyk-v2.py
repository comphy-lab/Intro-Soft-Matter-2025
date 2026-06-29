import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def solve_and_compare():
    # 1. Define Parameters
    # --------------------
    # We need L=10000 to cover the requested plotting range up to 10^4
    epsilon = 1e-6    
    L = 100000.0       
    
    # 2. Define the ODE System
    # ------------------------
    # y = [h, h', h'']
    def fun(x, y):
        h = y[0]
        dh = y[1]
        d2h = y[2]
        # d^3h/dx^3 = -0.01 / (h^2 + h)
        d3h = -0.01 / (h**2 + h)
        return np.vstack((dh, d2h, d3h))

    # 3. Define Boundary Conditions
    # -----------------------------
    def bc(ya, yb):
        # h(eps) ~ eps, h'(eps) ~ 1, h''(L) = 0
        return np.array([
            ya[0] - epsilon,
            ya[1] - 1.0,
            yb[2]
        ])

    # 4. Solve
    # --------
    # Use geometric space for mesh to handle the 7 orders of magnitude (10^-3 to 10^4)
    x_mesh = np.geomspace(epsilon, L, 5000)
    
    # Initial guess
    y_guess = np.zeros((3, x_mesh.size))
    y_guess[0] = x_mesh
    y_guess[1] = 1.0
    
    print(f"Solving ODE on domain [{epsilon}, {L}]...")
    res = solve_bvp(fun, bc, x_mesh, y_guess, tol=1e-5, max_nodes=100000)

    if not res.success:
        print(f"Optimization failed: {res.message}")
        return
    else:
        print("Optimization successful!")

    # 5. Extract Data
    # ---------------
    x = res.x
    h = res.y[0]
    h_prime = res.y[1]
    h_double_prime = res.y[2]

    # 6. Plotting
    # -----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Subplot 1: h'(x) vs x (Linear) ---
    ax1 = axes[0, 0]
    ax1.plot(x, h_prime, 'b-', linewidth=2)
    ax1.set_ylabel(r"$h'(x)$", fontsize=12)
    ax1.set_title(r"1. Standard: $h'(x)$ vs $x$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    # Zoom in to relevant start area to see detail, or keep full range
    ax1.set_xlim(0, 20) # Limiting x to see the initial rise clearly
    ax1.set_ylim(0.98, 1.05) # so full range seen clearly
    
    # --- Subplot 2: h''(x) vs x (Linear) ---
    ax2 = axes[0, 1]
    ax2.plot(x, h_double_prime, 'r-', linewidth=2)
    ax2.set_ylabel(r"$h''(x)$", fontsize=12)
    ax2.set_title(r"2. Standard: $h''(x)$ vs $x$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)

    # --- Subplot 3: (h')^3 vs x (Linear-Log) ---
    # Requested: Linear Y, Log X
    # Eq: (h')^3 = 1 + 0.03 * ln(e*x)
    ax3 = axes[1, 0]
    
    # Analytical approximation
    # Use a range that doesn't break log (x > 0)
    x_eq = np.geomspace(1e-3, 1e4, 100)
    eq1 = 1 + 0.03 * np.log(np.e * x_eq)
    
    ax3.semilogx(x, h_prime**3, 'b-', linewidth=2, label='Numerical')
    ax3.semilogx(x_eq, eq1, 'k:', linewidth=2, label=r'$1 + 0.03 \ln(ex)$')
    
    ax3.set_xlim(1e-3, 1e4)
    ax3.set_ylim(0.8, 1.5)
    ax3.set_xlabel('x (log scale)', fontsize=12)
    ax3.set_ylabel(r"$(h')^3$", fontsize=12)
    ax3.set_title(r"3. Linear-Log: $(h')^3$ vs $x$", fontsize=12)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    # --- Subplot 4: h'' vs x (Log-Log) ---
    # Requested: "Linear-Log" but with range 10^2 to 10^-8. 
    # This range requires a Log Y axis (Log-Log plot).
    # Eq: h'' = 0.01 / (x * (h')^2)
    ax4 = axes[1, 1]
    
    eq2 = 0.01 / (x * h_prime**2)
    
    ax4.loglog(x, h_double_prime, 'r-', linewidth=2, label='Numerical')
    ax4.loglog(x, eq2, 'k:', linewidth=2, label=r'$\frac{0.01}{x (h\')^2}$')
    
    ax4.set_xlim(1e-3, 1e4)
    ax4.set_ylim(1e-8, 1e2)
    ax4.set_xlabel('x (log scale)', fontsize=12)
    ax4.set_ylabel(r"$h''(x)$", fontsize=12)
    ax4.set_title(r"4. Log-Log: $h''(x)$ vs $x$", fontsize=12)
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_and_compare()