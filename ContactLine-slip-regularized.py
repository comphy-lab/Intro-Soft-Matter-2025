#!/usr/bin/env python3
"""
Solve the contact line problem with no-slip boundary condition using BVP solver.

Third-order ODE:
    d³h/dx³ = -0.01/(h² + h)

Boundary conditions:
    - h(0) = 0         (height vanishes at contact line)
    - h'(0) = 1        (unit slope at contact line)
    - h''(∞) = 0       (curvature vanishes in far field)

Integration range: x ∈ [10⁻⁶, 10⁵]
Plotting range: x ∈ [10⁻³, 10⁵]
(numerically approximates x → 0 and x → ∞)

Comparison with analytical approximations:
    θ³ = (dh/dx)³ = 1 + 0.03*ln(e*x)
    dθ/dx = 0.03/(e*x*θ²)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from matplotlib import rc

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif", size=16)
rc("axes", labelsize=18, titlesize=18)
rc("legend", fontsize=16)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)


def droplet_ode(x, y):
    """
    System of first-order ODEs for the droplet profile.

    Convert d³h/dx³ = -0.01/(h² + h) to system:
        h' = p
        p' = q
        q' = -0.01/(h² + h)

    where:
        y[0] = h (height)
        y[1] = p = dh/dx (slope)
        y[2] = q = d²h/dx² (curvature)

    Parameters
    ----------
    x : float or array
        Position
    y : array_like
        State vector [h, dh/dx, d²h/dx²]

    Returns
    -------
    array_like
        Derivatives [dh/dx, d²h/dx², d³h/dx³]
    """
    h, p, q = y

    # Add regularization to prevent division by very small numbers
    h_reg = np.maximum(h, 1e-10)

    dh_dx = p
    dp_dx = q
    dq_dx = -0.01 / (h_reg**2 + h_reg)

    return np.vstack([dh_dx, dp_dx, dq_dx])


def boundary_conditions(ya, yb):
    """
    Boundary conditions for the BVP.

    Physical BCs:
        h(0) = 0, h'(0) = 1, h''(∞) = 0

    Numerical implementation:
        At x = x_start → 0: h = 0, h' = 1
        At x = x_end → ∞: h'' = 0

    Parameters
    ----------
    ya : array
        State at left boundary [h, h', h'']
    yb : array
        State at right boundary [h, h', h'']

    Returns
    -------
    array
        Residuals for 3 boundary conditions
    """
    h_a, p_a, q_a = ya
    h_b, p_b, q_b = yb

    # Boundary conditions:
    # h(0) = 0, h'(0) = 1, h''(∞) = 0
    return np.array(
        [
            h_a,  # h(x_start → 0) = 0
            p_a - 1.0,  # h'(x_start → 0) = 1
            q_b,  # h''(x_end → ∞) = 0
        ]
    )


def theta_cubed_analytical(x):
    """
    Analytical approximation for θ³ = (dh/dx)³.

    θ³ = 1 + 0.03*ln(e*x)
    """
    return 1 + 0.03 * np.log(np.e * x)


def theta_analytical(x):
    """
    Analytical approximation for θ = dh/dx.

    θ = [1 + 0.03*ln(e*x)]^(1/3)
    """
    return theta_cubed_analytical(x) ** (1 / 3)


def dtheta_dx_analytical(x):
    """
    Analytical approximation for dθ/dx.

    dθ/dx = 0.01/(x*θ²)
    """
    theta = theta_analytical(x)
    return 0.01 / (x * theta**2)


# ============================================================================
# Set up and solve the BVP
# ============================================================================

print("Solving contact line ODE with no-slip boundary condition using BVP solver...")

# Integration domain
x_start = 1e-6
x_end = 1e5

# Plotting domain (subset of integration domain)
x_plot_start = 1e-3

# Create logarithmically spaced mesh
N_points = 5000
x_mesh = np.logspace(np.log10(x_start), np.log10(x_end), N_points)

# Initial guess for the solution
# h: grows from ~0 at x→0
# p (dh/dx): θ = (θ³)^(1/3) where θ³ = 1 + 0.03*ln(ex)
# q (d²h/dx²): small negative values, goes to 0 at far field
y_guess = np.zeros((3, x_mesh.size))
y_guess[0, :] = x_mesh  # h ≈ x for small x
theta_cubed_guess = 1 + 0.03 * np.log(np.e * x_mesh)
y_guess[1, :] = theta_cubed_guess ** (1 / 3)  # h' = θ = (θ³)^(1/3)
y_guess[2, :] = np.linspace(-0.01, 0.0, N_points)  # h'' guess

print(f"Integration domain: x ∈ [{x_start:.0e}, {x_end:.0e}]")
print(f"Initial mesh size: {N_points} points")

# Solve the BVP
solution = solve_bvp(
    droplet_ode,
    boundary_conditions,
    x_mesh,
    y_guess,
    max_nodes=50000,
    tol=1e-6,
    verbose=2,
)

if solution.success:
    print(f"\nSolution converged successfully!")
    print(f"Number of iterations: {solution.niter}")
    print(f"Final mesh size: {len(solution.x)} points")
else:
    print(f"\nWarning: Solution did not converge fully")
    print(f"Message: {solution.message}")

# Extract solution
x_full = solution.x
h_full = solution.y[0]
dh_dx_full = solution.y[1]
d2h_dx2_full = solution.y[2]

# Verify boundary conditions
print(f"\nBoundary condition verification:")
print(f"  h(0) ≈ h({x_full[0]:.2e}) = {h_full[0]:.6f} (expected: 0)")
print(f"  h'(0) ≈ h'({x_full[0]:.2e}) = {dh_dx_full[0]:.6f} (expected: 1.0)")
print(f"  h''(∞) ≈ h''({x_full[-1]:.2e}) = {d2h_dx2_full[-1]:.6e} (expected: 0)")

print(f"\nSolution obtained with {len(x_full)} points")

# Filter solution for plotting (only x >= x_plot_start)
plot_mask = x_full >= x_plot_start
x = x_full[plot_mask]
h = h_full[plot_mask]
dh_dx = dh_dx_full[plot_mask]
d2h_dx2 = d2h_dx2_full[plot_mask]

print(f"Plotting {len(x)} points (x >= {x_plot_start:.0e})")

# Calculate analytical approximations
theta_cubed_anal = theta_cubed_analytical(x)
dtheta_dx_anal = dtheta_dx_analytical(x)

# ============================================================================
# Figure: Log-log scale plots with analytical comparison
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1,1: θ³ = (dh/dx)³ on log-log scale with analytical comparison
mask_dh_positive = dh_dx > 0
theta_cubed_numerical = dh_dx[mask_dh_positive] ** 3
ax1.loglog(
    x[mask_dh_positive],
    theta_cubed_numerical,
    "r-",
    linewidth=3,
    label=r"Numerical: $\theta^3 = \left(\frac{dh}{dx}\right)^3$",
)
mask_theta_cubed_positive = theta_cubed_anal > 0
ax1.loglog(
    x[mask_theta_cubed_positive],
    theta_cubed_anal[mask_theta_cubed_positive],
    "k--",
    linewidth=2,
    label=r"Analytical: $\theta^3 = 1 + 0.03\ln(ex)$",
)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$\theta^3$")
ax1.set_title(r"Contact Angle Cubed")
ax1.grid(True, alpha=0.3, linestyle="--", which="both")
ax1.legend(loc="best")
ax1.set_xlim([1e-3, 1e4])

# Subplot 1,2: dθ/dx on log-log scale with analytical comparison
# dθ/dx = d²h/dx² (since θ = dh/dx)
ax2.loglog(
    x,
    np.abs(d2h_dx2),
    "g-",
    linewidth=3,
    label=r"Numerical: $\left|\frac{d\theta}{dx}\right|$",
)
ax2.loglog(
    x,
    np.abs(dtheta_dx_anal),
    "k--",
    linewidth=2,
    label=r"Analytical: $\frac{d\theta}{dx} = \frac{0.01}{x\theta^2}$",
)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$\left|\frac{d\theta}{dx}\right|$")
ax2.set_title(r"Derivative of Contact Angle")
ax2.grid(True, alpha=0.3, linestyle="--", which="both")
ax2.legend(loc="best")
ax2.set_xlim([1e-3, 1e4])
ax2.set_ylim([1e-8, None])  # Set y minimum to 1e-8

plt.tight_layout()
plt.savefig("ContactLine-slip.pdf", dpi=300, bbox_inches="tight")
print("\nSaved: ContactLine-slip.pdf")

plt.show()
