#!/usr/bin/env python3
"""
Plot theta cubed and its derivative for soft matter visualization.

Plots:
1. Linear scale: θ³(x) and dθ/dx for x ∈ (0+, 1000]
2. Log-log scale: θ³(x) and dθ/dx for x ∈ [10^-3, 10^5]

Analytical expressions:
    θ³ = 1 + 0.03*ln(e*x)
    dθ/dx = 0.01/(x*θ²) where θ = (θ³)^(1/3)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('axes', labelsize=18, titlesize=18)
rc('legend', fontsize=16)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)


def theta_cubed(x):
    """
    Calculate θ³ = 1 + 0.03*ln(e*x)

    Parameters
    ----------
    x : array_like
        Input values (must be positive)

    Returns
    -------
    array_like
        θ³ values
    """
    return 1 + 0.03 * np.log(np.e * x)


def theta(x):
    """
    Calculate θ = (θ³)^(1/3) = [1 + 0.03*ln(e*x)]^(1/3)

    Parameters
    ----------
    x : array_like
        Input values (must be positive)

    Returns
    -------
    array_like
        θ values
    """
    return theta_cubed(x)**(1/3)


def dtheta_dx(x):
    """
    Calculate dθ/dx = 0.01/(x*θ²)

    Parameters
    ----------
    x : array_like
        Input values (must be positive)

    Returns
    -------
    array_like
        dθ/dx values
    """
    theta_val = theta(x)
    return 0.01 / (x * theta_val**2)


# ============================================================================
# Figure 1: Linear scale plots
# ============================================================================

# Create x values for linear scale (avoid x=0 exactly)
x_linear = np.linspace(0.01, 1000, 2000)

# Calculate function values
theta_cubed_linear = theta_cubed(x_linear)
dtheta_linear = dtheta_dx(x_linear)

# Create figure with two subplots
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: θ³(x)
ax1.plot(x_linear, theta_cubed_linear, 'b-', linewidth=3, label=r'$\theta^3 = 1 + 0.03\ln(ex)$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\theta^3$')
ax1.set_title(r'Contact Angle Cubed $\theta^3$ vs Position')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best')
ax1.set_xlim([0, 1000])

# Subplot 2: dθ/dx
ax2.plot(x_linear, dtheta_linear, 'r-', linewidth=3, label=r'$\frac{d\theta}{dx} = \frac{0.01}{x\theta^2}$')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\frac{d\theta}{dx}$')
ax2.set_title(r'Derivative of Contact Angle')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best')
ax2.set_xlim([0, 1000])
ax2.set_ylim([0, max(dtheta_linear) * 1.1])

plt.tight_layout()
plt.savefig('Cox-Voinov-largeX_linear.pdf', dpi=300, bbox_inches='tight')
print("Saved: Cox-Voinov-largeX_linear.pdf")

# ============================================================================
# Figure 2: Log-log scale plots
# ============================================================================

# Create x values for log scale
x_log = np.logspace(-3, 5, 2000)

# Calculate function values
theta_cubed_log = theta_cubed(x_log)
dtheta_log = dtheta_dx(x_log)

# Create figure with two subplots
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: θ³(x) on log-log scale
ax3.loglog(x_log, theta_cubed_log, 'b-', linewidth=3, label=r'$\theta^3 = 1 + 0.03\ln(ex)$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$\theta^3$')
ax3.set_title(r'Contact Angle Cubed $\theta^3$ (Log-Log)')
ax3.grid(True, alpha=0.3, linestyle='--', which='both')
ax3.legend(loc='best')
ax3.set_xlim([1e-3, 1e5])

# Subplot 2: dθ/dx on log-log scale
ax4.loglog(x_log, dtheta_log, 'r-', linewidth=3, label=r'$\frac{d\theta}{dx} = \frac{0.01}{x\theta^2}$')
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$\frac{d\theta}{dx}$')
ax4.set_title(r'Derivative of Contact Angle (Log-Log)')
ax4.grid(True, alpha=0.3, linestyle='--', which='both')
ax4.legend(loc='best')
ax4.set_xlim([1e-3, 1e5])

plt.tight_layout()
plt.savefig('Cox-Voinov-largeX_loglog.pdf', dpi=300, bbox_inches='tight')
print("Saved: Cox-Voinov-largeX_loglog.pdf")

plt.show()
