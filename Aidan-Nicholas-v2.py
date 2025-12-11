#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 12:34:36 2025

@author: aidannicholas
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Problem domain
# ----------------------------------------------------------

eps = 1e-22       # start slightly away from 0 (singularity)
xmax = 100000.0        # large domain representing "infinity"

# Create mesh for solve_bvp
x = np.linspace(eps, xmax, 600)

# ----------------------------------------------------------
# Asymptotic expansion near x = 0
# ----------------------------------------------------------

def asymptotics_at_eps(C1):
    """Return h, h', h'' at x = eps using the asymptotic series."""
    x = eps

    h2 = -0.01 * np.log(x) + C1
    h1 = -0.01 * x * np.log(x) + (C1 - 0.01) * x + 1
    h  = -0.005 * x**2 * np.log(x) + ((C1/2) - 0.005) * x**2 + x

    return h, h1, h2

# ----------------------------------------------------------
# Define the ODE in first-order form
# y = [h, h', h'']
# y' = [h', h'', h''']
# h''' = -0.01 / (h(h+1))
# ----------------------------------------------------------

def ode_system(x, y, p):
    h  = y[0]
    h1 = y[1]
    h2 = y[2]

    denom = h * (h + 1)
    tiny = np.abs(denom) < 1e-12
    denom[tiny] = 1e-12 * np.sign(denom[tiny])

    h3 = -0.01 / denom
    return np.vstack([h1, h2, h3])


def bc(ya, yb, p):
    C1 = p[0]

    h_eps, h1_eps, h2_eps = asymptotics_at_eps(C1)

    return np.array([
        ya[0] - h_eps,     # 1. h(eps)  = asymptotic
        ya[1] - h1_eps,    # 2. h'(eps) = asymptotic
        ya[2] - h2_eps,    # 3. h''(eps)= asymptotic
        yb[2] - 0.0        # 4. h''(xmax) = 0
    ])
# ----------------------------------------------------------
# Initial guess for y(x)
# ----------------------------------------------------------

# initial guess: linear h, constant h', decaying h''
y_guess = np.zeros((3, x.size))
y_guess[0] = x     # h ≈ x
y_guess[1] = 1.0   # h' ≈ 1
y_guess[2] = 0.0   # h'' ≈ 0

# Initial guess for parameter C1
C1_guess = np.array([0.0])

# ----------------------------------------------------------
# Solve the BVP
# ----------------------------------------------------------

print("Running solve_bvp ...")

sol = solve_bvp(
    ode_system,
    bc,
    x,
    y_guess,
    p=C1_guess,
    max_nodes=20000,
    tol=1e-5
)

print("Status:", sol.status)
print("Message:", sol.message)
print("Optimized C1 =", sol.p[0])

# ----------------------------------------------------------
# Extract solution
# ----------------------------------------------------------

x_sol = sol.x
h  = sol.y[0]
h1 = sol.y[1]
h2 = sol.y[2]

# ----------------------------------------------------------
# Plot results
# ----------------------------------------------------------

def theta3(x) :
    return 1 + 0.03*np.log(np.e*x)
analyticalTheta = theta3(np.array(x_sol))
analytivalDer = np.zeros((len(x_sol)))
for i, theta in enumerate(analyticalTheta) :
    analytivalDer[i] = 0.01/(x_sol[i]*pow(theta,2))



#fig,axs =  plt.subplot(figsize=(10,6),shareX=True))
fig, axs = plt.subplots(1, 2, sharex = True, figsize = (25,10))

w = 4


axs[0].plot(x_sol, pow(h1,3), c = "red", linewidth = w, label = "Numerical")
#axs[0].set_yscale('log')
#axs[0].set_xscale('log')
axs[0].set_xlim(1e-3,1e3)
axs[0].set_ylim(0.8,1.25)
axs[0].plot(x_sol, analyticalTheta, c = "black", linestyle = '--', linewidth = w, label = "Analytical")


#plt.xlabel("x")
#plt.ylabel("h'(x)")
#plt.title("First Derivative h'(x)")


axs[1].plot(x_sol, h2, c = "darkgreen", linewidth = w, label = "Numerical")
axs[1].set_yscale('log')
#axs[1].set_xscale('log')
axs[1].plot(x_sol, analytivalDer, c = "black", linestyle = '--', linewidth = w, label = "Analytical")
axs[1].set_ylim(0, 0.1)



s = 20
l = 10
w = 4
axs[0].minorticks_on()
axs[0].set_yticks(np.arange(0.8,1.5,0.1))
axs[0].tick_params(labelsize= s)
axs[0].tick_params(axis='both', which='minor', width= 0.5*w, length = 0.5*l)
axs[0].tick_params(axis='both', which='major', width= w, length = l)


axs[1].minorticks_on()
axs[1].tick_params(labelsize= s)
axs[1].tick_params(axis='both', which='minor', width= 0.5*w, length = 0.5*l)
axs[1].tick_params(axis='both', which='major', width= w, length = l)


s = 30
axs[0].set_ylabel(r"$ \theta^{3} $",fontsize =s)
axs[0].set_xlabel("$x$",fontsize =s)
axs[1].set_ylabel(r"$\frac{d \theta}{d x} $",fontsize = 1.4*s)
axs[1].set_xlabel("$x$",fontsize =s)
axs[0].legend(loc = 'upper left', fontsize =s )
axs[1].legend(loc = 'upper right', fontsize = s)
axs[0].set_title("Contact Angle Cubed", fontsize = s)
axs[1].set_title("Derivative of Contact Angle", fontsize = s)

#plt.xlabel("x")
#plt.ylabel("h''(x)")
#plt.title("Second Derivative h''(x)")

plt.tight_layout()
plt.show()
