"""
Configuration module for the Cone Gyroscope Precession and Nutation Simulation.

Defines physical parameters, initial conditions, integration settings,
and output path configuration.
"""

import os

# ========================= Physical Parameters =========================
# Typical cone gyroscope parameters (adjust according to actual setup)

MASS = 0.5             # Mass of the rigid body (kg)
GRAVITY = 9.81         # Gravitational acceleration (m/s^2)
R_CM = 0.08            # Distance from center of mass to pivot point (m)
J_SPIN = 0.002         # Moment of inertia about the spin axis (kg*m^2)
J_COUPLING = MASS * R_CM**2  # Equivalent coupling moment of inertia (kg*m^2)

# ========================= Initial Conditions ==========================
# Default initial conditions for the main simulation

THETA_0 = 3.141592653589793 / 6   # Initial nutation angle: 30 degrees (rad)
PHI_0 = 0.0                       # Initial precession angle (rad)
PSI_0 = 0.0                       # Initial spin angle (rad)
OMEGA1_0 = 50.0                    # Initial spin angular velocity (rad/s)
OMEGA2_0 = 0.0                     # Initial nutation angular velocity (rad/s)
OMEGA3_0 = 2.0                     # Initial precession angular velocity (rad/s)

# ========================= Integration Settings ========================
# Numerical integration parameters

T_SPAN = (0, 3)              # Simulation time interval (s)
N_EVAL_POINTS = 1500          # Number of evaluation time points
METHOD = "DOP853"             # ODE solver method (DOP853 for high precision)
RTOL = 1e-10                  # Relative tolerance
ATOL = 1e-12                  # Absolute tolerance

# ========================= Visualization Settings ======================
# Geometry parameters for plotting

GYRO_HEIGHT = 0.15            # Gyroscope height for trajectory display (m)
ANIM_GYRO_HEIGHT = 0.12       # Gyroscope height used in animations (m)
ANIM_DISK_RADIUS = 0.025      # Disk radius for gyroscope schematic animation (m)

# ========================= Output Configuration ========================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")   # Default output directory relative to project root


def get_initial_state():
    """Return the initial state vector as a list.

    Returns
    -------
    list of float
        State vector [theta, phi, psi, omega1, omega2, omega3].
    """
    return [THETA_0, PHI_0, PSI_0, OMEGA1_0, OMEGA2_0, OMEGA3_0]


def get_time_eval():
    """Return the array of time evaluation points.

    Returns
    -------
    numpy.ndarray
        Linearly spaced time points over T_SPAN.
    """
    import numpy as np
    return np.linspace(T_SPAN[0], T_SPAN[1], N_EVAL_POINTS)
