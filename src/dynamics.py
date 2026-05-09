"""
Core physics engine for the Cone Gyroscope Precession and Nutation Simulation.

Implements the equations of motion derived from Euler-Lagrange formulation
for a cone gyroscope with a fixed pivot point. The dynamical equations are
solved using scipy's ODE integrator.

Coordinate convention
---------------------
- omega1: spin angular velocity (about the symmetry axis)
- omega2: nutation angular velocity (d(theta)/dt)
- omega3: precession angular velocity (d(phi)/dt)

The relationship between angular velocities and Euler angle rates:
    omega2 = d(theta)/dt
    omega3 = d(phi)/dt
    omega1 = d(psi)/dt + d(phi)/dt * cos(theta)

Equations of motion
-------------------
Eq.1: J*alpha1*sin(theta) + J*omega1*omega2*cos(theta)
       - 0.5*J_c*alpha3*sin(2*theta) - J_c*omega2*omega3*cos(2*theta)
       - J_c*omega2*omega3 = 0
Eq.2: J*omega1*omega3*sin(theta) - 0.5*J_c*omega3^2*sin(2*theta)
       + J_c*alpha2 = m*g*r_c*sin(theta)
Eq.3: J*alpha1*cos(theta) - J*omega1*omega2*sin(theta)
       + J_c*alpha3*sin^2(theta) + J_c*omega2*omega3*sin(2*theta) = 0
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import config


def solve_accelerations(theta, omega1, omega2, omega3):
    """Solve for angular accelerations from the three dynamical equations.

    alpha2 is obtained directly from Equation 2.
    alpha1 and alpha3 are obtained by solving the 2x2 linear system
    formed by Equations 1 and 3.

    Parameters
    ----------
    theta : float
        Nutation angle (rad).
    omega1 : float
        Spin angular velocity (rad/s).
    omega2 : float
        Nutation angular velocity (rad/s).
    omega3 : float
        Precession angular velocity (rad/s).

    Returns
    -------
    tuple of float
        (alpha1, alpha2, alpha3) angular accelerations (rad/s^2).
    """
    m = config.MASS
    g = config.GRAVITY
    r_c = config.R_CM
    J = config.J_SPIN
    J_c = config.J_COUPLING

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)

    # Solve alpha2 directly from Equation 2
    alpha2 = (m * g * r_c * sin_theta
              - J * omega1 * omega3 * sin_theta
              + 0.5 * J_c * omega3**2 * sin_2theta) / J_c

    # Solve alpha1 and alpha3 from Equations 1 & 3 (2x2 linear system)
    # Eq.1: J*sin(theta)*alpha1 - 0.5*J_c*sin(2*theta)*alpha3
    #       = -J*omega1*omega2*cos(theta) + J_c*omega2*omega3*cos(2*theta)
    #         + J_c*omega2*omega3
    # Eq.3: J*cos(theta)*alpha1 + J_c*sin^2(theta)*alpha3
    #       = J*omega1*omega2*sin(theta) - J_c*omega2*omega3*sin(2*theta)

    A = np.array([
        [J * sin_theta, -0.5 * J_c * sin_2theta],
        [J * cos_theta, J_c * sin_theta**2]
    ])

    b1 = (-J * omega1 * omega2 * cos_theta
           + J_c * omega2 * omega3 * cos_2theta
           + J_c * omega2 * omega3)
    b2 = (J * omega1 * omega2 * sin_theta
          - J_c * omega2 * omega3 * sin_2theta)
    b = np.array([b1, b2])

    try:
        alpha1, alpha3 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        alpha1, alpha3 = 0.0, 0.0

    return alpha1, alpha2, alpha3


def equations_of_motion(t, y):
    """Define the system of ordinary differential equations.

    State vector: y = [theta, phi, psi, omega1, omega2, omega3]

    Parameters
    ----------
    t : float
        Current time (not explicitly used, system is autonomous).
    y : array_like
        Current state vector.

    Returns
    -------
    list of float
        Time derivatives of the state vector.
    """
    theta, phi, psi, omega1, omega2, omega3 = y

    # Solve for angular accelerations
    alpha1, alpha2, alpha3 = solve_accelerations(theta, omega1, omega2, omega3)

    # Euler angle rates from angular velocity relations
    dtheta_dt = omega2
    dphi_dt = omega3
    dpsi_dt = omega1 - omega3 * np.cos(theta)

    return [dtheta_dt, dphi_dt, dpsi_dt, alpha1, alpha2, alpha3]


def simulate(y0=None, t_span=None, t_eval=None, method=None,
             rtol=None, atol=None):
    """Run the numerical simulation of the cone gyroscope.

    Parameters
    ----------
    y0 : list of float, optional
        Initial state vector. Defaults to config values.
    t_span : tuple of float, optional
        Time span (t_start, t_end). Defaults to config values.
    t_eval : numpy.ndarray, optional
        Time points for solution output. Defaults to config values.
    method : str, optional
        ODE solver method. Defaults to config value.
    rtol : float, optional
        Relative tolerance. Defaults to config value.
    atol : float, optional
        Absolute tolerance. Defaults to config value.

    Returns
    -------
    scipy.integrate.OdeResult
        Solution object with attributes t, y, etc.
    """
    if y0 is None:
        y0 = config.get_initial_state()
    if t_span is None:
        t_span = config.T_SPAN
    if t_eval is None:
        t_eval = config.get_time_eval()
    if method is None:
        method = config.METHOD
    if rtol is None:
        rtol = config.RTOL
    if atol is None:
        atol = config.ATOL

    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    return sol


def compute_angular_accelerations(t, theta, omega1, omega2, omega3):
    """Compute angular accelerations at all time points.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    theta : numpy.ndarray
        Nutation angle array.
    omega1, omega2, omega3 : numpy.ndarray
        Angular velocity arrays.

    Returns
    -------
    tuple of numpy.ndarray
        (alpha1, alpha2, alpha3) angular acceleration arrays.
    """
    alpha1_arr = np.zeros_like(t)
    alpha2_arr = np.zeros_like(t)
    alpha3_arr = np.zeros_like(t)

    for i in range(len(t)):
        alpha1_arr[i], alpha2_arr[i], alpha3_arr[i] = solve_accelerations(
            theta[i], omega1[i], omega2[i], omega3[i]
        )

    return alpha1_arr, alpha2_arr, alpha3_arr


def compute_trajectory(theta, phi, r_c=None, h=None):
    """Compute 3D trajectory of the center of mass and the top point.

    Parameters
    ----------
    theta : numpy.ndarray
        Nutation angle array.
    phi : numpy.ndarray
        Precession angle array.
    r_c : float, optional
        Distance from pivot to center of mass. Defaults to config.R_CM.
    h : float, optional
        Height for top-point trajectory. Defaults to config.GYRO_HEIGHT.

    Returns
    -------
    dict
        Dictionary with keys 'x_c', 'y_c', 'z_c', 'x_top', 'y_top', 'z_top'.
    """
    if r_c is None:
        r_c = config.R_CM
    if h is None:
        h = config.GYRO_HEIGHT

    x_c = r_c * np.sin(theta) * np.cos(phi)
    y_c = r_c * np.sin(theta) * np.sin(phi)
    z_c = r_c * np.cos(theta)

    x_top = h * np.sin(theta) * np.cos(phi) + x_c
    y_top = h * np.sin(theta) * np.sin(phi) + y_c
    z_top = h * np.cos(theta) + z_c

    return {
        'x_c': x_c, 'y_c': y_c, 'z_c': z_c,
        'x_top': x_top, 'y_top': y_top, 'z_top': z_top
    }


def compute_energy(theta, omega1, omega2, omega3):
    """Compute kinetic, potential, and total energy.

    Kinetic energy:
        T = 0.5 * J * omega1^2
          + 0.5 * J_c * (omega2^2 + omega3^2 * sin^2(theta))

    Potential energy (zero at the pivot point, i.e., V=0 when theta=pi/2):
        V = m * g * r_c * cos(theta)

    Parameters
    ----------
    theta : numpy.ndarray
        Nutation angle array.
    omega1, omega2, omega3 : numpy.ndarray
        Angular velocity arrays.

    Returns
    -------
    dict
        Dictionary with keys 'T' (kinetic), 'V' (potential), 'E' (total).
    """
    m = config.MASS
    g = config.GRAVITY
    r_c = config.R_CM
    J = config.J_SPIN
    J_c = config.J_COUPLING

    T = 0.5 * J * omega1**2 + 0.5 * J_c * (omega2**2 + omega3**2 * np.sin(theta)**2)
    V = m * g * r_c * np.cos(theta)
    E = T + V

    return {'T': T, 'V': V, 'E': E}


def extract_solution(sol):
    """Extract named arrays from an OdeResult object.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution from simulate().

    Returns
    -------
    dict
        Dictionary with keys: t, theta, phi, psi, omega1, omega2, omega3.
    """
    return {
        't': sol.t,
        'theta': sol.y[0],
        'phi': sol.y[1],
        'psi': sol.y[2],
        'omega1': sol.y[3],
        'omega2': sol.y[4],
        'omega3': sol.y[5]
    }


def print_parameters():
    """Print physical parameters to stdout."""
    m = config.MASS
    g = config.GRAVITY
    r_c = config.R_CM
    J = config.J_SPIN
    J_c = config.J_COUPLING

    print("=" * 60)
    print("Cone Gyroscope Physical Parameters:")
    print(f"  Mass           m   = {m} kg")
    print(f"  Gravity        g   = {g} m/s^2")
    print(f"  CM distance    r_c = {r_c} m")
    print(f"  Spin inertia   J   = {J} kg*m^2")
    print(f"  Coupling inertia J_c = {J_c:.6f} kg*m^2")
    print("=" * 60)


def print_initial_conditions():
    """Print initial conditions to stdout."""
    print("\nInitial Conditions:")
    print(f"  Nutation angle      theta_0 = {np.degrees(config.THETA_0):.1f} deg")
    print(f"  Precession angle    phi_0   = {np.degrees(config.PHI_0):.1f} deg")
    print(f"  Spin angle          psi_0   = {np.degrees(config.PSI_0):.1f} deg")
    print(f"  Spin velocity       omega1_0 = {config.OMEGA1_0:.1f} rad/s "
          f"({config.OMEGA1_0 * 60 / (2 * np.pi):.1f} rpm)")
    print(f"  Nutation velocity   omega2_0 = {config.OMEGA2_0:.1f} rad/s")
    print(f"  Precession velocity omega3_0 = {config.OMEGA3_0:.1f} rad/s")
    print("=" * 60)


def print_results(data):
    """Print simulation results summary.

    Parameters
    ----------
    data : dict
        Dictionary from extract_solution() with additional keys from
        compute_energy() and compute_trajectory().
    """
    t = data['t']
    theta = data['theta']
    phi = data['phi']
    psi = data['psi']
    omega1 = data['omega1']
    omega2 = data['omega2']
    omega3 = data['omega3']
    E = data.get('energy', {}).get('E', None)
    if E is None:
        E = data.get('E', np.zeros_like(t))

    print("\n" + "=" * 60)
    print("Simulation Results Summary:")
    print("=" * 60)
    print(f"  Simulation time: {t[0]:.1f} ~ {t[-1]:.1f} s")
    print(f"  Time steps: {len(t)}")
    print(f"\n  Nutation angle range: "
          f"{np.degrees(theta).min():.2f} deg ~ {np.degrees(theta).max():.2f} deg")
    print(f"  Precession angle final: {np.degrees(phi[-1]):.2f} deg")
    print(f"  Spin angle final: {np.degrees(psi[-1]):.2f} deg")
    print(f"\n  Spin velocity:      {omega1[0]:.2f} -> {omega1[-1]:.2f} rad/s")
    print(f"  Nutation velocity:  {omega2[0]:.2f} -> {omega2[-1]:.2f} rad/s")
    print(f"  Precession velocity: {omega3[0]:.2f} -> {omega3[-1]:.2f} rad/s")
    if len(E) > 0 and E[0] != 0:
        print(f"\n  Total energy: {E[0]:.6f} J -> {E[-1]:.6f} J")
        print(f"  Energy relative error: {abs(E[-1] - E[0]) / abs(E[0]) * 100:.6f}%")
    print("=" * 60)
