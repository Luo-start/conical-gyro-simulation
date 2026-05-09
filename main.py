"""
Main entry point for the Cone Gyroscope Precession and Nutation Simulation.

Usage examples:
    # Run simulation and generate all plots
    python main.py
    # python -m Gyro-Simulation

    # Run with animations
    python main.py --animation
    # python -m Gyro-Simulation --animation

    # Run diagnostic
    python main.py --diagnostic
    # python -m Gyro-Simulation --diagnostic

    # Specify output directory
    python main.py --output results/
    # python -m Gyro-Simulation --output results/

    # Custom initial conditions
    python main.py --omega1 100 --omega3 3 --theta0 45
    # python -m Gyro-Simulation --omega1 100 --omega3 3 --theta0 45
"""

import argparse
import os
import sys

import numpy as np

from src import config
from src.dynamics import (
    simulate,
    extract_solution,
    compute_angular_accelerations,
    compute_trajectory,
    compute_energy,
    print_parameters,
    print_results,
)
from src.visualization import generate_all_plots



def run_diagnostic(data):
    """Run energy conservation diagnostic analysis.

    Checks the consistency of the energy formula with the dynamical
    equations and reports potential parameter mismatches.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    """
    m = config.MASS
    g = config.GRAVITY
    r_c = config.R_CM
    J = config.J_SPIN
    J_c = config.J_COUPLING

    print("=" * 60)
    print("Energy Conservation Diagnostic")
    print("=" * 60)

    print(f"\nParameter definitions:")
    print(f"  m   = {m} kg")
    print(f"  r_c = {r_c} m  (distance from pivot to center of mass)")
    print(f"  J   = {J} kg*m^2  (moment of inertia about spin axis)")
    print(f"  J_c = m*r_c^2 = {J_c:.6f} kg*m^2")
    print(f"  g   = {g} m/s^2")

    # Check energy conservation
    E = data['energy']['E']
    E_initial = E[0]
    E_final = E[-1]
    E_max = np.max(E)
    E_min = np.min(E)
    relative_error = abs(E_final - E_initial) / abs(E_initial) * 100
    max_fluctuation = (E_max - E_min) / abs(E_initial) * 100

    print(f"\nEnergy conservation check:")
    print(f"  E_initial    = {E_initial:.8f} J")
    print(f"  E_final      = {E_final:.8f} J")
    print(f"  E_max        = {E_max:.8f} J")
    print(f"  E_min        = {E_min:.8f} J")
    print(f"  Relative error (final vs initial): {relative_error:.8f}%")
    print(f"  Max fluctuation: {max_fluctuation:.8f}%")

    # Check angular momentum-like quantities
    theta = data['theta']
    omega1 = data['omega1']
    omega3 = data['omega3']

    # J*omega1 should be approximately conserved (spin angular momentum component)
    L_spin = J * omega1
    L_spin_error = abs(L_spin[-1] - L_spin[0]) / abs(L_spin[0]) * 100

    # J_c * omega3 * sin^2(theta) + J * omega1 * cos(theta) should be conserved
    L_z = J_c * omega3 * np.sin(theta)**2 + J * omega1 * np.cos(theta)
    L_z_error = abs(L_z[-1] - L_z[0]) / abs(L_z[0]) * 100

    print(f"\nConserved quantity checks:")
    print(f"  J*omega1 relative error: {L_spin_error:.6f}%")
    print(f"  L_z = J_c*omega3*sin^2(theta) + J*omega1*cos(theta)")
    print(f"  L_z relative error: {L_z_error:.6f}%")

    if relative_error < 0.01:
        print(f"\n  [PASS] Energy conservation is excellent (error < 0.01%)")
    elif relative_error < 0.1:
        print(f"\n  [OK] Energy conservation is good (error < 0.1%)")
    else:
        print(f"\n  [WARN] Energy conservation may need improvement (error >= 0.1%)")

    print("=" * 60)


def parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Cone Gyroscope Precession and Nutation Simulation'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=config.OUTPUT_DIR,
        help=f'Output directory for plots and animations (default: {config.OUTPUT_DIR})'
    )
    parser.add_argument(
        '--animation', '-a',
        action='store_true',
        help='Generate GIF animations in addition to static plots'
    )
    parser.add_argument(
        '--diagnostic', '-d',
        action='store_true',
        help='Run energy conservation diagnostic analysis'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip static plot generation (only run simulation)'
    )
    parser.add_argument(
        '--theta0',
        type=float,
        default=None,
        help='Initial nutation angle in degrees (default: from config)'
    )
    parser.add_argument(
        '--omega1',
        type=float,
        default=None,
        help='Initial spin angular velocity in rad/s (default: from config)'
    )
    parser.add_argument(
        '--omega2',
        type=float,
        default=None,
        help='Initial nutation angular velocity in rad/s (default: from config)'
    )
    parser.add_argument(
        '--omega3',
        type=float,
        default=None,
        help='Initial precession angular velocity in rad/s (default: from config)'
    )
    parser.add_argument(
        '--t-end',
        type=float,
        default=None,
        help='Simulation end time in seconds (default: from config)'
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Main execution function.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.
    """
    args = parse_args(argv)

    # Build initial conditions from config, override with CLI args if provided
    y0 = config.get_initial_state()
    if args.theta0 is not None:
        y0[0] = np.radians(args.theta0)
    if args.omega1 is not None:
        y0[3] = args.omega1
    if args.omega2 is not None:
        y0[4] = args.omega2
    if args.omega3 is not None:
        y0[5] = args.omega3

    t_span = list(config.T_SPAN)
    if args.t_end is not None:
        t_span[1] = args.t_end

    # Print setup
    print_parameters()
    # Print actual initial conditions (may differ from config defaults if CLI args provided)
    print("\nInitial Conditions:")
    print(f"  Nutation angle      theta_0 = {np.degrees(y0[0]):.1f} deg")
    print(f"  Precession angle    phi_0   = {np.degrees(y0[1]):.1f} deg")
    print(f"  Spin angle          psi_0   = {np.degrees(y0[2]):.1f} deg")
    print(f"  Spin velocity       omega1_0 = {y0[3]:.1f} rad/s "
          f"({y0[3] * 60 / (2 * np.pi):.1f} rpm)")
    print(f"  Nutation velocity   omega2_0 = {y0[4]:.1f} rad/s")
    print(f"  Precession velocity omega3_0 = {y0[5]:.1f} rad/s")
    print("=" * 60)

    # Run simulation
    print("\nStarting numerical integration...")
    sol = simulate(y0=y0, t_span=tuple(t_span))

    if not sol.success:
        print(f"Integration failed: {sol.message}")
        sys.exit(1)
    print("Integration successful!")

    # Extract and compute derived quantities
    data = extract_solution(sol)
    alpha1, alpha2, alpha3 = compute_angular_accelerations(
        data['t'], data['theta'], data['omega1'], data['omega2'], data['omega3']
    )
    data['alpha1'] = alpha1
    data['alpha2'] = alpha2
    data['alpha3'] = alpha3
    data['traj'] = compute_trajectory(data['theta'], data['phi'])
    data['energy'] = compute_energy(
        data['theta'], data['omega1'], data['omega2'], data['omega3']
    )

    # Print results
    print_results(data)

    # Generate plots
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_plots:
        generate_all_plots(data, output_dir)

    # Run diagnostic if requested
    if args.diagnostic:
        run_diagnostic(data)

    # Generate animations if requested (lazy import to avoid unused-import warnings)
    if args.animation:
        from src.animation import generate_all_animations
        generate_all_animations(output_dir)

    print(f"\nDone. Output saved to '{output_dir}/'")


if __name__ == '__main__':
    main()
