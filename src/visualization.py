"""
Static visualization module for the Cone Gyroscope Simulation.

Generates 10 individual plots and 1 comprehensive composite figure
from simulation data. All plot labels and titles are in English.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import config


def setup_matplotlib():
    """Configure matplotlib with safe font fallback.

    Tries common CJK fonts first; falls back to DejaVu Sans if none
    are available. This ensures the code runs on any platform without
    raising font-related errors.
    """
    matplotlib.rcParams['axes.unicode_minus'] = False
    # Try CJK fonts first, fall back to DejaVu Sans
    for font in ['Arial Unicode MS', 'SimHei', 'STSong', 'WenQuanYi Micro Hei']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
        except Exception:
            continue
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']


def plot_angular_velocities(t, omega1, omega2, omega3, output_dir):
    """Plot 1: Angular velocities vs. time.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    omega1, omega2, omega3 : numpy.ndarray
        Angular velocity arrays.
    output_dir : str
        Directory to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, omega1, 'b-', linewidth=1.5, label=r'$\omega_1$ (spin)')
    axes[0].set_ylabel(r'$\omega_1$ (rad/s)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Angular Velocities vs Time', fontsize=14, fontweight='bold')

    axes[1].plot(t, omega2, 'r-', linewidth=1.5, label=r'$\omega_2$ (nutation)')
    axes[1].set_ylabel(r'$\omega_2$ (rad/s)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, omega3, 'g-', linewidth=1.5, label=r'$\omega_3$ (precession)')
    axes[2].set_xlabel('Time t (s)', fontsize=12)
    axes[2].set_ylabel(r'$\omega_3$ (rad/s)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig01_angular_velocities.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_angular_accelerations(t, alpha1, alpha2, alpha3, output_dir):
    """Plot 2: Angular accelerations vs. time.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    alpha1, alpha2, alpha3 : numpy.ndarray
        Angular acceleration arrays.
    output_dir : str
        Directory to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, alpha1, 'b-', linewidth=1.5, label=r'$\alpha_1$ (spin)')
    axes[0].set_ylabel(r'$\alpha_1$ (rad/s$^2$)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Angular Accelerations vs Time', fontsize=14, fontweight='bold')

    axes[1].plot(t, alpha2, 'r-', linewidth=1.5, label=r'$\alpha_2$ (nutation)')
    axes[1].set_ylabel(r'$\alpha_2$ (rad/s$^2$)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, alpha3, 'g-', linewidth=1.5, label=r'$\alpha_3$ (precession)')
    axes[2].set_xlabel('Time t (s)', fontsize=12)
    axes[2].set_ylabel(r'$\alpha_3$ (rad/s$^2$)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig02_angular_accelerations.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_euler_angles(t, theta, phi, psi, output_dir):
    """Plot 3: Euler angles vs. time.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    theta, phi, psi : numpy.ndarray
        Euler angle arrays (rad).
    output_dir : str
        Directory to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, np.degrees(theta), 'b-', linewidth=1.5, label=r'$\theta$ (nutation)')
    axes[0].set_ylabel(r'$\theta$ (deg)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Euler Angles vs Time', fontsize=14, fontweight='bold')

    axes[1].plot(t, np.degrees(phi), 'r-', linewidth=1.5, label=r'$\phi$ (precession)')
    axes[1].set_ylabel(r'$\phi$ (deg)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, np.degrees(psi), 'g-', linewidth=1.5, label=r'$\psi$ (spin)')
    axes[2].set_xlabel('Time t (s)', fontsize=12)
    axes[2].set_ylabel(r'$\psi$ (deg)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig03_euler_angles.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_3d_trajectories(t, traj, output_dir):
    """Plot 4: 3D trajectories of center of mass and top point.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    traj : dict
        Trajectory dictionary from compute_trajectory().
    output_dir : str
        Directory to save the figure.
    """
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(traj['x_c'], traj['y_c'], traj['z_c'], c=t, cmap='viridis', s=1)
    ax1.plot([0], [0], [0], 'ko', markersize=8, label='Pivot O')
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('Center of Mass 3D Trajectory', fontsize=13, fontweight='bold')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Time (s)')

    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(traj['x_top'], traj['y_top'], traj['z_top'],
                           c=t, cmap='plasma', s=1)
    ax2.plot([0], [0], [0], 'ko', markersize=8, label='Pivot O')
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_zlabel('Z (m)', fontsize=11)
    ax2.set_title('Top Point 3D Trajectory', fontsize=13, fontweight='bold')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='Time (s)')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig04_3d_trajectories.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_xy_projections(t, traj, output_dir):
    """Plot 5: XY-plane projection trajectories.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    traj : dict
        Trajectory dictionary from compute_trajectory().
    output_dir : str
        Directory to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = ax1.scatter(traj['x_c'], traj['y_c'], c=t, cmap='viridis', s=2)
    ax1.plot(0, 0, 'ko', markersize=10, label='Pivot O')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('CM Horizontal Projection (XY Plane)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Time (s)')

    scatter2 = ax2.scatter(traj['x_top'], traj['y_top'], c=t, cmap='plasma', s=2)
    ax2.plot(0, 0, 'ko', markersize=10, label='Pivot O')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Top Point Horizontal Projection (XY Plane)',
                  fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Time (s)')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig05_xy_projections.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_energy(t, energy, output_dir):
    """Plot 6: Kinetic, potential, and total energy vs. time.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    energy : dict
        Energy dictionary from compute_energy().
    output_dir : str
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(t, energy['T'], 'r-', linewidth=1.5, label='T (Kinetic)')
    ax.plot(t, energy['V'], 'b-', linewidth=1.5, label='V (Potential)')
    ax.plot(t, energy['E'], 'k--', linewidth=2, label='E (Total)')
    ax.set_xlabel('Time t (s)', fontsize=12)
    ax.set_ylabel('Energy (J)', fontsize=12)
    ax.set_title('Energy vs Time (Numerical Integration Accuracy Check)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate energy error
    E = energy['E']
    energy_error = abs(E[-1] - E[0]) / abs(E[0]) * 100
    ax.text(0.5, 0.95, f'Energy Error: {energy_error:.6f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig06_energy.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_nutation_phase(theta, omega2, output_dir):
    """Plot 7: Nutation phase portrait (theta vs. omega2).

    Parameters
    ----------
    theta : numpy.ndarray
        Nutation angle array.
    omega2 : numpy.ndarray
        Nutation angular velocity array.
    output_dir : str
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(np.degrees(theta), omega2, 'b-', linewidth=1.5)
    ax.set_xlabel(r'Nutation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'Nutation velocity $\omega_2$ (rad/s)', fontsize=12)
    ax.set_title('Nutation Phase Portrait', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.plot(np.degrees(theta[0]), omega2[0], 'go', markersize=12, label='Start')
    ax.plot(np.degrees(theta[-1]), omega2[-1], 'rs', markersize=12, label='End')
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig07_nutation_phase.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_combined_motion(t, theta, phi, omega1, omega3, output_dir):
    """Plot 8: Combined motion curves (4 subplots).

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    theta, phi : numpy.ndarray
        Euler angle arrays.
    omega1, omega3 : numpy.ndarray
        Spin and precession angular velocities.
    output_dir : str
        Directory to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].plot(t, np.degrees(theta), 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time t (s)', fontsize=11)
    axes[0, 0].set_ylabel(r'Nutation angle $\theta$ (deg)', fontsize=11)
    axes[0, 0].set_title('Nutation Angle vs Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, np.degrees(phi), 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time t (s)', fontsize=11)
    axes[0, 1].set_ylabel(r'Precession angle $\phi$ (deg)', fontsize=11)
    axes[0, 1].set_title('Precession Angle vs Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, omega1, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time t (s)', fontsize=11)
    axes[1, 0].set_ylabel(r'Spin velocity $\omega_1$ (rad/s)', fontsize=11)
    axes[1, 0].set_title('Spin Angular Velocity vs Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, omega3, 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time t (s)', fontsize=11)
    axes[1, 1].set_ylabel(r'Precession velocity $\omega_3$ (rad/s)', fontsize=11)
    axes[1, 1].set_title('Precession Angular Velocity vs Time',
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig08_combined_motion.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_angular_velocity_phase(omega1, omega2, omega3, output_dir):
    """Plot 9: Angular velocity 3D phase space.

    Parameters
    ----------
    omega1, omega2, omega3 : numpy.ndarray
        Angular velocity arrays.
    output_dir : str
        Directory to save the figure.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(omega1, omega2, omega3, 'b-', linewidth=1, alpha=0.7)
    ax.scatter([omega1[0]], [omega2[0]], [omega3[0]],
               c='g', s=100, marker='o', label='Start')
    ax.scatter([omega1[-1]], [omega2[-1]], [omega3[-1]],
               c='r', s=100, marker='s', label='End')

    ax.set_xlabel(r'$\omega_1$ (rad/s)', fontsize=11)
    ax.set_ylabel(r'$\omega_2$ (rad/s)', fontsize=11)
    ax.set_zlabel(r'$\omega_3$ (rad/s)', fontsize=11)
    ax.set_title('Angular Velocity Phase Space', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig09_angular_velocity_phase.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_nutation_detail(t, theta, output_dir, t_limit=2.0):
    """Plot 10: Nutation angle detail (first t_limit seconds).

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    theta : numpy.ndarray
        Nutation angle array.
    output_dir : str
        Directory to save the figure.
    t_limit : float, optional
        Time limit for the detail view (s). Default 2.0.
    """
    mask = t < t_limit
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(t[mask], np.degrees(theta[mask]), 'b-', linewidth=2)
    ax.set_xlabel('Time t (s)', fontsize=12)
    ax.set_ylabel(r'Nutation angle $\theta$ (deg)', fontsize=12)
    ax.set_title(f'Nutation Angle Detail (first {t_limit}s) - Nutation Oscillation',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig10_nutation_detail.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def plot_comprehensive(data, output_dir):
    """Generate the comprehensive composite figure (3x3 grid).

    Parameters
    ----------
    data : dict
        Dictionary containing all simulation data (from extract_solution,
        plus 'alpha1', 'alpha2', 'alpha3', trajectory and energy data).
    output_dir : str
        Directory to save the figure.
    """
    t = data['t']
    theta = data['theta']
    phi = data['phi']
    psi = data['psi']
    omega1 = data['omega1']
    omega2 = data['omega2']
    omega3 = data['omega3']
    alpha1 = data['alpha1']
    alpha2 = data['alpha2']
    alpha3 = data['alpha3']
    traj = data['traj']
    energy = data['energy']

    fig = plt.figure(figsize=(20, 16))

    # Title area with initial parameters
    param_text = (
        f"Initial Parameters:\n"
        f"m = {config.MASS} kg,  g = {config.GRAVITY} m/s²,  "
        f"r_c = {config.R_CM} m\n"
        f"J = {config.J_SPIN} kg·m²,  "
        f"J_c = m·r_c² = {config.J_COUPLING:.4f} kg·m²\n"
        f"θ₀ = {np.degrees(config.THETA_0):.1f}°,  "
        f"φ₀ = {np.degrees(config.PHI_0):.1f}°,  "
        f"ψ₀ = {np.degrees(config.PSI_0):.1f}°\n"
        f"ω₁₀ = {config.OMEGA1_0:.1f} rad/s "
        f"({config.OMEGA1_0 * 60 / (2 * np.pi):.1f} rpm),  "
        f"ω₂₀ = {config.OMEGA2_0:.1f} rad/s,  "
        f"ω₃₀ = {config.OMEGA3_0:.1f} rad/s"
    )
    fig.suptitle('Cone Gyroscope Simulation - Comprehensive View\n' + param_text,
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Angular velocities (top-left)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(t, omega1, 'b-', linewidth=1.5, label=r'$\omega_1$ (spin)')
    ax1.plot(t, omega2, 'r-', linewidth=1.5, label=r'$\omega_2$ (nutation)')
    ax1.plot(t, omega3, 'g-', linewidth=1.5, label=r'$\omega_3$ (precession)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('1. Angular Velocities vs Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Angular accelerations (top-center)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(t, alpha1, 'b-', linewidth=1.5, label=r'$\alpha_1$ (spin)')
    ax2.plot(t, alpha2, 'r-', linewidth=1.5, label=r'$\alpha_2$ (nutation)')
    ax2.plot(t, alpha3, 'g-', linewidth=1.5, label=r'$\alpha_3$ (precession)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Acceleration (rad/s²)')
    ax2.set_title('2. Angular Accelerations vs Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Euler angles (top-right)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3_twin = ax3.twinx()
    line1, = ax3.plot(t, np.degrees(theta), 'b-', linewidth=2, label=r'$\theta$ (nutation)')
    line2, = ax3_twin.plot(t, np.degrees(phi), 'r-', linewidth=1.5,
                           label=r'$\phi$ (precession)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'$\theta$ (deg)', color='b')
    ax3_twin.set_ylabel(r'$\phi$ (deg)', color='r')
    ax3.set_title('3. Euler Angles vs Time')
    ax3.legend([line1, line2], [r'$\theta$ (nutation)', r'$\phi$ (precession)'],
               loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. 3D trajectory (middle-left)
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    ax4.scatter(traj['x_top'], traj['y_top'], traj['z_top'], c=t, cmap='viridis', s=1)
    ax4.plot([0], [0], [0], 'ko', markersize=8, label='Pivot O')
    ax4.plot([0, traj['x_top'][0]], [0, traj['y_top'][0]], [0, traj['z_top'][0]],
             'r-', linewidth=2, label='Initial axis')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('4. 3D Trajectory (Top Point)')
    ax4.legend(loc='upper left')

    # 5. XY projection (middle-center)
    ax5 = fig.add_subplot(3, 3, 5)
    scatter5 = ax5.scatter(traj['x_top'], traj['y_top'], c=t, cmap='plasma', s=2)
    ax5.plot(0, 0, 'ko', markersize=10, label='Pivot O')
    ax5.plot(traj['x_top'][0], traj['y_top'][0], 'g^', markersize=10, label='Start')
    ax5.plot(traj['x_top'][-1], traj['y_top'][-1], 'rs', markersize=8, label='End')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('5. XY Projection Trajectory')
    ax5.set_aspect('equal')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax5, label='Time (s)')

    # 6. Energy (middle-right)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(t, energy['T'], 'r-', linewidth=1.5, label='T (Kinetic)')
    ax6.plot(t, energy['V'], 'b-', linewidth=1.5, label='V (Potential)')
    ax6.plot(t, energy['E'], 'k--', linewidth=2, label='E (Total)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Energy (J)')
    ax6.set_title('6. Energy vs Time')
    ax6.legend(loc='right')
    ax6.grid(True, alpha=0.3)
    E = energy['E']
    energy_error = abs(E[-1] - E[0]) / abs(E[0]) * 100
    ax6.text(0.5, 0.95, f'Energy Error: {energy_error:.6f}%',
             transform=ax6.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

    # 7. Nutation phase portrait (bottom-left)
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(np.degrees(theta), omega2, 'b-', linewidth=1.5)
    ax7.plot(np.degrees(theta[0]), omega2[0], 'go', markersize=10, label='Start')
    ax7.plot(np.degrees(theta[-1]), omega2[-1], 'rs', markersize=8, label='End')
    ax7.set_xlabel(r'$\theta$ (deg)')
    ax7.set_ylabel(r'$\omega_2$ (rad/s)')
    ax7.set_title('7. Nutation Phase Portrait')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Combined motion curves (bottom-center)
    ax8 = fig.add_subplot(3, 3, 8)
    ax8_twin1 = ax8.twinx()
    ax8_twin2 = ax8.twinx()
    ax8_twin2.spines['right'].set_position(('outward', 60))

    line8_1, = ax8.plot(t, np.degrees(theta), 'b-', linewidth=2, label=r'$\theta$ (deg)')
    line8_2, = ax8_twin1.plot(t, omega1, 'r-', linewidth=1.5,
                              label=r'$\omega_1$ (rad/s)')
    line8_3, = ax8_twin2.plot(t, omega3, 'g-', linewidth=1.5,
                              label=r'$\omega_3$ (rad/s)')

    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel(r'$\theta$ (deg)', color='b')
    ax8_twin1.set_ylabel(r'$\omega_1$ (rad/s)', color='r')
    ax8_twin2.set_ylabel(r'$\omega_3$ (rad/s)', color='g')
    ax8.set_title('8. Combined Motion Curves')
    ax8.legend([line8_1, line8_2, line8_3], [r'$\theta$', r'$\omega_1$', r'$\omega_3$'],
               loc='upper left')
    ax8.grid(True, alpha=0.3)

    # 9. Angular velocity phase space (bottom-right)
    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    ax9.plot(omega1, omega2, omega3, 'b-', linewidth=1, alpha=0.7)
    ax9.scatter([omega1[0]], [omega2[0]], [omega3[0]],
                c='g', s=80, marker='o', label='Start')
    ax9.scatter([omega1[-1]], [omega2[-1]], [omega3[-1]],
                c='r', s=80, marker='s', label='End')
    ax9.set_xlabel(r'$\omega_1$ (rad/s)')
    ax9.set_ylabel(r'$\omega_2$ (rad/s)')
    ax9.set_zlabel(r'$\omega_3$ (rad/s)')
    ax9.set_title('9. Angular Velocity Phase Space')
    ax9.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_dir, 'comprehensive_figure.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {filepath}")


def generate_all_plots(data, output_dir):
    """Generate all 10 individual plots and the comprehensive composite figure.

    Parameters
    ----------
    data : dict
        Dictionary containing all simulation data.
    output_dir : str
        Directory to save the figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_matplotlib()

    print("\nGenerating visualization plots...")

    t = data['t']
    theta = data['theta']
    phi = data['phi']
    psi = data['psi']
    omega1 = data['omega1']
    omega2 = data['omega2']
    omega3 = data['omega3']
    alpha1 = data['alpha1']
    alpha2 = data['alpha2']
    alpha3 = data['alpha3']
    traj = data['traj']
    energy = data['energy']

    plot_angular_velocities(t, omega1, omega2, omega3, output_dir)
    plot_angular_accelerations(t, alpha1, alpha2, alpha3, output_dir)
    plot_euler_angles(t, theta, phi, psi, output_dir)
    plot_3d_trajectories(t, traj, output_dir)
    plot_xy_projections(t, traj, output_dir)
    plot_energy(t, energy, output_dir)
    plot_nutation_phase(theta, omega2, output_dir)
    plot_combined_motion(t, theta, phi, omega1, omega3, output_dir)
    plot_angular_velocity_phase(omega1, omega2, omega3, output_dir)
    plot_nutation_detail(t, theta, output_dir)

    plot_comprehensive(data, output_dir)

    print(f"\n  All plots saved to '{output_dir}/'")
