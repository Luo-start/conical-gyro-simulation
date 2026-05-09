"""
Animation module for the Cone Gyroscope Simulation.

Generates GIF animations showing the gyroscope motion from multiple
viewpoints: 3D trajectory, XY-plane projection, multi-view composite,
and a spinning-top schematic with disk.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import config
from .dynamics import simulate, extract_solution, compute_trajectory


def setup_matplotlib():
    """Configure matplotlib with safe font fallback for animations."""
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def _get_animation_params(omega1_0=150.0, omega3_0=5.0,
                          t_end=5.0, n_points=250):
    """Return animation-specific simulation parameters.

    Animations use higher spin speed and longer duration for better
    visual effect. These are separate from the main simulation defaults.

    Parameters
    ----------
    omega1_0 : float
        Initial spin angular velocity (rad/s).
    omega3_0 : float
        Initial precession angular velocity (rad/s).
    t_end : float
        End time (s).
    n_points : int
        Number of evaluation points.

    Returns
    -------
    dict
        Simulation keyword arguments for dynamics.simulate().
    """
    return {
        'y0': [config.THETA_0, config.PHI_0, config.PSI_0,
                omega1_0, config.OMEGA2_0, omega3_0],
        't_span': (0, t_end),
        't_eval': np.linspace(0, t_end, n_points),
        'method': 'RK45',
        'rtol': 1e-8,
        'atol': 1e-10,
    }


def animate_3d_motion(sol, traj, output_dir, fps=25, dpi=80):
    """Animation 1: 3D gyroscope motion with physical quantities.

    Left panel: 3D trajectory and spin axis.
    Right panels: Euler angles and angular velocities vs. time.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        ODE solution.
    traj : dict
        Trajectory dictionary from compute_trajectory().
    output_dir : str
        Directory to save the GIF.
    fps : int
        Frames per second for the output GIF.
    dpi : int
        DPI for the output GIF.
    """
    data = extract_solution(sol)
    t = data['t']
    theta = data['theta']
    phi = data['phi']
    omega1 = data['omega1']
    omega3 = data['omega3']

    x_top = traj['x_top']
    y_top = traj['y_top']
    z_top = traj['z_top']

    fig = plt.figure(figsize=(14, 6))

    # Left: 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    # Right: physical quantities
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    # 3D elements
    trajectory_line, = ax1.plot([], [], [], 'b-', linewidth=0.8, alpha=0.5,
                                label='Trajectory')
    axis_line, = ax1.plot([], [], [], 'r-', linewidth=3, label='Spin axis')
    top_point, = ax1.plot([], [], [], 'ro', markersize=10)
    ax1.plot([0], [0], [0], 'ko', markersize=12, label='Pivot')

    # Physical quantity curves
    line_theta, = ax2.plot([], [], 'b-', linewidth=1.5,
                           label=r'$\theta$ (nutation)')
    line_phi, = ax2.plot([], [], 'r-', linewidth=1.5,
                         label=r'$\phi$ (precession)')
    time_marker, = ax2.plot([], [], 'ko', markersize=8)

    line_w1, = ax3.plot([], [], 'b-', linewidth=1.5, label=r'$\omega_1$')
    line_w3, = ax3.plot([], [], 'g-', linewidth=1.5, label=r'$\omega_3$')
    time_marker2, = ax3.plot([], [], 'ko', markersize=8)

    # 3D axes setup
    ax1.set_xlim(-0.15, 0.15)
    ax1.set_ylim(-0.15, 0.15)
    ax1.set_zlim(-0.02, 0.15)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Cone Gyroscope 3D Motion', fontsize=14, fontweight='bold')

    # Physical quantity axes setup
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, max(np.degrees(phi)) * 1.1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (deg)')
    ax2.set_title('Euler Angles vs Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(0, max(omega1) * 1.1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_title('Angular Velocities vs Time')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    time_text = ax1.text2D(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12)

    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        axis_line.set_data([], [])
        axis_line.set_3d_properties([])
        top_point.set_data([], [])
        top_point.set_3d_properties([])
        line_theta.set_data([], [])
        line_phi.set_data([], [])
        time_marker.set_data([], [])
        line_w1.set_data([], [])
        line_w3.set_data([], [])
        time_marker2.set_data([], [])
        time_text.set_text('')
        return (trajectory_line, axis_line, top_point, line_theta, line_phi,
                time_marker, line_w1, line_w3, time_marker2, time_text)

    def animate(i):
        trajectory_line.set_data(x_top[:i+1], y_top[:i+1])
        trajectory_line.set_3d_properties(z_top[:i+1])
        axis_line.set_data([0, x_top[i]], [0, y_top[i]])
        axis_line.set_3d_properties([0, z_top[i]])
        top_point.set_data([x_top[i]], [y_top[i]])
        top_point.set_3d_properties([z_top[i]])

        line_theta.set_data(t[:i+1], np.degrees(theta[:i+1]))
        line_phi.set_data(t[:i+1], np.degrees(phi[:i+1]) / 10)
        time_marker.set_data([t[i]], [np.degrees(theta[i])])

        line_w1.set_data(t[:i+1], omega1[:i+1])
        line_w3.set_data(t[:i+1], omega3[:i+1] * 20)
        time_marker2.set_data([t[i]], [omega1[i]])

        time_text.set_text(f'Time: {t[i]:.2f} s')
        return (trajectory_line, axis_line, top_point, line_theta, line_phi,
                time_marker, line_w1, line_w3, time_marker2, time_text)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'anim01_3d_motion.gif')
    anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {filepath}")


def animate_xy_trajectory(sol, traj, output_dir, fps=25, dpi=80):
    """Animation 2: XY-plane projection trajectory.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        ODE solution.
    traj : dict
        Trajectory dictionary from compute_trajectory().
    output_dir : str
        Directory to save the GIF.
    fps : int
        Frames per second.
    dpi : int
        DPI.
    """
    data = extract_solution(sol)
    t = data['t']
    theta = data['theta']
    phi = data['phi']

    x_top = traj['x_top']
    y_top = traj['y_top']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#f0f0f0')

    trajectory, = ax.plot([], [], 'b-', linewidth=1, alpha=0.6, label='Trajectory')
    current_pos, = ax.plot([], [], 'ro', markersize=12, label='Top position')
    ax.plot([0], [0], 'k^', markersize=15, label='Pivot O')
    trail, = ax.plot([], [], 'r-', linewidth=3, alpha=0.8)

    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Gyroscope Top Trajectory (XY Plane)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    angle_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=11,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def init():
        trajectory.set_data([], [])
        current_pos.set_data([], [])
        trail.set_data([], [])
        time_text.set_text('')
        angle_text.set_text('')
        return trajectory, current_pos, trail, time_text, angle_text

    def animate(i):
        trajectory.set_data(x_top[:i+1], y_top[:i+1])
        trail_len = min(20, i)
        trail.set_data(x_top[max(0, i - trail_len):i+1],
                       y_top[max(0, i - trail_len):i+1])
        current_pos.set_data([x_top[i]], [y_top[i]])

        time_text.set_text(f'Time: {t[i]:.2f} s')
        angle_text.set_text(
            f'θ={np.degrees(theta[i]):.1f}°  '
            f'φ={np.degrees(phi[i]) % 360:.1f}°'
        )
        return trajectory, current_pos, trail, time_text, angle_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    filepath = os.path.join(output_dir, 'anim02_xy_trajectory.gif')
    anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {filepath}")


def animate_multi_view(sol, traj, output_dir, fps=25, dpi=80):
    """Animation 3: Multi-view composite animation.

    Four panels: 3D view, XY projection, Euler angles, angular velocities.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        ODE solution.
    traj : dict
        Trajectory dictionary from compute_trajectory().
    output_dir : str
        Directory to save the GIF.
    fps : int
        Frames per second.
    dpi : int
        DPI.
    """
    data = extract_solution(sol)
    t = data['t']
    theta = data['theta']
    phi = data['phi']
    omega1 = data['omega1']
    omega3 = data['omega3']

    x_top = traj['x_top']
    y_top = traj['y_top']
    z_top = traj['z_top']

    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(221, projection='3d')
    ax_xy = fig.add_subplot(222)
    ax_angles = fig.add_subplot(223)
    ax_omega = fig.add_subplot(224)

    # 3D view init
    traj3d, = ax3d.plot([], [], [], 'b-', linewidth=0.8, alpha=0.5)
    axis3d, = ax3d.plot([], [], [], 'r-', linewidth=3)
    top3d, = ax3d.plot([], [], [], 'ro', markersize=8)
    ax3d.plot([0], [0], [0], 'ko', markersize=10)
    ax3d.set_xlim(-0.15, 0.15)
    ax3d.set_ylim(-0.15, 0.15)
    ax3d.set_zlim(-0.02, 0.15)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D View')

    # XY projection init
    traj_xy, = ax_xy.plot([], [], 'b-', linewidth=1, alpha=0.6)
    pos_xy, = ax_xy.plot([], [], 'ro', markersize=10)
    ax_xy.plot([0], [0], 'k^', markersize=12)
    ax_xy.set_xlim(-0.15, 0.15)
    ax_xy.set_ylim(-0.15, 0.15)
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Projection')
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)

    # Angles init
    theta_line, = ax_angles.plot([], [], 'b-', linewidth=2,
                                 label=r'$\theta$ (nutation)')
    phi_line_scaled, = ax_angles.plot([], [], 'r-', linewidth=2,
                                      label=r'$\phi$/10 (precession)')
    theta_marker, = ax_angles.plot([], [], 'bo', markersize=8)
    ax_angles.set_xlim(0, t[-1])
    ax_angles.set_ylim(0, 40)
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (deg)')
    ax_angles.set_title('Euler Angles')
    ax_angles.legend(loc='upper left')
    ax_angles.grid(True, alpha=0.3)

    # Angular velocity init
    w1_line, = ax_omega.plot([], [], 'b-', linewidth=2,
                             label=r'$\omega_1$ (spin)')
    w3_line_scaled, = ax_omega.plot([], [], 'g-', linewidth=2,
                                    label=r'$\omega_3 \times 20$ (precession)')
    w1_marker, = ax_omega.plot([], [], 'bo', markersize=8)
    ax_omega.set_xlim(0, t[-1])
    ax_omega.set_ylim(min(omega1) * 0.95, max(omega1) * 1.05)
    ax_omega.set_xlabel('Time (s)')
    ax_omega.set_ylabel('Angular Velocity (rad/s)')
    ax_omega.set_title('Angular Velocities')
    ax_omega.legend(loc='upper right')
    ax_omega.grid(True, alpha=0.3)

    info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=14,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def init():
        traj3d.set_data([], [])
        traj3d.set_3d_properties([])
        axis3d.set_data([], [])
        axis3d.set_3d_properties([])
        top3d.set_data([], [])
        top3d.set_3d_properties([])
        traj_xy.set_data([], [])
        pos_xy.set_data([], [])
        theta_line.set_data([], [])
        phi_line_scaled.set_data([], [])
        theta_marker.set_data([], [])
        w1_line.set_data([], [])
        w3_line_scaled.set_data([], [])
        w1_marker.set_data([], [])
        info_text.set_text('')
        return (traj3d, axis3d, top3d, traj_xy, pos_xy, theta_line,
                phi_line_scaled, theta_marker, w1_line, w3_line_scaled,
                w1_marker, info_text)

    def animate(i):
        traj3d.set_data(x_top[:i+1], y_top[:i+1])
        traj3d.set_3d_properties(z_top[:i+1])
        axis3d.set_data([0, x_top[i]], [0, y_top[i]])
        axis3d.set_3d_properties([0, z_top[i]])
        top3d.set_data([x_top[i]], [y_top[i]])
        top3d.set_3d_properties([z_top[i]])

        traj_xy.set_data(x_top[:i+1], y_top[:i+1])
        pos_xy.set_data([x_top[i]], [y_top[i]])

        theta_line.set_data(t[:i+1], np.degrees(theta[:i+1]))
        phi_line_scaled.set_data(t[:i+1], np.degrees(phi[:i+1]) / 10)
        theta_marker.set_data([t[i]], [np.degrees(theta[i])])

        w1_line.set_data(t[:i+1], omega1[:i+1])
        w3_line_scaled.set_data(t[:i+1], omega3[:i+1] * 20)
        w1_marker.set_data([t[i]], [omega1[i]])

        info_text.set_text(
            f'Time: {t[i]:.2f}s | '
            f'θ={np.degrees(theta[i]):.1f}° | '
            f'ω₁={omega1[i]:.1f} rad/s'
        )
        return (traj3d, axis3d, top3d, traj_xy, pos_xy, theta_line,
                phi_line_scaled, theta_marker, w1_line, w3_line_scaled,
                w1_marker, info_text)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t), interval=20, blit=True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'anim03_multi_view.gif')
    anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {filepath}")


def generate_all_animations(output_dir, omega1_0=150.0, omega3_0=5.0,
                            t_end=5.0, n_points=250, fps=25, dpi=80):
    """Generate all GIF animations.

    Animations use separate simulation parameters (higher spin speed,
    longer duration) for better visual effect.

    Parameters
    ----------
    output_dir : str
        Directory to save the GIFs.
    omega1_0 : float
        Initial spin angular velocity for animation (rad/s).
    omega3_0 : float
        Initial precession angular velocity for animation (rad/s).
    t_end : float
        Simulation end time for animations (s).
    n_points : int
        Number of evaluation points.
    fps : int
        Frames per second for output GIFs.
    dpi : int
        DPI for output GIFs.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_matplotlib()

    print("\nRunning animation simulation...")

    # Run animation-specific simulation
    anim_params = _get_animation_params(omega1_0, omega3_0, t_end, n_points)
    sol = simulate(**anim_params)

    # Compute trajectory with animation-specific height
    data = extract_solution(sol)
    traj = compute_trajectory(data['theta'], data['phi'], h=config.ANIM_GYRO_HEIGHT)

    print("Generating animations...")

    animate_3d_motion(sol, traj, output_dir, fps, dpi)
    animate_xy_trajectory(sol, traj, output_dir, fps, dpi)
    animate_multi_view(sol, traj, output_dir, fps, dpi)

    print(f"\n  All animations saved to '{output_dir}/'")
