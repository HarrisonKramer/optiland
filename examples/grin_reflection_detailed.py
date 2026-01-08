"""
Detailed test for GRIN reflection effect.

This script traces rays with very small steps to visualize
the curved path in a GRIN medium with reflection.
"""

import matplotlib.pyplot as plt
import numpy as np

from optiland.materials import GradientMaterial
from optiland.propagation.grin import GRINPropagation
from optiland.rays import RealRays


def test_grin_reflection_detailed():
    """Test GRIN reflection with detailed path visualization."""
    print("\n" + "="*70)
    print("Detailed GRIN Reflection Test")
    print("="*70)

    # Create material with strong negative gradient
    # Use extreme parameters to ensure reflection occurs
    n0 = 2.0
    nz1 = -0.3  # Very strong negative gradient: n(z) = 2.0 - 0.3*z
    thickness = 10.0

    material = GradientMaterial(n0=n0, nz1=nz1)

    print(f"\nMaterial: n(z) = {n0} + {nz1}*z")
    print(f"Thickness: {thickness} mm")
    print(f"\nRefractive index profile:")
    for z in [0, 2.5, 5.0, 7.5, 10.0]:
        n = material.n(0.55, x=0.0, y=0.0, z=z)
        print(f"  z = {z:4.1f} mm: n = {n:.3f}")

    # Test single ray with detailed path recording
    angle = 0.3  # ~17.2 degrees
    L_init = angle
    N_init = np.sqrt(1 - angle**2)

    rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[L_init], M=[0.0], N=[N_init],
        intensity=[1.0], wavelength=[0.55]
    )

    # Record detailed path
    z_points = [0.0]
    x_points = [0.0]
    L_points = [L_init]
    N_points = [N_init]
    n_points = [material.n(0.55, x=0.0, y=0.0, z=0.0)]

    # Temporarily reduce step size for detailed tracing
    prop_model = GRINPropagation(material)
    original_step_size = prop_model.step_size
    prop_model.step_size = 0.001  # Very small step for detailed path

    print(f"\nTracing ray at {np.arcsin(angle)*180/np.pi:.1f}° incidence...")
    print(f"Initial: L={L_init:.4f}, N={N_init:.4f}")

    # Manually integrate step by step for visualization
    # We'll directly use the RK4 method to trace intermediate points
    max_steps = 100000
    dt = 0.001  # Very small integration step

    print(f"Integrating with step size dt = {dt} mm...")

    inside = False  # Track if ray has entered the medium

    for i in range(max_steps):
        # Get current state
        x_curr = float(rays.x[0])
        y_curr = float(rays.y[0])
        z_curr = float(rays.z[0])
        L_curr = float(rays.L[0])
        M_curr = float(rays.M[0])
        N_curr = float(rays.N[0])
        w_curr = float(rays.w[0])

        # Record current point
        z_points.append(z_curr)
        x_points.append(rays.x[0])
        L_points.append(rays.L[0])
        N_points.append(rays.N[0])
        n_points.append(material.n(0.55, x=0.0, y=0.0, z=z_curr))

        # Check if ray has entered the medium
        if z_curr > 0.001:
            inside = True

        # Check if ray has exited (after entering)
        if inside and (z_curr <= 0.0 or z_curr >= thickness):
            print(f"\nRay exited at step {i+1}")
            break

        # Manually perform one RK4 step
        # K1
        dx_ds, dy_ds, dz_ds, dL_ds, dM_ds, dN_ds = prop_model._ray_derivative(
            np.array([x_curr]), np.array([y_curr]), np.array([z_curr]),
            np.array([L_curr]), np.array([M_curr]), np.array([N_curr]),
            np.array([w_curr])
        )

        # Update state
        rays.x[0] = x_curr + dt * dx_ds[0]
        rays.y[0] = y_curr + dt * dy_ds[0]
        rays.z[0] = z_curr + dt * dz_ds[0]
        rays.L[0] = L_curr + dt * dL_ds[0]
        rays.M[0] = M_curr + dt * dM_ds[0]
        rays.N[0] = N_curr + dt * dN_ds[0]

        # Normalize direction
        norm = np.sqrt(rays.L[0]**2 + rays.M[0]**2 + rays.N[0]**2)
        if norm > 0:
            rays.L[0] /= norm
            rays.M[0] /= norm
            rays.N[0] /= norm

    print(f"Final: L={rays.L[0]:.4f}, N={rays.N[0]:.4f}")
    print(f"Final z position: {rays.z[0]:.4f} mm")

    # Find turning point (where N changes sign)
    n_array = np.array(n_points)
    N_array = np.array(N_points)
    turning_idx = np.argmin(np.abs(N_array))
    turning_z = z_points[turning_idx]
    turning_x = x_points[turning_idx]
    turning_n = n_array[turning_idx]

    print(f"\nTurning point:")
    print(f"  z = {turning_z:.4f} mm")
    print(f"  x = {turning_x:.4f} mm")
    print(f"  n = {turning_n:.4f}")
    print(f"  N = {N_array[turning_idx]:.6f} (close to zero)")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Ray path (side view)
    ax1 = axes[0, 0]
    ax1.plot(z_points, x_points, 'r-', linewidth=2, label='Ray path')
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Entrance')
    ax1.axvline(x=thickness, color='b', linestyle='--', alpha=0.3, label='Exit')
    ax1.scatter([turning_z], [turning_x], color='green', s=100,
                zorder=5, label=f'Turning point\n(z={turning_z:.2f}, n={turning_n:.2f})')
    ax1.set_xlabel('Z Position (mm)', fontsize=11)
    ax1.set_ylabel('X Position (mm)', fontsize=11)
    ax1.set_title('Ray Path in GRIN Medium (Side View)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Direction cosines vs z
    ax2 = axes[0, 1]
    ax2.plot(z_points, L_points, 'r-', linewidth=2, label='L (x-direction)')
    ax2.plot(z_points, N_points, 'b-', linewidth=2, label='N (z-direction)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=turning_z, color='green', linestyle='--', alpha=0.5, label='Turning point')
    ax2.set_xlabel('Z Position (mm)', fontsize=11)
    ax2.set_ylabel('Direction Cosine', fontsize=11)
    ax2.set_title('Direction Cosines vs Position', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Refractive index along path
    ax3 = axes[1, 0]
    ax3.plot(z_points, n_points, 'b-', linewidth=2, label='n(z)')
    ax3.axvline(x=turning_z, color='green', linestyle='--', alpha=0.5, label='Turning point')
    ax3.axhline(y=turning_n, color='green', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Z Position (mm)', fontsize=11)
    ax3.set_ylabel('Refractive Index', fontsize=11)
    ax3.set_title('Refractive Index Profile', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Ray parameter evolution
    ax4 = axes[1, 1]
    distance = np.cumsum(np.sqrt(np.diff(np.array(z_points))**2 +
                                np.diff(np.array(x_points))**2))
    distance = np.insert(distance, 0, 0)
    ax4.plot(distance, np.array(L_points), 'r-', linewidth=2, label='L')
    ax4.plot(distance, np.array(N_points), 'b-', linewidth=2, label='N')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=distance[turning_idx], color='green', linestyle='--',
                alpha=0.5, label='Turning point')
    ax4.set_xlabel('Path Length (mm)', fontsize=11)
    ax4.set_ylabel('Direction Cosine', fontsize=11)
    ax4.set_title('Direction vs Path Length', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('grin_reflection_detailed.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'grin_reflection_detailed.png'")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    test_grin_reflection_detailed()
