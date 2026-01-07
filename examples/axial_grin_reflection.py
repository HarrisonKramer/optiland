"""
Example: Axial GRIN Reflection Effect

This example demonstrates how an axial gradient in refractive index
can create a gradual "reflection" effect, where obliquely incident rays
bend toward the normal as they propagate through the medium.

Physical principle:
When n increases with z, rays propagating at an angle gradually bend
toward the z-axis (the normal), similar to how light bends when entering
a higher-index medium, but continuously throughout the GRIN medium.
"""

import matplotlib.pyplot as plt
import numpy as np

from optiland.materials import GradientMaterial
from optiland.propagation.grin import GRINPropagation
from optiland.rays import RealRays


def visualize_axial_gradient_reflection():
    """Visualize the reflection effect of axial GRIN medium."""
    print("\n" + "="*70)
    print("Axial GRIN Medium - Gradual Reflection Effect")
    print("="*70)

    # Create a material with axial gradient
    # n(z) = n0 + nz1*z
    # nz1 > 0 means refractive index increases with z
    n0 = 1.5
    nz1 = 0.05  # n increases from 1.5 at z=0 to 2.5 at z=20

    material = GradientMaterial(n0=n0, nz1=nz1)

    print(f"\nMaterial properties:")
    print(f"  n0  = {n0}")
    print(f"  nz1 = {nz1}")
    print(f"  Formula: n(z) = {n0} + {nz1}*z")

    # Calculate refractive index at different z positions
    z_positions = [0, 5, 10, 15, 20]
    print(f"\nRefractive index distribution:")
    for z in z_positions:
        n = material.n(0.55, x=0.0, y=0.0, z=z)
        print(f"  z = {z:2d} mm: n = {n:.3f}")

    # Test rays at different incident angles
    angles = [0.1, 0.2, 0.3, 0.4, 0.5]  # sin(angle)
    thickness = 20.0

    ray_data = []

    for angle in angles:
        # Create ray with oblique angle
        L_init = angle
        N_init = np.sqrt(1 - angle**2)

        rays = RealRays(
            x=[0.0], y=[0.0], z=[0.0],
            L=[L_init], M=[0.0], N=[N_init],
            intensity=[1.0], wavelength=[0.55]
        )

        L_initial = rays.L[0]
        N_initial = rays.N[0]

        # Propagate
        prop_model = GRINPropagation(material)
        prop_model.propagate(rays, t=thickness)

        # Calculate results
        L_final = rays.L[0]
        N_final = rays.N[0]
        bending = L_initial - L_final
        angle_deg = np.arcsin(angle) * 180 / np.pi

        ray_data.append({
            'angle_deg': angle_deg,
            'L_init': L_initial,
            'L_final': L_final,
            'bending': bending,
            'z_final': rays.z[0]
        })

        print(f"\nRay at {angle_deg:.1f}° incidence:")
        print(f"  Initial direction: L={L_init:.3f}, N={N_initial:.3f}")
        print(f"  Final direction:   L={L_final:.3f}, N={N_final:.3f}")
        print(f"  Bending amount:    {bending:.4f}")
        print(f"  Final z position:  {rays.z[0]:.2f} mm")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Ray paths
    ax1 = axes[0]

    # Trace rays for visualization
    for angle in angles:
        L_init = angle
        N_init = np.sqrt(1 - angle**2)

        rays = RealRays(
            x=[0.0], y=[0.0], z=[0.0],
            L=[L_init], M=[0.0], N=[N_init],
            intensity=[1.0], wavelength=[0.55]
        )

        # Store initial position
        z_points = [0]
        x_points = [0]

        # Propagate step by step for visualization
        prop_model = GRINPropagation(material)

        # Create multiple steps to visualize the curved path
        num_steps = 10
        step_thickness = thickness / num_steps

        for _ in range(num_steps):
            z_before = rays.z[0]
            prop_model.propagate(rays, t=step_thickness)
            z_after = rays.z[0]
            z_points.append(z_after)
            x_points.append(rays.x[0])

        # Plot this ray
        angle_deg = np.arcsin(angle) * 180 / np.pi
        color = plt.cm.plasma(angle / max(angles))
        ax1.plot(z_points, x_points, 'o-', color=color,
                linewidth=2, markersize=4, label=f'{angle_deg:.1f}°')

    # Draw refractive index gradient indicator
    ax1.axvspan(0, thickness, alpha=0.1, color='blue',
                label='GRIN Medium')
    ax1_twin = ax1.twinx()

    # Plot refractive index profile
    z_vals = np.linspace(0, thickness, 100)
    n_vals = [material.n(0.55, x=0.0, y=0.0, z=z) for z in z_vals]
    ax1_twin.plot(z_vals, n_vals, 'b--', linewidth=2, alpha=0.5)
    ax1_twin.set_ylabel('Refractive Index', color='blue', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    ax1_twin.set_ylim(n0 - 0.1, n0 + nz1 * thickness + 0.1)

    ax1.set_xlabel('Z Position (mm)', fontsize=12)
    ax1.set_ylabel('X Position (mm)', fontsize=12)
    ax1.set_title('Ray Paths in Axial GRIN Medium\n(Bending toward higher index)',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-1, max(angles) * thickness * 0.6 + 1)

    # Plot 2: Bending effect vs incident angle
    ax2 = axes[1]

    incident_angles = [d['angle_deg'] for d in ray_data]
    bending_amounts = [d['bending'] for d in ray_data]

    ax2.plot(incident_angles, bending_amounts, 'ro-',
             linewidth=2, markersize=8, label='Actual bending')
    ax2.set_xlabel('Incident Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Bending Amount (ΔL)', fontsize=12)
    ax2.set_title('Ray Bending vs Incident Angle', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add reference line
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='No bending')

    # Add info text
    info_text = (f'GRIN Parameters:\n'
                 f'n₀ = {n0}\n'
                 f'nz₁ = {nz1}\n'
                 f'Thickness = {thickness} mm\n\n'
                 f'Higher angle →\n'
                 f'More bending')
    ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('axial_grin_reflection.png', dpi=150, bbox_inches='tight')
    print(f"\n" + "="*70)
    print(f"Plot saved as 'axial_grin_reflection.png'")
    print(f"="*70)

    plt.show()


def compare_uniform_vs_gradient():
    """Compare uniform medium with axial GRIN medium."""
    print("\n" + "="*70)
    print("Comparison: Uniform vs Axial GRIN Medium")
    print("="*70)

    thickness = 20.0
    initial_angle = 0.3  # ~17.2°
    L_init = initial_angle
    N_init = np.sqrt(1 - initial_angle**2)

    # Uniform medium
    from optiland.materials import IdealMaterial
    uniform_material = IdealMaterial(n=1.5, k=0.0)

    rays_uniform = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[L_init], M=[0.0], N=[N_init],
        intensity=[1.0], wavelength=[0.55]
    )

    from optiland.propagation.homogeneous import HomogeneousPropagation
    prop_uniform = HomogeneousPropagation(uniform_material)
    prop_uniform.propagate(rays_uniform, t=thickness)

    print(f"\nUniform medium (n=1.5):")
    print(f"  Initial: L={L_init:.4f}, N={N_init:.4f}")
    print(f"  Final:   L={rays_uniform.L[0]:.4f}, N={rays_uniform.N[0]:.4f}")
    print(f"  Direction unchanged (straight line)")

    # Axial GRIN medium
    grin_material = GradientMaterial(n0=1.5, nz1=0.05)

    rays_grin = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[L_init], M=[0.0], N=[N_init],
        intensity=[1.0], wavelength=[0.55]
    )

    prop_grin = GRINPropagation(grin_material)
    prop_grin.propagate(rays_grin, t=thickness)

    print(f"\nAxial GRIN medium (n(z)=1.5+0.05*z):")
    print(f"  Initial: L={L_init:.4f}, N={N_init:.4f}")
    print(f"  Final:   L={rays_grin.L[0]:.4f}, N={rays_grin.N[0]:.4f}")

    bending = L_init - rays_grin.L[0]
    percent_bending = (bending / L_init) * 100

    print(f"  Bending: {bending:.4f} ({percent_bending:.1f}%)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Visualize axial gradient reflection effect
    visualize_axial_gradient_reflection()

    # Compare with uniform medium
    compare_uniform_vs_gradient()

    print("\nAll demonstrations completed!")
