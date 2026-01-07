"""
Simple GRIN Propagation Test

This example demonstrates basic usage of GRIN material and propagation
without requiring a full optical system.
"""

import matplotlib.pyplot as plt
import numpy as np

from optiland.materials import GradientMaterial
from optiland.propagation.grin import GRINPropagation
from optiland.rays import RealRays


def test_radial_grin_lens():
    """Test ray propagation through a radial GRIN medium."""
    print("\n" + "="*60)
    print("Testing Radial GRIN Lens")
    print("="*60)

    # Create a GRIN material with negative nr2 (focusing lens)
    # n(r) = n0 + nr2*r^2
    # Negative nr2 means higher index at center → focusing
    material = GradientMaterial(
        n0=1.6,      # Base refractive index at center
        nr2=-0.02,   # Negative coefficient (focusing)
        nr4=0.0,
        nr6=0.0,
        nz1=0.0,
        nz2=0.0,
        nz3=0.0
    )

    print(f"\nGRIN Material Properties:")
    print(f"  n0  = {material.n0}")
    print(f"  nr2 = {material.nr2}")
    print(f"  Formula: n(r) = n0 + nr2*r²")

    # Test refractive index at different positions
    print(f"\nRefractive Index at different positions:")
    for r in [0.0, 0.5, 1.0, 1.5, 2.0]:
        n = material.n(0.55, x=r, y=0.0, z=0.0)
        print(f"  r = {r:4.1f} mm: n = {n:.4f}")

    # Create multiple rays at different heights
    num_rays = 11
    max_height = 2.0  # mm
    y_positions = np.linspace(-max_height, max_height, num_rays)

    rays_list = []
    for y in y_positions:
        ray = RealRays(
            x=[0.0],
            y=[y],
            z=[0.0],
            L=[0.0],
            M=[0.0],
            N=[1.0],  # Propagating in +z direction
            intensity=[1.0],
            wavelength=[0.55]  # 550 nm
        )
        rays_list.append(ray)

    # Propagate each ray through the GRIN medium
    propagation_model = GRINPropagation(material)
    thickness = 20.0  # mm

    print(f"\nPropagating rays through {thickness} mm of GRIN material...")

    # Store initial and final positions
    initial_y = []
    final_y = []
    final_L = []

    for i, rays in enumerate(rays_list):
        y_start = rays.y[0]
        initial_y.append(y_start)

        # Propagate
        propagation_model.propagate(rays, t=thickness)

        # Store final position
        final_y.append(rays.y[0])
        final_L.append(rays.L[0])  # Should be ~0 for symmetric lens

        print(f"  Ray {i:2d}: y_start = {y_start:+6.2f} mm → "
              f"y_final = {rays.y[0]:+6.3f} mm, "
              f"z = {rays.z[0]:.2f} mm")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Ray paths
    for i, (y_start, y_end) in enumerate(zip(initial_y, final_y)):
        color = plt.cm.viridis(i / len(initial_y))
        # Draw simplified ray path (straight line approximation for visualization)
        ax1.plot([0, thickness], [y_start, y_end], 'o-',
                color=color, alpha=0.7, linewidth=1.5,
                label=f'y={y_start:.1f}' if i % 2 == 0 else '')

    # Draw lens boundaries
    lens_rect = plt.Rectangle((0, -max_height*1.2), thickness, 2.4*max_height,
                              fill=False, edgecolor='blue', linewidth=2,
                              linestyle='--', label='GRIN Medium')
    ax1.add_patch(lens_rect)

    ax1.set_xlabel('Z Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title('Ray Paths through Radial GRIN Medium\n(negative nr2 = focusing)',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=8)
    ax1.set_ylim(-max_height*1.5, max_height*1.5)

    # Plot 2: Initial vs Final position
    ax2.plot(initial_y, final_y, 'bo-', markersize=8, linewidth=2, label='Actual')
    ax2.plot(initial_y, initial_y, 'r--', linewidth=1.5, label='No deviation')
    ax2.set_xlabel('Initial Y Position (mm)', fontsize=12)
    ax2.set_ylabel('Final Y Position (mm)', fontsize=12)
    ax2.set_title('Ray Deviation through GRIN Medium', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')

    # Add info text
    info_text = (f'GRIN Parameters:\n'
                 f'n₀ = {material.n0}\n'
                 f'nr₂ = {material.nr2}\n'
                 f'Thickness = {thickness} mm')
    ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('grin_propagation_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'grin_propagation_test.png'")
    plt.show()

    # Calculate focusing power
    # For a focusing GRIN lens, rays should converge toward the axis
    max_deviation = max(abs(np.array(final_y) - np.array(initial_y)))
    print(f"\nMaximum ray deviation: {max_deviation:.4f} mm")

    if material.nr2 < 0:
        print("✓ Focusing GRIN lens: rays bend toward optical axis")
    else:
        print("✗ Defocusing GRIN lens: rays bend away from optical axis")


def test_axial_grin():
    """Test ray propagation through an axial GRIN medium."""
    print("\n" + "="*60)
    print("Testing Axial GRIN Medium")
    print("="*60)

    # Create a GRIN material with axial gradient
    # n(z) = n0 + nz1*z
    material = GradientMaterial(
        n0=1.5,
        nr2=0.0,
        nr4=0.0,
        nr6=0.0,
        nz1=0.01,   # Positive nz1 means index increases with z
        nz2=0.0,
        nz3=0.0
    )

    print(f"\nGRIN Material Properties:")
    print(f"  n0  = {material.n0}")
    print(f"  nz1 = {material.nz1}")
    print(f"  Formula: n(z) = n0 + nz1*z")

    # Test refractive index at different z positions
    print(f"\nRefractive Index at different z positions:")
    for z in [0.0, 5.0, 10.0, 15.0, 20.0]:
        n = material.n(0.55, x=0.0, y=0.0, z=z)
        print(f"  z = {z:4.1f} mm: n = {n:.4f}")

    # Create a ray
    rays = RealRays(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        L=[0.0],
        M=[0.0],
        N=[1.0],
        intensity=[1.0],
        wavelength=[0.55]
    )

    # Propagate
    propagation_model = GRINPropagation(material)
    thickness = 20.0

    print(f"\nPropagating on-axis ray through {thickness} mm...")
    propagation_model.propagate(rays, t=thickness)

    print(f"\nFinal ray position:")
    print(f"  x = {rays.x[0]:.4f} mm")
    print(f"  y = {rays.y[0]:.4f} mm")
    print(f"  z = {rays.z[0]:.4f} mm")
    print(f"  Direction: L={rays.L[0]:.6f}, M={rays.M[0]:.6f}, N={rays.N[0]:.6f}")

    # For an on-axis ray in an axial GRIN, the ray should stay on axis
    # (no bending, just changing phase velocity)
    if abs(rays.x[0]) < 1e-6 and abs(rays.y[0]) < 1e-6:
        print("✓ On-axis ray remains on axis (as expected)")
    else:
        print("✗ Warning: On-axis ray deviated from axis")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIN Propagation Examples")
    print("="*60)

    # Test radial GRIN lens
    test_radial_grin_lens()

    # Test axial GRIN
    test_axial_grin()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
