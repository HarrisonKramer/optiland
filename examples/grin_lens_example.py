"""
Example: GRIN Lens Ray Tracing

This example demonstrates how to create and use a Gradient-Index (GRIN) lens
in Optiland. GRIN lenses have a refractive index that varies with position,
allowing for unique optical properties.

In this example, we create a simple GRIN rod lens with a radial gradient
and trace rays through it to observe the focusing/defocusing behavior.
"""

import matplotlib.pyplot as plt
import numpy as np

from optiland import optic
from optiland.materials import GradientMaterial
from optiland.geometries import Plane


def create_grin_lens_system():
    """Create an optical system with a GRIN lens.

    Returns:
        An Optiland OpticalSystem instance.

    """
    # Create a new optical system
    lens = optic.OpticalSystem()

    # Add object surface (point source at infinity)
    lens.add_surface(
        geometry=Plane(),
        material=None,  # Air before first surface
        is_stop=True
    )

    # Create a GRIN material with radial gradient
    # n(r) = n0 + nr2*r^2 + nr4*r^4
    # Negative nr2 creates a focusing lens (higher index at center)
    grin_material = GradientMaterial(
        n0=1.6,      # Base refractive index
        nr2=-0.05,   # Negative coefficient for focusing (n decreases with r²)
        nr4=0.0,     # No r^4 term
        nr6=0.0,     # No r^6 term
    )

    # Add GRIN lens entrance surface (plane at z=0)
    lens.add_surface(
        geometry=Plane(),
        material=grin_material,
        thickness=10.0  # 10 mm thick GRIN rod
    )

    # Add GRIN lens exit surface (plane at z=10)
    lens.add_surface(
        geometry=Plane(),
        material=None  # Air after lens
    )

    # Add image surface
    lens.add_surface(
        geometry=Plane()
    )

    return lens


def trace_rays_through_grin_lens():
    """Trace rays through the GRIN lens and visualize the results."""
    # Create the GRIN lens system
    system = create_grin_lens_system()

    # Define ray source parameters
    wavelength = 0.55  # 550 nm (green light)
    num_rays = 11       # Number of rays
    ray_height = 2.0    # Maximum ray height (mm)

    # Create a fan of rays starting at different heights
    y_positions = np.linspace(-ray_height, ray_height, num_rays)

    # Trace rays
    for y in y_positions:
        # Add ray starting at x=0, y=current height, z=-infinity
        # Direction: along z-axis (0, 0, 1)
        system.add_rays(
            x=[0.0],
            y=[y],
            z=[np.inf],  # Object at infinity
            L=[0.0],
            M=[0.0],
            N=[1.0],
            wavelength=[wavelength]
        )

    # Trace through system
    system.trace()

    # Extract ray paths for visualization
    # Plot the ray trace
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw lens boundaries
    lens_length = 10.0
    lens_radius = 3.0

    # Draw GRIN lens
    lens_rect = plt.Rectangle((0, -lens_radius), lens_length, 2*lens_radius,
                              fill=False, edgecolor='blue',
                              linewidth=2, label='GRIN Lens')
    ax.add_patch(lens_rect)

    # Plot ray paths
    for i, y_start in enumerate(y_positions):
        # Get ray data from the system
        # Each ray has x, y, z coordinates at each surface
        x_data = []
        y_data = []

        # Surface 0: Object (at infinity, show as z=-5)
        x_data.append(-5.0)
        y_data.append(y_start)

        # Surface 1: Entrance to GRIN lens
        ray_data_1 = system.surface_group.surfaces[1]
        if len(ray_data_1.x) > i:
            x_data.append(0.0)
            y_data.append(ray_data_1.y[i] if hasattr(ray_data_1, 'y') and len(ray_data_1.y) > i else y_start)

        # Surface 2: Exit from GRIN lens
        ray_data_2 = system.surface_group.surfaces[2]
        if len(ray_data_2.x) > i:
            x_data.append(lens_length)
            y_data.append(ray_data_2.y[i] if hasattr(ray_data_2, 'y') and len(ray_data_2.y) > i else y_start)

        # Surface 3: Image plane (approximately at focal point)
        ray_data_3 = system.surface_group.surfaces[3]
        if len(ray_data_3.x) > i:
            x_data.append(lens_length + 5.0)
            y_data.append(ray_data_3.y[i] if hasattr(ray_data_3, 'y') and len(ray_data_3.y) > i else y_start)

        # Plot this ray
        color = plt.cm.viridis(i / len(y_positions))
        ax.plot(x_data, y_data, 'o-', color=color, alpha=0.7, linewidth=1.5)

    # Formatting
    ax.set_xlabel('Z Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title('Ray Tracing through GRIN Lens\n(Radial Gradient: n(r) = 1.6 - 0.05*r²)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(-ray_height*1.5, ray_height*1.5)
    ax.set_xlim(-1, lens_length + 6)

    # Add refractive index gradient indicator
    textstr = 'GRIN Profile:\nn₀ = 1.6\nnr₂ = -0.05\n(negative = focusing)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('grin_lens_raytrace.png', dpi=150, bbox_inches='tight')
    print("Ray trace plot saved as 'grin_lens_raytrace.png'")
    plt.show()


def compare_grin_vs_uniform():
    """Compare GRIN lens with uniform refractive index lens."""
    print("\n" + "="*60)
    print("Comparison: GRIN Lens vs Uniform Index Lens")
    print("="*60)

    # Create two systems: one with GRIN, one with uniform index
    # GRIN lens
    grin_system = optic.OpticalSystem()
    grin_system.add_surface(geometry=Plane(), material=None, is_stop=True)

    grin_material = GradientMaterial(n0=1.6, nr2=-0.05)
    grin_system.add_surface(geometry=Plane(), material=grin_material, thickness=10.0)
    grin_system.add_surface(geometry=Plane(), material=None)
    grin_system.add_surface(geometry=Plane())

    # Uniform index lens (same base index, no gradient)
    uniform_system = optic.OpticalSystem()
    uniform_system.add_surface(geometry=Plane(), material=None, is_stop=True)

    from optiland.materials import IdealMaterial
    uniform_material = IdealMaterial(1.6)
    uniform_system.add_surface(geometry=Plane(), material=uniform_material, thickness=10.0)
    uniform_system.add_surface(geometry=Plane(), material=None)
    uniform_system.add_surface(geometry=Plane())

    # Trace a marginal ray through both systems
    y_start = 2.0  # 2 mm off-axis

    for system, name in [(grin_system, "GRIN Lens"), (uniform_system, "Uniform Lens")]:
        system.add_rays(
            x=[0.0], y=[y_start], z=[np.inf],
            L=[0.0], M=[0.0], N=[1.0], wavelength=[0.55]
        )
        system.trace()

        # Get final ray position
        final_surface = system.surface_group.surfaces[3]
        if hasattr(final_surface, 'y') and len(final_surface.y) > 0:
            y_final = final_surface.y[0]
            print(f"\n{name}:")
            print(f"  Initial ray height: {y_start:.2f} mm")
            print(f"  Final ray height: {y_final:.4f} mm")
            print(f"  Ray deflection: {abs(y_final - y_start):.4f} mm")


if __name__ == "__main__":
    print("GRIN Lens Example")
    print("="*60)

    # Run the ray tracing visualization
    trace_rays_through_grin_lens()

    # Compare with uniform index
    compare_grin_vs_uniform()

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
