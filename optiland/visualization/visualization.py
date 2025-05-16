"""Visualization Module

This module provides visualization tools for optical systems using VTK and
Matplotlib. It includes the `OpticViewer` class, which allows for the
visualization of lenses, rays, and their interactions within an optical system.
The module supports plotting rays with different distributions, wavelengths,
and through various fields of view.

Kramer Harrison, 2024
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import vtk

import optiland.backend as be
from optiland import materials
from optiland.physical_apertures import RadialAperture
from optiland.visualization.rays import Rays2D, Rays3D
from optiland.visualization.system import OpticalSystem


class OpticViewer:
    """A class used to visualize optical systems.

    Args:
        optic: The optical system to be visualized.

    Attributes:
        optic: The optical system to be visualized.
        rays: An instance of Rays2D for ray tracing.
        system: An instance of OpticalSystem for system representation.

    Methods:
        view(fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4), xlim=None, ylim=None):
            Visualizes the optical system with specified parameters.

    """

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays2D(optic)
        self.system = OpticalSystem(optic, self.rays, projection="2d")

    def view(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        figsize=(10, 4),
        xlim=None,
        ylim=None,
        title=None,
        reference=None,
    ):
        """Visualizes the optical system.

        Args:
            fields (str, optional): The fields to be visualized.
                Defaults to 'all'.
            wavelengths (str, optional): The wavelengths to be visualized.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be visualized.
                Defaults to 3.
            distribution (str, optional): The distribution of rays.
                Defaults to 'line_y'.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 4).
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        _, ax = plt.subplots(figsize=figsize)

        self.rays.plot(
            ax,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
        )
        self.system.plot(ax)

        plt.gca().set_facecolor("#f8f9fa")  # off-white background
        plt.axis("image")

        ax.set_xlabel("Z [mm]")
        ax.set_ylabel("Y [mm]")

        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.grid(alpha=0.25)

        plt.show()


class OpticViewer3D:
    """A class used to visualize optical systems in 3D.

    Args:
        optic: The optical system to be visualized.

    Attributes:
        optic: The optical system to be visualized.
        rays: An instance of Rays3D for ray tracing.
        system: An instance of OpticalSystem for system representation.
        ren_win: The vtkRenderWindow object for visualization.
        iren: The vtkRenderWindowInteractor object for interaction.

    Methods:
        view(fields='all', wavelengths='primary', num_rays=24,
             distribution='ring', figsize=(1200, 800), dark_mode=False):
            Visualizes the optical system in 3D.

    """

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays3D(optic)
        self.system = OpticalSystem(optic, self.rays, projection="3d")

        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    def view(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=24,
        distribution="ring",
        figsize=(1200, 800),
        dark_mode=False,
        reference=None,
    ):
        """Visualizes the optical system in 3D.

        Args:
            fields (str, optional): The fields to be visualized.
                Defaults to 'all'.
            wavelengths (str, optional): The wavelengths to be visualized.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be visualized.
                Defaults to 24.
            distribution (str, optional): The distribution of rays.
                Defaults to 'ring'.
            figsize (tuple, optional): The size of the figure.
                Defaults to (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode.
                Defaults to False.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        renderer = vtk.vtkRenderer()
        self.ren_win.AddRenderer(renderer)

        self.iren.SetRenderWindow(self.ren_win)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.rays.plot(
            renderer,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
        )
        self.system.plot(renderer)

        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)

        if dark_mode:
            renderer.SetBackground(0.13, 0.15, 0.19)
            renderer.SetBackground2(0.195, 0.21, 0.24)
        else:
            renderer.SetBackground(0.8, 0.9, 1.0)
            renderer.SetBackground2(0.4, 0.5, 0.6)

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName("Optical System - 3D Viewer")
        self.ren_win.Render()

        renderer.GetActiveCamera().SetPosition(1, 0, 0)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Elevation(0)
        renderer.GetActiveCamera().Azimuth(150)

        self.ren_win.Render()
        self.iren.Start()


class LensInfoViewer:
    """A class for viewing information about a lens.

    Args:
        optic (Optic): The optic object containing the lens information.

    Attributes:
        optic (Optic): The optic object containing the lens information.

    Methods:
        view(): Prints the lens information in a tabular format.

    """

    def __init__(self, optic):
        self.optic = optic

    def view(self):
        """Prints the lens information in a tabular format.

        The lens information includes the surface type, radius, thickness,
        material, conic, and semi-aperture of each surface.
        """
        self.optic.update_paraxial()

        surf_type = self._get_surface_types()
        comments = self._get_comments()
        radii = be.to_numpy(self.optic.surface_group.radii)
        thicknesses = self._get_thicknesses()
        conic = be.to_numpy(self.optic.surface_group.conic)
        semi_aperture = self._get_semi_apertures()
        mat = self._get_materials()

        self.optic.update_paraxial()

        df = pd.DataFrame(
            {
                "Type": surf_type,
                "Comment": comments,
                "Radius": radii,
                "Thickness": thicknesses,
                "Material": mat,
                "Conic": conic,
                "Semi-aperture": semi_aperture,
            },
        )
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def _get_surface_types(self):
        """Extracts and formats the surface types."""
        surf_type = []
        for surf in self.optic.surface_group.surfaces:
            g = surf.geometry

            # Check if __str__ method exists
            if type(g).__dict__.get("__str__"):
                surf_type.append(str(surf.geometry))
            else:
                raise ValueError("Unknown surface type")

            if surf.is_stop:
                surf_type[-1] = "Stop - " + surf_type[-1]
        return surf_type

    def _get_comments(self):
        """Extracts comments for each surface."""
        return [surf.comment for surf in self.optic.surface_group.surfaces]

    def _get_thicknesses(self):
        """Calculates thicknesses between surfaces."""
        thicknesses = be.diff(
            be.ravel(self.optic.surface_group.positions), append=be.array([be.nan])
        )
        return be.to_numpy(thicknesses)

    def _get_semi_apertures(self):
        """Extracts semi-aperture values for each surface."""
        semi_apertures = []
        for surf in self.optic.surface_group.surfaces:
            if isinstance(surf.aperture, RadialAperture):
                semi_apertures.append(be.to_numpy(surf.aperture.r_max))
            else:
                semi_apertures.append(be.to_numpy(surf.semi_aperture))
        return semi_apertures

    def _get_materials(self):
        """Determines the material for each surface."""
        mat = []
        for surf in self.optic.surface_group.surfaces:
            if surf.is_reflective:
                mat.append("Mirror")
            elif isinstance(surf.material_post, materials.Material):
                mat.append(surf.material_post.name)
            elif isinstance(surf.material_post, materials.MaterialFile):
                mat.append(os.path.basename(surf.material_post.filename))
            elif surf.material_post.index == 1:
                mat.append("Air")
            elif isinstance(surf.material_post, materials.IdealMaterial):
                mat.append(surf.material_post.index.item())
            elif isinstance(surf.material_post, materials.AbbeMaterial):
                mat.append(
                    f"{surf.material_post.index.item():.4f}, "
                    f"{surf.material_post.abbe.item():.2f}",
                )
            else:
                raise ValueError("Unknown material type")
        return mat
