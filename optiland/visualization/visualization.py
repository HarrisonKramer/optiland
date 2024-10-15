"""Optiland Visualization Module

This module provides visualization tools for optical systems using VTK and
Matplotlib. It includes the `OpticViewer` class, which allows for the
visualization of lenses, rays, and their interactions within an optical system.
The module supports plotting rays with different distributions, wavelengths,
and through various fields of view. It also visualizes the surfaces of the
optical elements, providing insights into the design and performance of the
system.

Kramer Harrison, 2024
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from optiland import geometries, materials
from optiland.visualization.rays import Rays2D, Rays3D
from optiland.visualization.system import OpticalSystem


class OpticViewer:
    """
    A class used to visualize optical systems.

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
        self.system = OpticalSystem(optic, self.rays, projection='2d')

    def view(self, fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4), xlim=None, ylim=None):
        """
        Visualizes the optical system.

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
        """
        _, ax = plt.subplots(figsize=figsize)

        self.rays.plot(ax, fields=fields, wavelengths=wavelengths,
                       num_rays=num_rays, distribution=distribution)
        self.system.plot(ax)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.gca().set_facecolor('#f8f9fa')  # off-white background
        plt.axis('image')
        plt.show()


class OpticViewer3D:

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays3D(optic)
        self.system = OpticalSystem(optic, self.rays, projection='3d')

        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    def view(self, fields='all', wavelengths='primary', num_rays=2,
             distribution='hexapolar', figsize=(1200, 800)):
        renderer = vtk.vtkRenderer()
        self.ren_win.AddRenderer(renderer)

        self.iren.SetRenderWindow(self.ren_win)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.rays.plot(renderer, fields=fields, wavelengths=wavelengths,
                       num_rays=num_rays, distribution=distribution)
        self.system.plot(renderer)

        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(
            vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL
        )

        renderer.SetBackground(0.8, 0.9, 1.0)
        renderer.SetBackground2(0.4, 0.5, 0.6)

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName('Optical System - 3D Viewer')
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
    """
    A class for viewing information about a lens.

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
        """
        Prints the lens information in a tabular format.

        The lens information includes the surface type, radius, thickness,
        material, conic, and semi-aperture of each surface.
        """
        self.optic.update_paraxial()

        surf_type = []
        for surf in self.optic.surface_group.surfaces:
            if isinstance(surf.geometry, geometries.EvenAsphere):
                surf_type.append('Even Asphere')
            elif isinstance(surf.geometry, geometries.Plane):
                surf_type.append('Planar')
            elif isinstance(surf.geometry, geometries.StandardGeometry):
                surf_type.append('Standard')
            else:
                raise ValueError('Unknown surface type')

            if surf.is_stop:
                surf_type[-1] = 'Stop - ' + surf_type[-1]

        radii = self.optic.surface_group.radii
        thicknesses = np.diff(self.optic.surface_group.positions.ravel(),
                              append=np.nan)
        conic = self.optic.surface_group.conic
        semi_aperture = [surf.semi_aperture
                         for surf in self.optic.surface_group.surfaces]

        mat = []
        for surf in self.optic.surface_group.surfaces:
            if surf.is_reflective:
                mat.append('Mirror')
            elif isinstance(surf.material_post, materials.Material):
                mat.append(surf.material_post.name)
            elif isinstance(surf.material_post, materials.MaterialFile):
                mat.append(os.path.basename(surf.material_post.filename))
            elif surf.material_post.index == 1:
                mat.append('Air')
            elif isinstance(surf.material_post, materials.IdealMaterial):
                mat.append(surf.material_post.index)
            else:
                raise ValueError('Unknown material type')

        self.optic.update_paraxial()

        df = pd.DataFrame({
            'Type': surf_type,
            'Radius': radii,
            'Thickness': thicknesses,
            'Material': mat,
            'Conic': conic,
            'Semi-aperture': semi_aperture
        })
        print(df.to_markdown(headers='keys', tablefmt='psql'))
