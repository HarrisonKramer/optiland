"""Visualization Module

This module provides visualization tools for optical systems using VTK and
Matplotlib. It includes the `OpticViewer` class, which allows for the
visualization of lenses, rays, and their interactions within an optical system.
The module supports plotting rays with different distributions, wavelengths,
and through various fields of view.

Kramer Harrison, 2024
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from optiland import materials
from optiland.visualization.rays import Rays2D, Rays3D
from optiland.visualization.system import OpticalSystem

plt.rcParams.update({'font.size': 12, 'font.family': 'cambria'})


class SurfaceViewer:
    """
    A class used to visualize surfaces.

    Args:
        optic: The optical system to be visualized.
        surface_index: Index of the surface to be visualized.
    """

    def __init__(self, optic):
        self.optic = optic

    def view(self,
             surface_index: int,
             projection='2d',
             num_points=256,
             figsize=(7, 5.5),
             title: str = None):
        """
        Visualize the surface.

        Args:
            surface_index (int): Index of the surface to be visualized.
            projection (str): The type of projection to use for visualization.
                Can be '2d' or '3d'.
            num_points (int): The number of points to sample along each axis
                for the visualization.
            figsize (tuple): The size of the figure in inches.
                Defaults to (7, 5.5).
            title (str): Title.

        Raises:
            ValueError: If the projection is not '2d' or '3d'.
        """
        surface = self.optic.surface_group.surfaces[surface_index]
        x, y = np.meshgrid(np.linspace(-1, 1, num_points),
                           np.linspace(-1, 1, num_points))
        z = surface.geometry.sag(x, y)
        z[np.sqrt(x**2+y**2) > 1] = np.nan

        if projection == '2d':
            self._plot_2d(z, figsize=figsize, title=title, 
                          surface_type=surface.surface_type,
                          surface_index=surface_index)
        elif projection == '3d':
            self._plot_3d(x, y, z, figsize=figsize, title=title,
                          surface_type=surface.surface_type,
                          surface_index=surface_index)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def _plot_2d(self, z, figsize=(7, 5.5), title: str = None, **kwargs):
        """
        Plot a 2D representation of the given data.

        Args:
            z (numpy.ndarray): The data to be plotted.
            figsize (tuple, optional): The size of the figure
                (default is (7, 5.5)).
            title (str): Title.
        """
        _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.flipud(z), extent=[-1, 1, -1, 1])

        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f'Surface {kwargs.get("surface_index", None)} deviation to plane\n'
                         f'{kwargs.get("surface_type", None).capitalize()} surface')

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Deviation to plane [mm]", rotation=270)
        plt.grid(alpha=0.25)
        plt.show()

    def _plot_3d(self, x, y, z, figsize=(7, 5.5), title: str = None, **kwargs):
        """
        Plot a 3D surface plot of the given data.

        Args:
            x (numpy.ndarray): Array of x-coordinates.
            y (numpy.ndarray): Array of y-coordinates.
            z (numpy.ndarray): Array of z-coordinates.
            figsize (tuple, optional): Size of the figure (width, height).
                Default is (7, 5.5).
            title (str): Title.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                               figsize=figsize)

        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1,
                               cmap='viridis', linewidth=0,
                               antialiased=False)

        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.set_zlabel("Deviation to plane [mm]")
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f'Surface {kwargs.get("surface_index", None)} deviation to plane\n'
                         f'{kwargs.get("surface_type", None).capitalize()} surface')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                     pad=0.15)
        fig.tight_layout()
        plt.show()




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
             distribution='line_y', figsize=(10, 4), xlim=None, ylim=None,
             title=None, reference=None):
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
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.
        """
        _, ax = plt.subplots(figsize=figsize)

        self.rays.plot(ax, fields=fields, wavelengths=wavelengths,
                       num_rays=num_rays, distribution=distribution,
                       reference=reference)
        self.system.plot(ax)

        plt.gca().set_facecolor('#f8f9fa')  # off-white background
        plt.axis('image')

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
    """
    A class used to visualize optical systems in 3D.

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
        self.system = OpticalSystem(optic, self.rays, projection='3d')

        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    def view(self, fields='all', wavelengths='primary', num_rays=24,
             distribution='ring', figsize=(1200, 800), dark_mode=False,
             reference=None):
        """
        Visualizes the optical system in 3D.

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

        self.rays.plot(renderer, fields=fields, wavelengths=wavelengths,
                       num_rays=num_rays, distribution=distribution,
                       reference=reference)
        self.system.plot(renderer)

        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(
            vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL
        )

        if dark_mode:
            renderer.SetBackground(0.13, 0.15, 0.19)
            renderer.SetBackground2(0.195, 0.21, 0.24)
        else:
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
            g = surf.geometry

            # check if __str__ method exists
            if type(g).__dict__.get('__str__'):
                surf_type.append(str(surf.geometry))
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
            elif isinstance(surf.material_post, materials.AbbeMaterial):
                mat.append(f'{surf.material_post.index:.4f}, '
                           f'{surf.material_post.abbe:.2f}')
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
        print(df.to_markdown(headers='keys', tablefmt='fancy_outline'))
