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
from scipy import optimize

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
             projection: str = '2d',
             plot_dev_to_bfs: bool = False,
             num_points: int = 256,
             figsize: tuple = (7, 5.5),
             title: str = None):
        """
        Visualize the surface.

        Args:
            surface_index (int): Index of the surface to be visualized.
            projection (str): The type of projection to use for visualization.
            Can be '2d' or '3d'.
            plot_dev_to_bfs (bool): If True, plot the deviation to the best 
                fir sphere instead of the deviation to a plane.
            num_points (int): The number of points to sample along each axis
                for the visualization.
            figsize (tuple): The size of the figure in inches.
                Defaults to (7, 5.5).
            title (str): Title.

        Raises:
            ValueError: If the projection is not '2d' or '3d'.
        """
        # Update optics and compute surface sag
        self.optic.update_paraxial()
        surface = self.optic.surface_group.surfaces[surface_index]
        semi_aperture = surface.semi_aperture
        x, y = np.meshgrid(
            np.linspace(-semi_aperture, semi_aperture, num_points),
            np.linspace(-semi_aperture, semi_aperture, num_points),)
        
        z = surface.geometry.sag(x, y)

        if plot_dev_to_bfs:
            z = self._compute_deviation_to_best_fit_sphere(x, y, z)

        z[np.sqrt(x**2+y**2) > semi_aperture] = np.nan

        # Plot in 2D
        if projection == '2d':
            self._plot_2d(z, figsize=figsize, title=title, 
                          surface_type=surface.surface_type,
                          surface_index=surface_index,
                          semi_aperture=semi_aperture,
                          plot_dev_to_bfs=plot_dev_to_bfs)
        # Plot in 3D
        elif projection == '3d':
            self._plot_3d(x, y, z, figsize=figsize, title=title,
                          surface_type=surface.surface_type,
                          surface_index=surface_index,
                          semi_aperture=semi_aperture,
                          plot_dev_to_bfs=plot_dev_to_bfs)
        else:
            raise ValueError('Projection must be "2d" or "3d".')

    def _plot_2d(self,
                 z: np.ndarray,
                 figsize: tuple = (7, 5.5),
                 title: str = None,
                 **kwargs):
        """
        Plot a 2D representation of the given data.

        Args:
            z (numpy.ndarray): The data to be plotted.
            figsize (tuple, optional): The size of the figure
                (default is (7, 5.5)).
            title (str): Title.
        """
        _, ax = plt.subplots(figsize=figsize)

        semi_aperture = kwargs['semi_aperture']
        extent = [-semi_aperture, semi_aperture, -semi_aperture, semi_aperture]
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        im = ax.imshow(np.flipud(z), extent=extent)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(
                f'Surface {kwargs.get("surface_index", None)} '
                f'deviation to '
                f'{"BFS" if kwargs.get("plot_dev_to_bfs", False) else "plane"}\n'
                f'{kwargs.get("surface_type", None).capitalize()} surface'
            )

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Deviation to plane [mm]", rotation=270)
        plt.grid(alpha=0.25)
        plt.show()

    def _plot_3d(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 figsize: tuple = (7, 5.5),
                 title: str = None,
                 **kwargs):
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

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel("Deviation to plane [mm]")

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(
                f'Surface {kwargs.get("surface_index", None)} '
                f'deviation to '
                f'{"BFS" if kwargs.get("plot_dev_to_bfs", False) else "plane"}\n'
                f'{kwargs.get("surface_type", None).capitalize()} surface'
            )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                     pad=0.15)
        
        fig.tight_layout()
        plt.show()

    @staticmethod
    def _sphere_sag(x, y, R):
        """Compute the sag of a sphere with radius R.
        
        Args:
            x, y: 2D arrays of coordinates.
            R: Sphere radius.

        Returns:
            2D array of sag values.
        """
        return R - np.sqrt(R**2 - x**2 - y**2)

    def _best_fit_sphere(self, x, y, z, radius):
        """Find the best-fit sphere radius.
        
        Args:
            x, y: 2D arrays of coordinates.
            z: 2D array of sags.

        Returns:
            Optimal sphere radius.
        """
        def rms_error(R):
            z_s = self._sphere_sag(x, y, R)
            return np.sum((z - z_s) ** 2)  # RMS error

        initial_guess = np.max(np.sqrt(x**2 + y**2))
        res = optimize.minimize(rms_error, initial_guess)
        return res.x[0]

    def _compute_deviation_to_best_fit_sphere(self, x, y, z):
        """Compute deviation from the best-fit sphere.
        
        Args:
            x, y: 2D arrays of coordinates.
            z: 2D array of sags.

        Returns:
            2D array of deviation values.
        """
        R = self._best_fit_sphere(x, y, z)
        print("R=", R)
        z_s = self._sphere_sag(x, y, R)
        return z - z_s
    

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
