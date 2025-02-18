"""Surface Factory

This module contains the SurfaceFactory class, which is used to create surface
objects based on the given parameters. The SurfaceFactory class is used by the
SurfaceGroup class to create surfaces for the optical system. The class
abstracts the creation of surface objects and allows for easy configuration of
the surface parameters.

Kramer Harrison, 2024
"""

import numpy as np
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import BaseMaterial, IdealMaterial, Material
from optiland.coatings import BaseCoating, FresnelCoating
from optiland.geometries import (
    Plane,
    StandardGeometry,
    EvenAsphere,
    OddAsphere,
    PolynomialGeometry,
    ChebyshevPolynomialGeometry,
    ZernikePolynomialGeometry,
)
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface


class SurfaceFactory:
    """
    A factory class for creating surface objects.

    Args:
        surface_group (SurfaceGroup): The surface group to which the surfaces
            belong.

    Attributes:
        _surface_group (SurfaceGroup): The surface group to which the surfaces
            belong.
        last_thickness (float): The thickness of the last created surface.
    """

    def __init__(self, surface_group):
        self._surface_group = surface_group
        self.last_thickness = 0
        self._use_absolute_cs = False
        self._last_surface = None  # placeholder for last surface object

    def create_surface(self, surface_type, index, is_stop, material,
                       **kwargs):
        """
        Create a surface object based on the given parameters.

        Args:
            surface_type (str): The type of surface to create.
            index (int): The index of the surface.
            is_stop (bool): Indicates whether the surface is a stop surface.
            material (str): The material of the surface.
            **kwargs: Additional keyword arguments for configuring the surface.

        Returns:
            Surface: The created surface object.

        Raises:
            ValueError: If the index is greater than the number of surfaces.
        """
        if index > self._surface_group.num_surfaces:
            raise ValueError('Surface index cannot be greater than number of '
                             'surfaces.')

        cs = self._configure_cs(index, **kwargs)
        material_pre, material_post = self._configure_material(index, material)
        coating = self.configure_coating(kwargs.get('coating', None),
                                         material_pre, material_post)
        bsdf = kwargs.get('bsdf', None)

        is_reflective = material == 'mirror'

        # Configuration for each surface type
        surface_config = {
            'standard': {
                'geometry': self._configure_standard_geometry,
                'expected_params': ['radius', 'conic']
            },
            'even_asphere': {
                'geometry': self._configure_even_asphere_geometry,
                'expected_params': ['radius', 'conic', 'coefficients']
            },
            'odd_asphere': {
                'geometry': self._configure_odd_asphere_geometry,
                'expected_params': ['radius', 'conic', 'coefficients']
            },
            'polynomial': {
                'geometry': self._configure_polynomial_geometry,
                'expected_params': ['radius', 'conic', 'coefficients',
                                    'tol', 'max_iter']
            },
            'chebyshev': {
                'geometry': self._configure_chebyshev_geometry,
                'expected_params': ['radius', 'conic', 'coefficients',
                                    'tol', 'max_iter', 'norm_x', 'norm_y']
            },
            'zernike': {
                'geometry': self._configure_zernike_geometry,
                'expected_params': ['radius', 'conic', 'coefficients',
                                    'tol', 'max_iter', 'norm_radius']
            },
        }

        if surface_type not in surface_config:
            raise ValueError(f'Surface type {surface_type} not recognized.')

        # Generate geometry for the surface type
        config = surface_config[surface_type]
        filtered_params = {key: value for key, value in kwargs.items()
                           if key in config['expected_params']}
        geometry = config['geometry'](cs, **filtered_params)

        if index == 0:
            return ObjectSurface(geometry, material_post)

        # Filter out unexpected surface parameters
        common_params = ['aperture']
        filtered_kwargs = {key: value for key, value in kwargs.items()
                           if key in common_params}

        return Surface(geometry, material_pre, material_post, is_stop,
                       is_reflective=is_reflective, coating=coating,
                       bsdf=bsdf, surface_type=surface_type, **filtered_kwargs)

    def _configure_cs(self, index, **kwargs):
        """
        Configures the coordinate system for a given surface.

        Args:
            index (int): The index of the surface.
            **kwargs: Additional keyword arguments for the coordinate system.
                Options include dx, dy, rx, ry.

        Returns:
            CoordinateSystem: The configured coordinate system.
        """
        if 'z' in kwargs:
            if 'thickness' in kwargs:
                raise ValueError('Cannot define both "thickness" and "z".')

            x = kwargs.get('x', 0)
            y = kwargs.get('y', 0)
            z = kwargs['z']
            self._use_absolute_cs = True
        else:
            if self._use_absolute_cs:
                raise ValueError('Cannot pass "thickness" after defining '
                                 '"x", "y", "z" position for a previous '
                                 'surface.')

            thickness = kwargs.get('thickness', 0)
            x = kwargs.get('dx', 0)
            y = kwargs.get('dy', 0)

            if index == 0:  # object surface
                z = -thickness
            elif index == 1:
                z = 0  # first surface, always at zero
            else:
                z = float(self._surface_group.positions[index-1].item()) + \
                    self.last_thickness

                # self.last_thickness = thickness

        rx = kwargs.get('rx', 0)
        ry = kwargs.get('ry', 0)
        rz = kwargs.get('rz', 0)

        return CoordinateSystem(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)

    @staticmethod
    def _configure_standard_geometry(cs, **kwargs):
        """
        Configures a standard geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius and conic.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)

        if np.isinf(radius):
            geometry = Plane(cs)
        else:
            geometry = StandardGeometry(cs, radius, conic)

        return geometry

    @staticmethod
    def _configure_even_asphere_geometry(cs, **kwargs):
        """
        Configures an even asphere geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius, conic, and coefficients.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)
        tol = kwargs.get('tol', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        coefficients = kwargs.get('coefficients', [])

        geometry = EvenAsphere(cs, radius, conic, tol, max_iter, coefficients)

        return geometry

    @staticmethod
    def _configure_odd_asphere_geometry(cs, **kwargs):
        """
        Configures an odd asphere geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius, conic, and coefficients.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)
        tol = kwargs.get('tol', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        coefficients = kwargs.get('coefficients', [])

        geometry = OddAsphere(cs, radius, conic, tol, max_iter, coefficients)

        return geometry

    @staticmethod
    def _configure_polynomial_geometry(cs, **kwargs):
        """
        Configures a polynomial geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius, conic, coefficients, tol, and max_iter.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)
        tol = kwargs.get('tol', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        coefficients = kwargs.get('coefficients', [])

        geometry = PolynomialGeometry(cs, radius, conic, tol, max_iter,
                                      coefficients)

        return geometry

    @staticmethod
    def _configure_chebyshev_geometry(cs, **kwargs):
        """
        Configures a Chebyshev geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius, conic, coefficients, tol, and max_iter.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)
        tol = kwargs.get('tol', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        coefficients = kwargs.get('coefficients', [])
        norm_x = kwargs.get('norm_x', 1)
        norm_y = kwargs.get('norm_y', 1)

        geometry = ChebyshevPolynomialGeometry(cs, radius, conic, tol,
                                               max_iter, coefficients,
                                               norm_x, norm_y)

        return geometry

    # @staticmethod
    def _configure_zernike_geometry(self, cs, **kwargs):
        """
        Configures a Zernike geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            **kwargs: Additional keyword arguments for the geometry. Options
                include radius, conic, coefficients, tol, and max_iter.

        Returns:
            geometry: The configured geometry object.
        """
        radius = kwargs.get('radius', np.inf)
        conic = kwargs.get('conic', 0)
        tol = kwargs.get('tol', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        coefficients = kwargs.get('coefficients', [])
        norm_radius = kwargs.get('norm_radius', 1)

        geometry = ZernikePolynomialGeometry(cs, radius, conic, tol,
                                             max_iter, coefficients,
                                             norm_radius)

        return geometry

    def _configure_material(self, index, material):
        """
        Configures the material for a surface based on the given index and
            material input.

        Args:
            index (int): The index of the surface.
            material (BaseMaterial, tuple, str): The material input for the
                surface. It can be an instance of BaseMaterial, a tuple
                containing the name and reference of the material, or a string
                representing the material. See examples.

        Returns:
            tuple: A tuple containing the material before and after the
                surface.
        """
        # object surface
        if index == 0:
            material_pre = None

        # image surface
        elif (index == self._surface_group.num_surfaces
              and self._last_surface is not None):
            material_pre = self._last_surface.material_post

        # intermediate surface
        else:
            previous_surface = self._surface_group.surfaces[index-1]
            material_pre = previous_surface.material_post

        if isinstance(material, BaseMaterial):
            material_post = material
        elif isinstance(material, tuple):
            material_post = Material(name=material[0], reference=material[1])
        elif isinstance(material, str):
            if material == 'air':
                material_post = IdealMaterial(n=1.0, k=0.0)
            elif material == 'mirror':
                material_post = material_pre
            else:
                material_post = Material(material)

        return material_pre, material_post

    def configure_coating(self, coating, material_pre, material_post):
        """
        Configures the coating for a surface based on the given index and
            coating input.

        Args:
            coating (BaseCoating, str): The coating input for the
                surface. It can be an instance of BaseCoating or a string
                representing the coating. See examples.
            material_pre (BaseMaterial): The material before the surface.
            material_post (BaseMaterial): The material after the surface.

        Returns:
            BaseCoating: The coating for the surface.
        """
        if isinstance(coating, BaseCoating):
            return coating
        elif isinstance(coating, str):
            if coating == 'fresnel':
                return FresnelCoating(material_pre, material_post)
        else:
            return None
