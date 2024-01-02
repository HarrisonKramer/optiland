import numpy as np
from optiland.fields import Field, FieldGroup
from optiland.surfaces import SurfaceGroup, ObjectSurface
from optiland.wavelength import WavelengthGroup
from optiland.paraxial import Paraxial
from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.rays import RealRays
from optiland.distribution import create_distribution
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.visualization import LensViewer


class Optic:
    def __init__(self):
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)

    @property
    def primary_wavelength(self):
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self):
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self):
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self):
        z = self.surface_group.positions[1:-1]
        return np.max(z) - np.min(z)

    def add_surface(self, new_surface=None, index=None, thickness=0,
                    radius=np.inf, material='air', conic=0, is_stop=False,
                    dx=0, dy=0, rx=0, ry=0, aperture=None):
        self.surface_group.add_surface(new_surface, index, thickness, radius,
                                       material, conic, is_stop,
                                       dx, dy, rx, ry, aperture)

    def add_field(self, y, x=0):
        new_field = Field(self.field_type, x, y)
        self.fields.add_field(new_field)

    def add_wavelength(self, value, is_primary=False, unit='um'):
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type):
        self.field_type = field_type

    def set_radius(self, value, surface_number):
        surface = self.surface_group.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_thickness(self, value, surface_number):
        positions = self.surface_group.positions
        delta_t = value - positions[surface_number+1] + \
            positions[surface_number]
        positions[surface_number+1:] += delta_t
        positions -= positions[1]  # force surface 1 to be at zero
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.geometry.cs.z = positions[k]

    def set_index(self, value, surface_number):
        surface = self.surface_group.surfaces[surface_number]
        new_material = IdealMaterial(n=value, k=0)
        surface.material_post = new_material

        surface_post = self.surface_group.surfaces[surface_number+1]
        surface_post.material_pre = new_material

    def draw(self, projection='2d', fields='all', wavelengths='primary',
             num_rays=3, figsize=(10, 4)):
        if projection == '2d':
            viewer = LensViewer(self)
            viewer.view(fields, wavelengths, num_rays, distribution='line_y',
                        figsize=figsize)
        elif projection == '3d':
            pass
        else:
            raise ValueError('Projection must be "2d" or "3d".')

    def reset(self):
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)

    def n(self, wavelength='primary'):
        if wavelength == 'primary':
            wavelength = self.primary_wavelength
        n = []
        for surface in self.surface_group.surfaces:
            n.append(surface.material_post.n(wavelength))
        return np.array(n)

    def update_paraxial(self):
        ya, _ = self.paraxial.marginal_ray()
        yb, _ = self.paraxial.chief_ray()
        ya = np.abs(np.ravel(ya))
        yb = np.abs(np.ravel(yb))
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.set_semi_aperture(r_max=ya[k]+yb[k])

    def trace(self, Hx, Hy, wavelength, num_rays=100,
              distribution='hexapolar'):
        EPL = self.paraxial.EPL()
        EPD = self.paraxial.EPD()

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        x1 = distribution.x * EPD / 2
        y1 = distribution.y * EPD / 2
        z1 = np.ones_like(x1) * EPL

        rays = self._generate_rays(Hx, Hy, x1, y1, z1, wavelength, EPL)
        self.surface_group.trace(rays)

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        EPL = self.paraxial.EPL()
        EPD = self.paraxial.EPD()

        x1 = Px * EPD / 2
        y1 = Py * EPD / 2

        # assure all variables are arrays of the same size
        max_size = max([np.size(arr) for arr in [x1, y1, Hx, Hy]])
        x1, y1, Hx, Hy = [
            np.full(max_size, value) if isinstance(value, (float, int))
            else value if isinstance(value, np.ndarray)
            else None
            for value in [x1, y1, Hx, Hy]
        ]

        z1 = np.ones_like(x1) * EPL

        rays = self._generate_rays(Hx, Hy, x1, y1, z1, wavelength, EPL)
        self.surface_group.trace(rays)

    def _generate_rays(self, Hx, Hy, x1, y1, z1, wavelength, EPL):
        x0, y0, z0 = self._get_object_position(Hx, Hy, x1, y1, EPL)

        mag = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        x0 = np.ones_like(x1) * x0
        y0 = np.ones_like(x1) * y0
        z0 = np.ones_like(x1) * z0

        energy = np.ones_like(x1)
        wavelength = np.ones_like(x1) * wavelength

        return RealRays(x0, y0, z0, L, M, N, energy, wavelength)

    def _get_object_position(self, Hx, Hy, x1, y1, EPL):
        obj = self.object_surface
        max_field = self.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.field_type == 'object_height':
                raise ValueError('''Field type cannot be "object_height" for an
                                 object at infinity.''')

            # start rays just before left-most surface (1/7th of total track)
            z = self.surface_group.positions[1:-1]
            offset = self.total_track / 7 - np.min(z)

            # x, y, z positions of ray starting points
            x = np.tan(np.radians(field_x)) * (offset + EPL)
            y = -np.tan(np.radians(field_y)) * (offset + EPL)
            z = self.surface_group.positions[1] - offset

            x0 = x1 + x
            y0 = y1 + y
            z0 = np.ones_like(x1) * z
        else:
            if self.field_type == 'object_height':
                x = field_x
                y = -field_y
                z = obj.geometry.sag(x, y) + obj.geometry.cs.z

                x0 = np.ones_like(x1) * x
                y0 = np.ones_like(x1) * y
                z0 = np.ones_like(x1) * z

            elif self.field_type == 'angle':
                x = np.tan(np.radians(field_x))
                y = -np.tan(np.radians(field_y))
                z = self.surface_group.positions[0]

                x0 = x1 + x
                y0 = y1 + y
                z0 = np.ones_like(x1) * z

        return x0, y0, z0
