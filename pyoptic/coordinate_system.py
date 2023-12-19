from pyoptic.rays import RealRays


class CoordinateSystem:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0,
                 rx: float = 0, ry: float = 0, rz: float = 0,
                 reference_cs: 'CoordinateSystem' = None):
        self.x = x
        self.y = y
        self.z = z

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.reference_cs = reference_cs

    def localize(self, rays):
        if self.reference_cs:
            self.reference_cs.localize(rays)

        rays.translate(-self.x, -self.y, -self.z)
        if self.rx:
            rays.rotate_x(-self.rx)
        if self.ry:
            rays.rotate_y(-self.ry)
        if self.rz:
            rays.rotate_z(-self.rz)

    def globalize(self, rays):
        if self.rz:
            rays.rotate_z(self.rz)
        if self.ry:
            rays.rotate_y(self.ry)
        if self.rx:
            rays.rotate_x(self.rx)
        rays.translate(self.x, self.y, self.z)

        if self.reference_cs:
            self.reference_cs.globalize(rays)

    @property
    def position_in_gcs(self):
        vector = RealRays(0, 0, 0, 0, 0, 1, 1, 1)
        self.globalize(vector)
        return vector.x, vector.y, vector.z
