class RadialAperture:

    def __init__(self, r_max, r_min=0):
        self.r_max = r_max
        self.r_min = r_min

    def clip(self, rays):
        radius2 = rays.x**2 + rays.y**2
        condition = (radius2 > self.r_max**2) | (radius2 < self.r_min**2)
        rays.clip(condition)
