"""Odd Asphere Geometry

The Even Asphere geometry represents a surface defined by an even asphere in
two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^i)

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Ci are the aspheric coefficients

Kramer Harrison, 2025
"""
