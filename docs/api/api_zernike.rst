Zernike
=======

This section contains the Zernike modules of Optiland. This subpackage provides
functionality for working with Zernike polynomials, which
are used to represent wavefront aberrations. The `ZernikeStandard` class
implements the OSA/ANSI standard Zernike polynomials, allowing for the
calculation of Zernike terms and the evaluation of Zernike polynomial series
for given radial and azimuthal coordinates. The module also provides classes
for Zernike Fringe (University of Arizona) and Zernike Noll indices. Lastly,
a `ZernikeFit` class is provided for fitting the various Zernike polynomial
types to data points.

.. autosummary::
   :toctree: zernike/
   :caption: Zernike Modules

   zernike.base
   zernike.standard
   zernike.fringe
   zernike.noll
   zernike.fit
