OPD, PSF, MTF
=============

This section provides an overview of the wavefront-related analyses available in Optiland,
including wavefront error, Zernike decomposition, point spread function (PSF) and modulation transfer function (MTF) calculations.

Wavefront Analysis
------------------

These modules handle optical path difference (OPD) calculations and wavefront analysis.

.. autosummary::
   :toctree: wavefront/wavefront/
   :caption: Wavefront & OPD

   wavefront.opd_fan
   wavefront.opd
   wavefront.wavefront_data
   wavefront.wavefront
   wavefront.zernike_opd

Modulation Transfer Function (MTF)
----------------------------------

The MTF modules provide different methods for calculating the modulation transfer function,
which characterizes the spatial frequency response of optical systems.

.. autosummary::
   :toctree: wavefront/mtf/
   :caption: MTF Analysis

   mtf.base
   mtf.fft
   mtf.geometric
   mtf.sampled

Point Spread Function (PSF)
---------------------------

PSF modules calculate the point spread function using various computational methods,
from fast Fourier transforms to more rigorous diffraction calculations.

.. autosummary::
   :toctree: wavefront/psf/
   :caption: PSF Calculation

   psf.base
   psf.fft
   psf.huygens_fresnel
