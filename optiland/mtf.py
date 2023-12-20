from optiland.psf import FFTPSF


class FFTMTF(FFTPSF):

    def __init__(self, optic, field, wavelengths='all',
                 num_rays=128, grid_size=1024):
        super().__init__(optic, field, wavelengths, num_rays, grid_size)

    def view(self):
        pass

    def _compute_mtf(self):
        pass

    def _get_mtf_units(self):
        pass

    def _plot_2d(self):
        """Override to disable function"""
        pass

    def _plot_3d(self):
        """Override to disable function"""
        pass