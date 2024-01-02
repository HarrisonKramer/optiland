class Aperture:

    def __init__(self, aperture_type='EPD', value=10):
        if aperture_type not in ['EPD', 'imageFNO', 'objectNA']:
            raise ValueError('Aperrture type must be "EPD", "imageFNO", '
                             '"objectNA"')
        self.ap_type = aperture_type
        self.value = value
