class Aperture:

    def __init__(self, aperture_type, value, object_space_telecentric=False):
        if aperture_type not in ['EPD', 'imageFNO', 'objectNA']:
            raise ValueError('Aperture type must be "EPD", "imageFNO", '
                             '"objectNA"')

        if value in ['EPD', 'imageFNO'] and object_space_telecentric:
            raise ValueError('Cannot set aperture type to "EPD" or "imageFNO" '
                             'if lens is telecentric in object space.')

        self.ap_type = aperture_type
        self.value = value
        self.object_space_telecentric = object_space_telecentric
