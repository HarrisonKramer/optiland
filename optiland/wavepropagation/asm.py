import optiland.backend as be
from math import pi


class AngularSpectrumPropagator:
    def __init__(
        self,
        num_points: int,
        dx: float,
        evanescent: str = "clamp",
    ):
        self.dx = float(dx)
        self.evanescent = evanescent
        M_orig, N_orig = (num_points, num_points)
        self.Mpad = int(round(M_orig * 0.5))
        self.Npad = int(round(N_orig * 0.5))
        M = M_orig + 2 * self.Mpad
        N = N_orig + 2 * self.Npad
        fx = be.fftfreq(N, d=self.dx)
        fy = be.fftfreq(M, d=self.dx)
        FY, FX = be.meshgrid(fy, fx, indexing="ij")
        self.fx2_fy2 = (FX * FX + FY * FY)[None, None]

    def __call__(self, input_field, distance, wavelengths):
        return self.forward(input_field, distance, wavelengths)

    def forward(self, input_field, distance, wavelengths):
        if getattr(input_field, "ndim", None) != 4:
            raise ValueError("input_field must have shape (B, C, M, N).")

        B = input_field.shape[0]
        distance = be.array(distance).reshape(-1, 1, 1, 1)
        if distance.shape[0] == 1 and B > 1:
            distance = be.broadcast_to(distance, (B, 1, 1, 1))
        wavelengths = be.array(wavelengths).reshape(-1, 1, 1, 1)
        if wavelengths.shape[0] == 1 and B > 1:
            wavelengths = be.broadcast_to(wavelengths, (B, 1, 1, 1))

        k = 2.0 * pi / (wavelengths * 1e-3)

        if self.Mpad > 0 or self.Npad > 0:
            padded = be.pad(
                input_field, ((self.Mpad, self.Mpad), (self.Npad, self.Npad))
            )
        else:
            padded = input_field

        spectrum = be.fft2(padded)
        argument = k * k - (2.0 * pi) ** 2 * self.fx2_fy2

        if self.evanescent == "clamp":
            kz = be.sqrt(be.clamp(argument, min=0.0))
            H = be.exp(1j * be.to_complex(kz) * be.to_complex(distance))
        elif self.evanescent == "decay":
            kz = be.sqrt(be.to_complex(argument))
            H = be.exp(1j * kz * be.to_complex(distance))
        else:
            raise ValueError('evanescent must be "clamp" or "decay".')

        out = be.ifft2(spectrum * H)

        if self.Mpad > 0:
            out = out[:, :, self.Mpad : -self.Mpad, :]
        if self.Npad > 0:
            out = out[:, :, :, self.Npad : -self.Npad]

        return out
