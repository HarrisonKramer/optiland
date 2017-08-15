# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:51:36 2017

@author: kramer
"""

'''
In this tutorial, we will create a new_lens and perform different analyses

'''

# import the lens class from lens.py
from lens import lens

# create a lens object named "singlet". We assign the variable "new_lens" to this object
new_lens = lens(name='singlet')

# Add 4 surfaces: object surface, R1 of singlet, R2 of singlet, and the image plane
# The singlet will be immersed in air (default material type) and will be made of N-SF11
# The radii are arbitrarily chosen, R1 = 100 and R2 = infinity (default)
# The stop is chosen to be at the R1 surface and the lens thickness is chosen to be 5
# Note the units for radii, thickness, and diameters are lens units. Also note that the
# numbering for surfaces starts at 0
new_lens.add_surface(number=0, thickness=1e10, comment='object')
new_lens.add_surface(number=1, thickness=5, radius=50, stop=True, material='N-SF11')
new_lens.add_surface(number=2, thickness=60.9159)
new_lens.add_surface(number=3, comment='image')

# Add aperture: we set the aperture type to EPD (entrance pupil diameter) and the value to 25.4
new_lens.aperture_type = 'EPD'
new_lens.aperture_value = 25.4

# Add fields: we set the field type to angle (units are degrees) and add 3 field values
# at 0, 10, and 14 degrees. Note that field numbering starts at 0.
new_lens.field_type = 'angle'
new_lens.add_field(number=0, x=0, y=0)
new_lens.add_field(number=1, x=0, y=10)
new_lens.add_field(number=2, x=0, y=14)

# Add wavelength: this lens is designed for the visible spectrum, so we choose 3 wavelengths
# at the D, F, and C lines
new_lens.add_wavelength(number=0, value=486.1)
new_lens.add_wavelength(number=1, value=587.6, primary=True)
new_lens.add_wavelength(number=2, value=656.3)

# Lastly, we set the lens apertures for zero vignetting. This means all light passing through
# our entrance pupil makes it through the system.
new_lens.set_SA()

# Now we will show a 2D rendering of this plot. First, we import the function "plot2D"
# from the visualize module
from visualize import plot2D

# We plot our new lens using the syntax below
plot2D(new_lens).show()

# Next, we will look at the ray aberration fans of this lens at each field.
# To do this we will need to import the analysis module
import analysis

# Show the ray aberration plot for field 0 (0 degrees). We use normalized field
# coordinates when specifying the field value. This is defined as the field value
# divided by the maximum field value for X and Y separately.
analysis.ray_fan(new_lens, Hx=0, Hy=0)

# show for the 10 degree field. So, Hy = 10/14 = 0.714
analysis.ray_fan(new_lens, Hx=0, Hy=0.714)

# Lastly, show the 14 degree field. So, Hy = 14/14 = 1
analysis.ray_fan(new_lens, Hx=0, Hy=1)

# Next, we will perform spot diagram analyses for each field of our lens.
# To do so, we need to first define the pupil distribution that we wish to use
# for our spot diagrams. This requires the distribution class from the distribution
# module.
from distribution import distribution

# There are several distribution types that can be used. We will use a "hexapolar"
# distribution with 100 points. Note, the actual number of points may vary from the
# input value. This should only be treated as approximate
pupil_dist = distribution(type_='hexapolar', num_pts=100)

# After creating the distribution we plot the points.
pupil_dist.show()

# Now that the pupil distribution has been defined, we will use it to create spot
# diagrams at each field. We start with the 0 degree field, therefore Hy=0
analysis.spot_diagram_standard(new_lens, pupil_dist, Hx=0, Hy=0)































