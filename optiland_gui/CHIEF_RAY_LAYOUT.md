# Chief ray in system layouts

In **2D Layout → Viewer Settings**, **Highlight chief ray** shows the beam-center
ray for each field in blue, slightly thicker and above the other rays. Unchecking
restores its ordinary ray color and line width. This takes effect immediately,
without recalculating the design or moving the view.

The chief ray uses normalized pupil coordinates **Px=Py=0**, with the design's
configured aiming, field and polarization. It can leave the source at an angle
in an off-axis/decentered design. It is not a zero-angle construction line.
Blocked rays still stop at their first blocking surface.

**Num Rays includes the chief**, per field at the primary wavelength:

- **1** displays only the chief.
- **line_x / line_y:** 1, 3, 5… rays, with equal pairs on either side. Typing an
  even total rounds upward, and the control displays the actual total.
- **hexapolar:** complete bundles of 1, 7, 19, 37, 61 or 91 rays; arrows advance
  to the next bundle. The number now means rays, not rings.
- **random:** the chief plus the remaining randomly sampled rays.

Apply count/distribution changes with **Apply**. The checkbox changes appearance
only and never adds another ray. Clipping can prevent some requested rays from
reaching the image. Multiple fields each receive their own bundle.

When displaying only the chief, a few undisplayed rim rays determine auto-sized
lens outlines; finite physical apertures and the schematic image marker keep their
existing precedence. This does not change the prescription.

The **3D Layout** includes the chief alongside its existing 24 rim rays using
ordinary ray color and normal depth/occlusion. The 2D checkbox has no 3D overlay
effect. Scientific analysis and public Python pupil distributions are unchanged.
