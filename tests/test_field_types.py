import optiland.backend as be
import pytest
from unittest.mock import MagicMock, patch

from optiland.fields.field_types import RealImageHeightField
from optiland.optic import Optic
from optiland.samples import objectives
from .utils import assert_allclose


def test_paraxial_image_height_infinite_object(set_test_backend):
    """Test paraxial image height field type for an object at infinity."""
    optic = Optic()
    optic.add_surface(index=0, thickness=be.inf)
    optic.add_surface(material="N-BK7", radius=50, thickness=5, index=1, is_stop=True)
    optic.add_surface(thickness=100, index=2)
    optic.add_surface(index=3)
    optic.set_field_type("paraxial_image_height")
    optic.add_field(y=10)
    optic.set_aperture("EPD", 10)
    optic.add_wavelength(0.58756, is_primary=True)

    # trace a chief ray
    y, u = optic.paraxial.chief_ray()

    # verify that the ray's y-coordinate at the image plane matches the
    # expected paraxial image height
    assert_allclose(y[-1], 10, rtol=1e-5)


def test_paraxial_image_height_finite_object(set_test_backend):
    """Test paraxial image height field type for a finite object."""
    optic = Optic()
    optic.add_surface(index=0, thickness=50)
    optic.add_surface(material="N-BK7", radius=50, thickness=5, index=1, is_stop=True)
    optic.add_surface(thickness=100, index=2)
    optic.add_surface(index=3)
    optic.set_field_type("paraxial_image_height")
    optic.add_field(y=10)
    optic.set_aperture("EPD", 10)
    optic.add_wavelength(0.58756, is_primary=True)

    # trace a chief ray
    y, u = optic.paraxial.chief_ray()

    # verify that the ray's y-coordinate at the image plane matches the
    # expected paraxial image height
    assert_allclose(y[-1], 9.67243803, rtol=1e-5)


def test_field_definition_to_dict(set_test_backend):
    """Test that field definition to_dict method works."""
    from optiland.fields.field_types import AngleField, ObjectHeightField, ParaxialImageHeightField
    field_defs = [AngleField(), ObjectHeightField(), ParaxialImageHeightField()]
    for field_def in field_defs:
        d = field_def.to_dict()
        assert d["field_type"] == field_def.__class__.__name__

def test_field_definition_from_dict(set_test_backend):
    """Test that field definition from_dict method works."""
    from optiland.fields.field_types import AngleField, ObjectHeightField, ParaxialImageHeightField, BaseFieldDefinition
    field_defs = [AngleField(), ObjectHeightField(), ParaxialImageHeightField()]
    for field_def in field_defs:
        d = field_def.to_dict()
        new_field_def = BaseFieldDefinition.from_dict(d)
        assert isinstance(new_field_def, field_def.__class__)

def test_paraxial_image_height_cooke_triplet(set_test_backend):
    """Test that paraxial image height field type is equivalent to angle field
    type for a Cooke triplet."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # compute the chief ray with the default angle field type
    y_chief_angle, u_chief_angle = optic.paraxial.chief_ray()

    # get the paraxial image height for the original system
    paraxial_image_height = y_chief_angle[-1]

    # change the field type to paraxial image height and set the equivalent
    # field value
    optic.set_field_type("paraxial_image_height")
    optic.fields.fields = []
    optic.add_field(y=0)
    optic.add_field(y=paraxial_image_height[0])

    # recompute the chief ray
    y_chief_pih, u_chief_pih = optic.paraxial.chief_ray()

    # check that the two chief rays are allclose
    assert_allclose(y_chief_angle, y_chief_pih)
    assert_allclose(u_chief_angle, u_chief_pih)


def test_paraxial_get_ray_origins(set_test_backend):
    """Test that paraxial image height field type get_ray_origins method
    works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.set_field_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.add_field(y=0)
    optic.add_field(y=20.0)

    # get the ray origins
    origins = optic.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # check that the origins are correct
    assert_allclose(origins[0], [0])
    assert_allclose(origins[1], [-8.63986616])
    assert_allclose(origins[2], [-10])

    # Change to finite object
    optic = objectives.CookeTriplet()
    optic.object_surface.geometry.cs.z = be.array([-555.0])

    # get the ray origins
    origins = optic.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # check that the origins are correct
    assert_allclose(origins[0], [0.0])
    assert_allclose(origins[2], [-555.0])


def test_get_paraxial_object_position(set_test_backend):
    """Test that paraxial image height field type get_paraxial_object_position
    method works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.set_field_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.add_field(y=0)
    optic.add_field(y=20.0)

    # get the paraxial object position
    obj_pos = optic.field_definition.get_paraxial_object_position(optic, Hy=1, y1=0.5, EPL=optic.paraxial.EPL())

    # check that the object position is correct (should be at infinity)
    assert_allclose(obj_pos[0], [-4.12359504])
    assert_allclose(obj_pos[1], [0.0])

    # Change to finite object
    optic = objectives.CookeTriplet()
    optic.object_surface.geometry.cs.z = be.array([-555.0])

    # get the paraxial object position
    obj_pos = optic.field_definition.get_paraxial_object_position(optic, Hy=1, y1=0.5, EPL=optic.paraxial.EPL())

    # check that the object position is correct
    assert_allclose(obj_pos[0], [-3.69008309])
    assert_allclose(obj_pos[1], [0.0])


def test_get_starting_z_offset(set_test_backend):
    """Test that paraxial image height field type get_starting_z_offset method
    works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.set_field_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.add_field(y=0)
    optic.add_field(y=20.0)

    # get the starting z offset
    z_offset = optic.field_definition._get_starting_z_offset(optic)

    # check that the z offset is correct
    assert_allclose(z_offset, optic.paraxial.EPD())


@pytest.fixture
def mock_optic(request):
    """A mock optic that can be configured for infinite or finite conjugate."""
    is_infinite = request.param
    optic = MagicMock(spec=Optic)

    # Mock basic properties
    optic.fields = MagicMock()
    optic.fields.max_field = 10.0
    optic.fields.max_y_field = 10.0

    optic.object_surface = MagicMock()
    optic.object_surface.is_infinite = is_infinite

    optic.surface_group = MagicMock()
    optic.surface_group.positions = be.array([0, 10, 110])
    optic.surface_group.stop_index = 1
    optic.surface_group.num_surfaces = 3

    optic.primary_wavelength = 0.58756

    # Mock paraxial properties
    optic.paraxial = MagicMock()
    optic.paraxial.EPL.return_value = 10.0
    optic.paraxial.EPD.return_value = 20.0

    # Mock geometry for finite case
    optic.object_surface.geometry = MagicMock()
    optic.object_surface.geometry.sag.return_value = 0.0
    optic.object_surface.geometry.cs = MagicMock()
    optic.object_surface.geometry.cs.z = -50.0

    # Mock surface_group's y attribute, which is where the result is stored
    optic.surface_group.y = be.zeros((3, 1))
    optic.surface_group.x = be.zeros((3, 1))

    # Mock paraxial tracing
    if is_infinite:
        paraxial_trace_results = (10.0, 0.1, 0.0, 0.1) # y_img_unit, u_img_unit, y_obj_unit, u_obj_unit
    else:
        paraxial_trace_results = (10.0, 0.1, 5.0, 0.0)

    def mock_trace_unit_chief_ray(optic_arg, plane="image"):
        if plane == "image":
            return paraxial_trace_results[0], paraxial_trace_results[1]
        else:
            return paraxial_trace_results[2], paraxial_trace_results[3]

    with patch('optiland.fields.field_types.ParaxialImageHeightField._trace_unit_chief_ray', side_effect=mock_trace_unit_chief_ray):
        yield optic


@pytest.mark.parametrize("mock_optic", [True, False], indirect=True)
def test_real_image_height_vs_paraxial_perfect_lens(set_test_backend, mock_optic):
    """Test RealImageHeightField against ParaxialImageHeightField for a perfect lens."""

    def trace_generic(**kwargs):
        """Simulate a perfect lens where real trace equals paraxial prediction."""
        Hy = kwargs.get('Hy', 0.0)
        max_field = mock_optic.fields.max_field

        paraxial_image_height = 10.0
        height = Hy * paraxial_image_height

        mock_optic.surface_group.y[-1, 0] = height
        return None

    mock_optic.trace_generic = MagicMock(side_effect=trace_generic)

    field_def = RealImageHeightField()
    target_height = 5.0

    real_prop = field_def._find_chief_ray_object_properties(mock_optic, target_height, axis='y')

    paraxial_prop = mock_optic.fields.max_field * (target_height / 10.0)

    assert_allclose(real_prop, paraxial_prop, rtol=1e-5)


@pytest.mark.parametrize("mock_optic", [True], indirect=True)
def test_real_image_height_with_distortion(set_test_backend, mock_optic):
    """Test RealImageHeightField with simulated pincushion and barrel distortion."""
    paraxial_image_height = 10.0
    target_height = 5.0
    paraxial_prop = mock_optic.fields.max_field * (target_height / paraxial_image_height)

    def pincushion_trace(**kwargs):
        Hy = kwargs.get('Hy', 0.0)
        linear_height = Hy * paraxial_image_height
        distorted_height = linear_height * (1 + 0.2 * Hy**2)
        mock_optic.surface_group.y[-1, 0] = distorted_height
        return None

    def barrel_trace(**kwargs):
        Hy = kwargs.get('Hy', 0.0)
        linear_height = Hy * paraxial_image_height
        distorted_height = linear_height * (1 - 0.2 * Hy**2)
        mock_optic.surface_group.y[-1, 0] = distorted_height
        return None

    field_def = RealImageHeightField()

    mock_optic.trace_generic = MagicMock(side_effect=pincushion_trace)
    pincushion_prop = field_def._find_chief_ray_object_properties(mock_optic, target_height, axis='y')
    assert pincushion_prop < paraxial_prop

    mock_optic.trace_generic = MagicMock(side_effect=barrel_trace)
    barrel_prop = field_def._find_chief_ray_object_properties(mock_optic, target_height, axis='y')
    assert barrel_prop > paraxial_prop


def test_real_image_height_serialization(set_test_backend):
    """Test to_dict and from_dict for RealImageHeightField."""
    from optiland.fields.field_types import BaseFieldDefinition
    field_def = RealImageHeightField()
    d = field_def.to_dict()
    assert d == {"field_type": "RealImageHeightField"}

    new_field_def = BaseFieldDefinition.from_dict(d)
    assert isinstance(new_field_def, RealImageHeightField)


@pytest.mark.parametrize("mock_optic", [True], indirect=True)
def test_real_image_height_solver_failure(set_test_backend, mock_optic):
    """Test that a RuntimeError is raised when the solver fails."""
    def non_converging_trace(**kwargs):
        mock_optic.surface_group.y[-1, 0] = 1.0
        return None

    mock_optic.trace_generic = MagicMock(side_effect=non_converging_trace)

    field_def = RealImageHeightField()
    with pytest.raises(RuntimeError, match="Solver failed to converge"):
        field_def._find_chief_ray_object_properties(mock_optic, 5.0, axis='y')


@pytest.mark.parametrize("mock_optic", [True, False], indirect=True)
def test_real_image_height_get_ray_origins(set_test_backend, mock_optic):
    """Test get_ray_origins for RealImageHeightField."""

    def dummy_trace(**kwargs):
        mock_optic.surface_group.y[-1, 0] = kwargs.get('Hy', 0.0) * 10.0
        mock_optic.surface_group.x[-1, 0] = kwargs.get('Hx', 0.0) * 10.0
        return None
    mock_optic.trace_generic = MagicMock(side_effect=dummy_trace)

    field_def = RealImageHeightField()
    origins = field_def.get_ray_origins(mock_optic, Hx=0.5, Hy=1.0, Px=be.array([0.1]), Py=be.array([0.2]), vx=1.0, vy=1.0)

    assert origins is not None
    assert len(origins) == 3

    if mock_optic.object_surface.is_infinite:
        assert origins[2][0] != -50.0
    else:
        assert origins[2][0] == -50.0