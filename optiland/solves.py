"""Solves Module

The solves module is used to model the application of solves to an optic. A
solve is an operation that adjusts a property of the optic or a surface to
satisfy a specific condition. For example, a solve can adjust the height of a
marginal ray to a specified value on a specific surface.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod
from optiland.optimization import OptimizationProblem, OptimizerGeneric
from numpy import inf

class BaseSolve(ABC):
    """
    Applies a solve operation.

    This method should be implemented by subclasses to define the specific
    behavior of the solve operation.

    Raises:
        NotImplementedError: If the method is not implemented by the subclass.
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseSolve._registry[cls.__name__] = cls

    @abstractmethod
    def apply(self):
        pass  # pragma: no cover

    def to_dict(self):
        """
        Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve.
        """
        return {
            'type': self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, optic, data):
        """
        Creates a solve from a dictionary representation.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve.

        Returns:
            BaseSolve: The solve.
        """
        solve_type = data['type']
        if solve_type not in BaseSolve._registry:
            raise ValueError(f'Unknown solve type: {solve_type}')
        solve_class = BaseSolve._registry[data['type']]
        return solve_class.from_dict(optic, data)
    
class MarginalRayHeightSolve(BaseSolve):
    """
    Initializes a MarginalRayHeightSolve object.

    Args:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface.
        height (float): The height of the ray.
    """
    def __init__(self, optic, surface_idx, height):
        self.optic = optic
        self.surface_idx = surface_idx
        self.height = height

    def apply(self):
        """Applies the MarginalRayHeightSolve to the optic."""
        ya, ua = self.optic.paraxial.marginal_ray()
        offset = (self.height - ya[self.surface_idx]) / ua[self.surface_idx]

        # shift current surface and all subsequent surfaces
        for surface in self.optic.surface_group.surfaces[self.surface_idx:]:
            surface.geometry.cs.z += offset

    def to_dict(self):
        """
        Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve.
        """
        solve_dict = super().to_dict()
        solve_dict.update({
            'surface_idx': self.surface_idx,
            'height': self.height
        })
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """
        Creates a MarginalRayHeightSolve from a dictionary representation.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve.

        Returns:
            MarginalRayHeightSolve: The solve.
        """
        return cls(optic, data['surface_idx'], data['height'])

class QuickFocusSolve(BaseSolve):
    """ Quick Focus
    Args:
        optic (Optic): The optic object.

    Raises:
            ValueError: If the optical system is not defined.
    """
    def __init__(self, optic):
        self.optic = optic
        self.num_surfaces = self.optic.surface_group.num_surfaces
        self.last_surf_idx = self.optic.surface_group.num_surfaces - 2 # Index of the last surface before the image surface.
        if self.num_surfaces > 2:
            pos2 = self.optic.surface_group.positions[self.last_surf_idx]
            pos1 = self.optic.surface_group.positions[self.last_surf_idx-1]
            self.thickness = pos2 - pos1
        elif self.num_surfaces <= 2:
            raise ValueError('Can not optimize for an empty optical system')
        
    def apply(self):
        problem = OptimizationProblem()
        for wave in self.optic.wavelengths.get_wavelengths():
            for field in self.optic.fields.get_field_coords():
                input_data = {'optic': self.optic, 'Hx': field[0], 'Hy': field[1], 'num_rays': 3, 'wavelength': wave, 'distribution': 'gaussian_quad'}
                problem.add_operand(operand_type='OPD_difference', target=0, weight=1, input_data=input_data)

        problem.add_variable(self.optic, 'thickness', surface_number=self.last_surf_idx, min_val=0, max_val=1000)
        optimizer = OptimizerGeneric(problem)
        optimizer.optimize()
        pos2_optimized = self.optic.surface_group.positions[self.last_surf_idx]
        pos1_optimized = self.optic.surface_group.positions[self.last_surf_idx-1]
        self.thickness = pos2_optimized - pos1_optimized

    def to_dict(self):
        """
        Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve.
        """
        solve_dict = super().to_dict()
        solve_dict.update({
            'thickness': float(self.thickness)
        })
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """
        Creates a QuickFocusSolve from a dictionary representation.

        Args:
            optic (Optic): The optic object.

        Returns:
            QuickFocusSolve: The solve.
        """
        instance = cls(optic)
        instance.thickness = data['thickness']
        return instance

class SolveFactory:
    """
    Factory class for creating solves.

    Attributes:
        _solve_map (dict): A dictionary mapping solve types to solve classes.

    Methods:
        create_solve(solve_type, *args, **kwargs): Creates a solve instance
            based on the given solve type.
    """
    _solve_map = {
        'marginal_ray_height': MarginalRayHeightSolve,
        'quick_focus': QuickFocusSolve
    }

    @staticmethod
    def create_solve(optic, solve_type, surface_idx, *args, **kwargs):
        """
        Creates a solve instance based on the given solve type.

        Args:
            optic (Optic): The optic object.
            solve_type (str): The type of solve to create.
            surface_idx (int): The index of the surface.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            solve_instance: An instance of the solve class corresponding to
                the given solve type.

        Raises:
            ValueError: If the solve type is invalid.
        """
        solve_class = SolveFactory._solve_map.get(solve_type)
        if solve_class is None:
            raise ValueError(f'Invalid solve type: {solve_type}')
        return solve_class(optic, surface_idx, *args, **kwargs)


class SolveManager:
    """
    Manages the application of solves to an optic.

    Args:
        optic (Optic): The optic object

    Attributes:
        solves (list): A list of solve instances.

    Methods:
        add(solve_type, surface_idx, *args, **kwargs): Adds a solve
            instance to the list of solves.
        apply(): Applies all solves in the list.
    """
    def __init__(self, optic):
        self.optic = optic
        self.solves = []

    def __len__(self):
        return len(self.solves)

    def add(self, solve_type, surface_idx, *args, **kwargs):
        """
        Adds a solve instance to the list of solves.

        Args:
            solve_type (str): The type of solve to create.
            surface_idx (int): The index of the surface.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        solve = SolveFactory.create_solve(self.optic, solve_type, surface_idx,
                                          *args, **kwargs)
        solve.apply()
        self.solves.append(solve)

    def apply(self):
        """Applies all solves in the list."""
        for solve in self.solves:
            solve.apply()

    def clear(self):
        """Clears the list of solves."""
        self.solves.clear()

    def to_dict(self):
        """
        Returns a dictionary representation of the solve manager.

        Returns:
            dict: A dictionary representation of the solve manager.
        """
        return {
            'solves': [solve.to_dict() for solve in self.solves]
        }

    @classmethod
    def from_dict(cls, optic, data):
        """
        Creates a SolveManager from a dictionary representation.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve manager.

        Returns:
            SolveManager: The solve manager.
        """
        solve_manager = cls(optic)
        for solve_data in data['solves']:
            solve = BaseSolve.from_dict(optic, solve_data)
            solve_manager.solves.append(solve)
        return solve_manager
