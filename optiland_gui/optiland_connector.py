# optiland_gui/optiland_connector.py
from PySide6.QtCore import QObject, Signal

# from optiland.optic import Optic # Assuming Optiland is installed/accessible
# For now, a dummy Optic class to avoid direct dependency for GUI skeleton
class DummyOptic:
    def __init__(self, name="Untitled System"):
        self.name = name
        self.surfaces_data = [
            # Example data: Type, Radius, Thickness, Material, Comment
            ["Object", float('inf'), 10.0, "Air", "Object plane"],
            ["Standard", 50.0, 5.0, "N-BK7", "Lens 1 Front"],
            ["Standard", -50.0, 50.0, "Air", "Lens 1 Back"],
            ["Image", float('inf'), 0.0, "Air", "Image plane"],
        ]
        self.pickups = [] # Placeholder
        self.solves = [] # Placeholder
        print(f"DummyOptic '{self.name}' initialized.")

    def add_surface(self, index=-1, data=None):
        if data is None:
            data = ["Standard", float('inf'), 10.0, "Air", "New Surface"]
        if index == -1 or index >= len(self.surfaces_data):
            self.surfaces_data.append(data)
        else:
            self.surfaces_data.insert(index, data)
        print(f"Surface added: {data}")

    def remove_surface(self, index):
        if 0 <= index < len(self.surfaces_data):
            removed = self.surfaces_data.pop(index)
            print(f"Surface removed: {removed}")
            return True
        return False

    def get_surface_data(self, row, col):
        try:
            return self.surfaces_data[row][col]
        except IndexError:
            return None

    def set_surface_data(self, row, col, value):
        try:
            # Basic validation/conversion
            if col == 1 or col == 2: # Radius, Thickness
                value = float(value)
            self.surfaces_data[row][col] = value
            print(f"Surface data updated at ({row},{col}): {value}")
            return True
        except (IndexError, ValueError) as e:
            print(f"Error setting surface data: {e}")
            return False

    def get_surface_count(self):
        return len(self.surfaces_data)

    def get_column_headers(self):
        return ["Type", "Radius", "Thickness", "Material", "Comment", "Glass"]


class OptilandConnector(QObject):
    opticChanged = Signal()
    surfaceDataChanged = Signal(int, int) # row, col
    surfaceAdded = Signal(int) # index of new surface
    surfaceRemoved = Signal(int) # index of removed surface
    surfaceCountChanged = Signal()

    def __init__(self):
        super().__init__()
        self._optic = DummyOptic("Default System") # Replace with actual Optic()
        # self._optic = Optic("Default System")

    def get_optic(self):
        return self._optic

    def load_optic_from_file(self, filepath):
        # Placeholder for loading Optic from a file (e.g., JSON/YAML)
        # For now, re-initialize a dummy optic
        self._optic = DummyOptic(f"Loaded: {filepath}")
        print(f"Optic loaded from {filepath} (simulated)")
        self.opticChanged.emit()

    def save_optic_to_file(self, filepath):
        # Placeholder for saving Optic to a file
        print(f"Optic saved to {filepath} (simulated)")

    def get_surface_count(self):
        return self._optic.get_surface_count()

    def get_surface_data(self, row, col):
        return self._optic.get_surface_data(row, col)

    def set_surface_data(self, row, col, value):
        if self._optic.set_surface_data(row, col, value):
            self.surfaceDataChanged.emit(row, col)
            self.opticChanged.emit() # General signal that optic has changed

    def add_surface(self, index=-1, data=None):
        # Determine actual insertion index
        if index == -1:
            insert_idx = self.get_surface_count()
        else:
            insert_idx = index

        self._optic.add_surface(index, data)
        self.surfaceAdded.emit(insert_idx)
        self.surfaceCountChanged.emit()
        self.opticChanged.emit()

    def remove_surface(self, index):
        if self._optic.remove_surface(index):
            self.surfaceRemoved.emit(index)
            self.surfaceCountChanged.emit()
            self.opticChanged.emit()

    def get_column_headers(self):
        return self._optic.get_column_headers()