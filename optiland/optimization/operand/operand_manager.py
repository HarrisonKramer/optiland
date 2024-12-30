from optiland.optimization.operand import Operand


class OperandManager:
    """
    Manages a list of operands in a merit function.

    Attributes:
        operands (list): A list of Operand objects.

    Methods:
        add(operand_type, target, weight=1, input_data={}): Add an operand to
            the merit function.
        clear(): Clear all operands from the merit function.
    """

    def __init__(self):
        self.operands = []

    def add(self, operand_type, target, weight=1, input_data={}):
        """Add an operand to the merit function

        Args:
            operand_type (str): The type of the operand.
            target (float): The target value of the operand.
            weight (float): The weight of the operand.
            input_data (dict): Additional input data for the operand.
        """
        self.operands.append(Operand(operand_type, target, weight, input_data))

    def clear(self):
        """Clear all operands from the merit function"""
        self.operands = []

    def __iter__(self):
        """Return the iterator object itself"""
        self._index = 0
        return self

    def __next__(self):
        """Return the next variable in the list"""
        if self._index < len(self.variables):
            result = self.variables[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration