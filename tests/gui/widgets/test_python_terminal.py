import pytest
from PySide6.QtGui import QTextDocument, QColor
from optiland_gui.widgets.python_terminal import PythonHighlighter

@pytest.fixture
def highlighter_and_doc(qapp):
    """Fixture for a PythonHighlighter instance and its QTextDocument."""
    doc = QTextDocument()
    theme_colors = {
        "keyword": QColor("blue"),
        "string": QColor("green"),
        "comment": QColor("red"),
        "number": QColor("purple"),
    }
    highlighter = PythonHighlighter(doc, theme_colors)
    return highlighter, doc

def test_python_highlighter(highlighter_and_doc):
    """
    Test that the PythonHighlighter correctly highlights Python syntax.
    """
    highlighter, doc = highlighter_and_doc
    # Test that the highlightBlock method can be called without raising an exception
    highlighter.highlightBlock("def my_func():")
