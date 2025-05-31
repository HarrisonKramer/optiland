"""Configuration for plotting styles."""


class PlotConfig:
    """
    Manages plotting themes and provides access to style configurations.
    """

    def __init__(self, initial_theme="light"):
        """
        Initializes the PlotConfig with a specified theme.

        Args:
            initial_theme (str, optional): The initial theme to use.
                                           Defaults to "light".
        """
        self.themes = {
            "light": {
                "background_color": "#FFFFFF",
                "axis_color": "#000000",
                "grid_color": "#E0E0E0",
                "line_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                "patch_colors": {
                    "lens": {"facecolor": "#D0D0F0", "edgecolor": "#505060"},
                    "default": {"facecolor": "#CCCCCC", "edgecolor": "#333333"},
                },
                "specific_line_colors": {
                    "surface": "#808080"  # Gray
                },
                "text_color": "#000000",
            },
            "dark": {
                "background_color": "#121212",
                "axis_color": "#E0E0E0",
                "grid_color": "#444444",
                "line_colors": ["#569fdd", "#ffab57", "#63c763", "#e86465", "#ad82d0"],
                "patch_colors": {
                    "lens": {"facecolor": "#404070", "edgecolor": "#A0A0B0"},
                    "default": {"facecolor": "#555555", "edgecolor": "#CCCCCC"},
                },
                "specific_line_colors": {
                    "surface": "#A0A0A0"  # Light Gray
                },
                "text_color": "#E0E0E0",
            },
        }
        if initial_theme not in self.themes:
            raise ValueError(f"Theme '{initial_theme}' not found.")
        self.current_theme = initial_theme

    def set_theme(self, theme_name):
        """
        Sets the current plotting theme.

        Args:
            theme_name (str): The name of the theme to set.

        Raises:
            ValueError: If the theme_name is not a valid theme.
        """
        if theme_name not in self.themes:
            raise ValueError(
                f"Theme '{theme_name}' not found. "
                f"Available themes are: {list(self.themes.keys())}"
            )
        self.current_theme = theme_name

    def get_style(self, element_key):
        """
        Retrieves a style value from the current theme based on an element key.

        The element_key can be a simple string or a dot-separated string
        for nested dictionary or list access.

        Args:
            element_key (str): The key for the desired style element.
                               Examples: "background_color", "line_colors.0",
                               "patch_colors.lens.facecolor".

        Returns:
            The style value associated with the element_key.

        Raises:
            KeyError: If the element_key (or parts of it) is not found in the theme.
            IndexError: If an index in the element_key is out of bounds for a list.
            ValueError: If an index in the element_key is not a valid integer.
        """
        keys = element_key.split(".")
        current_value = self.themes[self.current_theme]

        for key_part in keys:
            if isinstance(current_value, dict):
                if key_part not in current_value:
                    raise KeyError(
                        f"Key '{key_part}' not found in {''.join(keys[: keys.index(key_part)]) if keys.index(key_part) > 0 else self.current_theme}' for key '{element_key}'"
                    )
                current_value = current_value[key_part]
            elif isinstance(current_value, list):
                try:
                    index = int(key_part)
                    current_value = current_value[index]
                except ValueError:
                    raise ValueError(
                        f"Invalid index '{key_part}' in key '{element_key}'. Index must be an integer."
                    )
                except IndexError:
                    raise IndexError(
                        f"Index {index} out of bounds for list in key '{element_key}'"
                    )
            else:
                # This case occurs if a key tries to go deeper into a non-dict/list value
                raise KeyError(
                    f"Cannot access sub-key '{key_part}' because "
                    f"{''.join(keys[: keys.index(key_part)])}' is not a dictionary or list "
                    f"for key '{element_key}'"
                )
        return current_value


if __name__ == "__main__":
    # Example Usage
    config = PlotConfig()

    print(f"Current theme: {config.current_theme}")
    print(f"Background color: {config.get_style('background_color')}")
    print(f"First line color: {config.get_style('line_colors.0')}")
    print(f"Lens face color: {config.get_style('patch_colors.lens.facecolor')}")

    config.set_theme("dark")
    print(f"\nCurrent theme: {config.current_theme}")
    print(f"Background color: {config.get_style('background_color')}")
    print(f"First line color: {config.get_style('line_colors.0')}")
    print(f"Lens face color: {config.get_style('patch_colors.lens.facecolor')}")

    try:
        config.set_theme("non_existent_theme")
    except ValueError as e:
        print(f"\nError setting theme: {e}")

    try:
        print(f"\nNon-existent key: {config.get_style('non_existent_key')}")
    except KeyError as e:
        print(f"Error getting style: {e}")

    try:
        print(f"\nInvalid index: {config.get_style('line_colors.invalid')}")
    except ValueError as e:
        print(f"Error getting style: {e}")

    try:
        print(f"\nOut of bounds index: {config.get_style('line_colors.99')}")
    except IndexError as e:
        print(f"Error getting style: {e}")

    try:
        print(
            f"\nAccessing sub-key of non-dict: {config.get_style('background_color.sub')}"
        )
    except KeyError as e:
        print(f"Error getting style: {e}")

    # Test initial theme validation
    try:
        error_config = PlotConfig(initial_theme="invalid_start_theme")
    except ValueError as e:
        print(f"\nError during initialization: {e}")

    # Test patch_colors.default.edgecolor
    config.set_theme("light")
    print(
        f"\nDefault patch edge color (light): {config.get_style('patch_colors.default.edgecolor')}"
    )
    config.set_theme("dark")
    print(
        f"Default patch edge color (dark): {config.get_style('patch_colors.default.edgecolor')}"
    )

    # Test new specific line colors
    config.set_theme("light")
    print(
        f"\nSurface line color (light): {config.get_style('specific_line_colors.surface')}"
    )
    config.set_theme("dark")
    print(
        f"Surface line color (dark): {config.get_style('specific_line_colors.surface')}"
    )

    try:
        print(
            f"Non-existent specific color: {config.get_style('specific_line_colors.non_existent')}"
        )
    except KeyError as e:
        print(f"Error for non-existent specific color: {e}")
