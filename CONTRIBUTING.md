# Contributing to Optiland

Thank you for your interest in contributing to **Optiland**! Contributions are welcome in many forms, including but not limited to:

- Bug reports and feature requests.
- Code contributions and pull requests.
- Improvements to documentation and examples.

## How to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your forked repository locally.
3. **Create** a new branch for your feature or bugfix.
4. **Commit** your changes with clear commit messages.
5. **Push** your changes to your fork.
6. **Open** a pull request with a detailed description of your changes.

## Guidelines

- **Coding Style:** Follow the project's style guidelines. We use automated tools like [`Ruff`](https://docs.astral.sh/ruff/) to enforce code formatting and linting.
- **Testing:** Write tests for new features and bug fixes. Ensure all tests pass before submitting a pull request.
- **Documentation:** Update documentation and examples as necessary.
- **Commit Messages:** Use clear and descriptive commit messages.

## Code Style Guidelines

Please adhere to the following guidelines when contributing.

### General Style Rules

- Follow [`PEP 8`](https://peps.python.org/pep-0008/) for code style.
- Use meaningful variable names that clearly describe their purpose.

### Formatting and Linting

We use [`Ruff`](https://docs.astral.sh/ruff/) for both linting and formatting. Formatting and linting are **automatically enforced** in pull requests through a GitHub Action and must pass before merging.

To ensure compliance before committing, install [`pre-commit`](https://pre-commit.com/) and set up the hook:

```sh
pip install pre-commit
pre-commit install
```

This will manually install the pre-commit hooks from the ``.pre-commit-config.yaml`` file in your local Optiland repository. The pre-commit hooks will automatically run Ruff checks on staged files before committing.

To manually run Ruff checks before committing, use:

```sh
pre-commit run --all-files
```

Ruff can be used to automatically apply fixes for formatting and linting issues where possible. To do this, first install Ruff:

```sh
pip install ruff
```

Then, you can run Ruff to automatically fix issues in your code:

```sh
ruff format .
```

#### Key Formatting Rules:

- Keep line length to a maximum of 88 characters.
- Use spaces instead of tabs for indentation.
- Organize imports as follows:
    1. Standard library imports
    2. Third-party library imports
    3. Local module imports

Example:

```python
import os
import numpy as np
from optiland.analysis import SpotDiagram
```

#### Docstrings and Comments

Write docstrings for all public functions, classes, and modules using the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Use inline comments sparingly and only when necessary to explain complex logic.

## Testing

- Write tests for new features or bug fixes in the tests/ directory.
- Use pytest for running tests:

```sh
pytest
```

- Run tests with coverage before submitting a PR:

```sh
pytest --cov=optiland --cov-report=xml
```

## Reporting Issues

If you encounter any bugs or issues, please report them on our GitHub issue tracker. Include detailed steps to reproduce the issue, along with any relevant logs or error messages.

Thank you for contributing to Optiland!
