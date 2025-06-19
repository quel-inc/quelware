# Developer's Guide

This guide provides essential information for setting up your development environment.

## Setting Up Your Environment with `uv`

We use **`uv`** for dependency management and virtual environment creation.

### Installation and Environment Creation

If you don't have `uv` installed, please follow the instructions on the [official uv documentation](https://www.google.com/search?q=https://astral.sh/uv/install/).
Once `uv` is installed, navigate to the project root directory and run `uv sync` to create a virtual environment and install all necessary dependencies.

```bash
uv sync
```

### Running Commands within the Virtual Environment

You can execute commands or scripts within the virtual environment in two ways.
The recommended way for one-off commands is using `uv run`, which automatically utilizes the project's virtual environment.

Alternatively, you can run `source .venv/bin/activate` to activate the virtual environment for a session.

## Checks and Tests

```bash
./static_check.sh
```

```bash
make test-without-device
```
