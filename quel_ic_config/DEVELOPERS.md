# Developer's Guide

This guide provides essential information for setting up your development environment.

## Setting Up Your Environment with `uv`

We use **`uv`** for dependency management and virtual environment creation.

### Installation and Environment Creation

If you don't have `uv` installed, please follow the instructions on the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
Once `uv` is installed, navigate to the project root directory and run `uv sync` to create a virtual environment and install all necessary dependencies.

```shell
uv sync
```

### Running Commands within the Virtual Environment

You can execute commands or scripts within the virtual environment in two ways.
The recommended way for one-off commands is using `uv run`, which automatically utilizes the project's virtual environment.
For example:

```shell
uv run quel1_linkup <options>
```

Alternatively, you can run `source .venv/bin/activate` to activate the virtual environment for a session.

## Checks and Tests

### Static checking

```shell
./static_check.sh
```

### Test without devices

```shell
make test-without-devices
```

### Executing Tests with Devices

(This part is exclusively for developers at QuEL, inc.)

To execute tests that interacts with hardware devices, use the following command:

```bash
./runtest_std.sh
```

If you wish to **exclude tests that use a spectrum analyzer**, you can add the `-r` option:

```bash
./runtest_std.sh -r
```

## Merge condition

Before merging your changes into the `main` branch, it's mandatory to ensure that all static checks pass and that all tests run with `./runtest_std.sh -r` complete successfully.
