import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from typing_extensions import Annotated

from quel_ic_config_utils import configuration

logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
logger = logging.getLogger()

err_console = Console(stderr=True)

app = typer.Typer(add_completion=False, rich_markup_mode=None)


def _edit_or_abort(original_content: str, suffix: str = ".tmp") -> Optional[str]:
    """
    Opens the user's default editor to modify content.

    Args:
        original_content (str): The initial content to be edited.
        suffix (str): The filename suffix for the temporary file.

    Returns:
        The modified content as a string, or None if no changes were made.
    """
    editor = os.environ.get("EDITOR", "vim")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(original_content)
        tmp_file_path = tmp_file.name

    try:
        subprocess.run([editor, tmp_file_path], check=True)

        with open(tmp_file_path, "r") as edited_file:
            edited_content = edited_file.read()

        if edited_content.strip() != original_content.strip():
            return edited_content
        else:
            return None

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Editor '{editor}' not found.") from e
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def _reconnect_to_box(sysconf_path: Optional[Path], box_name):
    if sysconf_path:
        with open(sysconf_path) as fp:
            conf_object = yaml.safe_load(fp)
        conf = configuration.SystemConfiguration.model_validate(conf_object)
    else:
        conf = configuration.load_default_configuration()

    if conf is None:
        err_console.print("ERROR: Could not find the configuration file.")
        raise typer.Exit(1)

    boxes = list(configuration.get_boxes(b for b in conf.boxes if b.name == box_name))
    if len(boxes) == 0:
        err_console.print(f"ERROR: box '{box_name}' is not found.")
        raise typer.Exit(1)
    box = boxes[0]
    configuration.reconnect_and_get_link_status(box)
    return box


@app.command()
def show(
    box_name: Annotated[str, typer.Argument(help="Name of the Box")],
    sysconf_path: Annotated[
        Optional[Path],
        typer.Option("--sysconf", exists=True, file_okay=True, dir_okay=False, help="System configuration file (yaml)"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "WARNING",
):
    logger.setLevel(getattr(logging, log_level.upper()))

    box = _reconnect_to_box(sysconf_path, box_name)

    config_obj = json.loads(box.dump_box_to_jsonstr())
    formatted_json_string = json.dumps(config_obj, indent=2)

    print(formatted_json_string)


@app.command()
def edit(
    box_name: Annotated[str, typer.Argument(help="Name of the Box")],
    sysconf_path: Annotated[
        Optional[Path],
        typer.Option("--sysconf", exists=True, file_okay=True, dir_okay=False, help="System configuration file (yaml)"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "WARNING",
):
    logger.setLevel(getattr(logging, log_level.upper()))

    box = _reconnect_to_box(sysconf_path, box_name)

    config_obj = json.loads(box.dump_box_to_jsonstr())
    formatted_json_string = json.dumps(config_obj, indent=2)

    del box  # to release lock during editor

    if (modified_str := _edit_or_abort(formatted_json_string)) is None:
        print("Aborted.")
    else:
        try:
            box = _reconnect_to_box(sysconf_path, box_name)
            config_obj = json.loads(box.dump_box_to_jsonstr())
            if formatted_json_string != json.dumps(config_obj, indent=2):
                print("Configuration was changed from another location.")
                print("Aborted.")
                raise typer.Exit(1)

            # to validate json format
            json.loads(modified_str)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print("Aborted.")
            raise typer.Exit(1)

        box.config_box_from_jsonstr(modified_str)
        print("Configuration has been updated successfully.")


@app.command()
def overwrite(
    box_name: Annotated[str, typer.Argument(help="Name of the Box")],
    file: Annotated[typer.FileText, typer.Argument(help="Path to JSON file ('-' for stdin)")],
    sysconf_path: Annotated[
        Optional[Path],
        typer.Option("--sysconf", exists=True, file_okay=True, dir_okay=False, help="System configuration file (yaml)"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "WARNING",
):
    logger.setLevel(getattr(logging, log_level.upper()))

    box = _reconnect_to_box(sysconf_path, box_name)

    input_content = file.read()
    try:
        # to validate json format
        json.loads(input_content)
    except json.JSONDecodeError as e:
        err_console.print(f"ERROR: Failed to decode JSON: {e}")
        print("Aborted.")
        raise typer.Exit(1)

    box.config_box_from_jsonstr(input_content)
    print("Configuration has been updated successfully.")


def main():
    app()


if __name__ == "__main__":
    main()
