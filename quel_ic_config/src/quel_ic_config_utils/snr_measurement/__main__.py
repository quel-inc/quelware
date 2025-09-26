import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from typing_extensions import Annotated

import quel_ic_config as qi
from quel_ic_config_utils import configuration

from ._snr_measurement import measure_snr

logger = logging.getLogger(__name__)
err_console = Console(stderr=True)

app = typer.Typer(add_completion=False, rich_markup_mode=None)


@app.command()
def entrypoint(
    source_box_name: Annotated[str, typer.Argument(help="Name of the Box for wave generation")],
    source_port: Annotated[str, typer.Argument(help="Port for wave generation")],
    dest_box_name: Annotated[str, typer.Argument(help="Name of the Box for signal capture")],
    dest_port: Annotated[str, typer.Argument(help="Port for signal capture")],
    sysconf_path: Annotated[
        Optional[Path],
        typer.Option("--sysconf", exists=True, file_okay=True, dir_okay=False, help="System configuration file (yaml)"),
    ] = None,
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose output."),
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
    """
    Measure the Signal-to-Noise Ratio using a random noise signal and show the result in dB.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format="{asctime} [{levelname:.4}] {name}: {message}", style="{"
    )

    if sysconf_path:
        with open(sysconf_path) as fp:
            sysconf_object = yaml.safe_load(fp)
        sysconf = configuration.SystemConfiguration.model_validate(sysconf_object)
    else:
        sysconf = configuration.load_default_configuration()

    if sysconf is None:
        err_console.print("Could not find the configuration file.")
        raise typer.Exit(1)

    for required_box_name in [source_box_name, dest_box_name]:
        if required_box_name not in [b.name for b in sysconf.boxes]:
            err_console.print(f"'{required_box_name}' is not found in the system configuration.")
            raise typer.Exit(1)

    name_to_box = {
        box.name: box
        for box in configuration.get_boxes_in_parallel(
            b for b in sysconf.boxes if b.name in [source_box_name, dest_box_name]
        )
    }

    name_to_status = configuration.reconnect_and_get_link_status_in_parallel(name_to_box.values())
    if not all(all(status.values()) for status in name_to_status.values()):
        err_console.print(f"Error occured during reconnecting. status={name_to_status}")
        raise typer.Exit(1)

    _trigger_port = qi.parse_port_str(source_port)
    if not name_to_box[source_box_name].is_valid_port(_trigger_port):
        err_console.print(f"Invalid port: '{_trigger_port}'")
        raise typer.Exit(1)

    _dest_port = qi.parse_port_str(dest_port)
    if not name_to_box[dest_box_name].is_valid_port(_dest_port):
        err_console.print(f"Invalid port: '{_dest_port}'")
        raise typer.Exit(1)

    snr = measure_snr(name_to_box[source_box_name], _trigger_port, name_to_box[dest_box_name], _dest_port)
    if verbose:
        print(f"SNR: {snr} dB")
    else:
        print(f"{snr}")


app()
