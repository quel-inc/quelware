import logging
from pathlib import Path
from typing import Collection, Dict, Sequence, Tuple, Union

from quel_ic_config.e7resource_mapper import AbstractQuel1E7ResourceMapper, validate_configuration_integrity
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import create_css_wss_rmap
from quel_ic_config.quel1_wave_subsystem import Quel1WaveSubsystem
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption

logger = logging.getLogger(__name__)


def init_box_with_linkup(
    *,
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfes_to_linkup: Sequence[int],
    config_root: Union[Path, None],
    config_options: Collection[Quel1ConfigOption],
    use_204b: bool = True,
    hard_reset: bool = False,
    skip_init: bool = False,
    background_noise_threshold: Union[float, None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
    ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ignore_extraordinal_converter_select_of_mxfe: Union[Collection[int], None] = None,
) -> Tuple[
    Dict[int, bool],
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    AbstractQuel1E7ResourceMapper,
    LinkupFpgaMxfe,
]:
    """create QuEL testing objects, reset all the ICs, and establish datalink.

    :param ipaddr_wss: IP address of the wave generation subsystem of the target box
    :param ipaddr_sss: IP address of the sequencer subsystem of the target box
    :param ipaddr_css: IP address of the configuration subsystem of the target box
    :param boxtype: type of the target box
    :param mxfes_to_linkup: target mxfes of the target box
    :param config_root: root path of config setting files to read
    :param config_options: a collection of config options
    :param use_204b: choose JESD204B link or 204C one
    :param refer_by_port: return an object which takes port index for specifying input and output site if True.
    :return: QuEL testing objects
    """

    css, wss, rmap, linkupper = create_objects_in_box(
        ipaddr_wss=ipaddr_wss,
        ipaddr_sss=ipaddr_sss,
        ipaddr_css=ipaddr_css,
        boxtype=boxtype,
        config_root=config_root,
        config_options=config_options,
    )

    linkup_ok = linkup_dev(
        linkupper=linkupper,
        mxfe_list=mxfes_to_linkup,
        hard_reset=hard_reset,
        use_204b=use_204b,
        skip_init=skip_init,
        background_noise_threshold=background_noise_threshold,
        ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
        ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
        ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinal_converter_select_of_mxfe,
    )

    return linkup_ok, css, wss, rmap, linkupper


def init_box_with_reconnect(
    *,
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfes_to_connect: Union[Sequence[int], None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_extraordinal_converter_select_of_mxfe: Union[Collection[int], None] = None,
) -> Tuple[
    Dict[int, bool],
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    AbstractQuel1E7ResourceMapper,
    LinkupFpgaMxfe,
]:
    css, wss, rmap, linkupper = create_objects_in_box(
        ipaddr_wss=ipaddr_wss,
        ipaddr_sss=ipaddr_sss,
        ipaddr_css=ipaddr_css,
        boxtype=boxtype,
        config_root=None,
        config_options={},
    )

    if mxfes_to_connect is None:
        mxfes_to_connect = list(css.get_all_groups())
        mxfes_to_connect.sort()

    link_ok = reconnect_dev(
        css=css,
        wss=wss,
        rmap=rmap,
        mxfe_list=mxfes_to_connect,
        ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
        ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinal_converter_select_of_mxfe,
    )
    return link_ok, css, wss, rmap, linkupper


def create_objects_in_box(
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    config_root: Union[Path, None] = None,
    config_options: Union[Collection[Quel1ConfigOption], None] = None,
    skip_init: bool = False,
) -> Tuple[
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    AbstractQuel1E7ResourceMapper,
    LinkupFpgaMxfe,
]:
    """create QuEL config objects

    :param ipaddr_wss: IP address of the wave generation subsystem of the target box
    :param ipaddr_sss: IP address of the sequencer subsystem of the target box
    :param ipaddr_css: IP address of the configuration subsystem of the target box
    :param boxtype: type of the target box
    :param config_root: root path of config setting files to read
    :param config_options: a collection of config options
    :return: QuEL config objects
    """
    css, wss, rmap = create_css_wss_rmap(
        ipaddr_wss=ipaddr_wss,
        ipaddr_sss=ipaddr_sss,
        ipaddr_css=ipaddr_css,
        boxtype=boxtype,
        config_root=config_root,
        config_options=config_options,
    )
    linkupper = LinkupFpgaMxfe(css, wss, rmap)
    if not skip_init:
        wss.initialize()

    return css, wss, rmap, linkupper


def linkup_dev(
    *,
    linkupper: LinkupFpgaMxfe,
    mxfe_list: Sequence[int],
    hard_reset: bool = False,
    use_204b: bool = True,
    use_bg_cal: bool = False,
    skip_init: bool = False,
    background_noise_threshold: Union[float, None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
    ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
    save_dirpath: Union[Path, None] = None,
) -> Dict[int, bool]:
    if ignore_crc_error_of_mxfe is None:
        ignore_crc_error_of_mxfe = {}

    if ignore_access_failure_of_adrf6780 is None:
        ignore_access_failure_of_adrf6780 = {}

    if ignore_lock_failure_of_lmx2594 is None:
        ignore_lock_failure_of_lmx2594 = {}

    if ignore_extraordinary_converter_select_of_mxfe is None:
        ignore_extraordinary_converter_select_of_mxfe = {}

    if not skip_init:
        linkupper._css.configure_peripherals(
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        linkupper._css.configure_all_mxfe_clocks(
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )

    linkup_ok: Dict[int, bool] = {}
    for mxfe in mxfe_list:
        linkup_ok[mxfe] = linkupper.linkup_and_check(
            mxfe,
            hard_reset=hard_reset,
            use_204b=use_204b,
            use_bg_cal=use_bg_cal,
            background_noise_threshold=background_noise_threshold,
            ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select=mxfe in ignore_extraordinary_converter_select_of_mxfe,
            save_dirpath=save_dirpath,
        )

    return linkup_ok


def reconnect_dev(
    *,
    css: Quel1AnyConfigSubsystem,
    wss: Quel1WaveSubsystem,
    rmap: AbstractQuel1E7ResourceMapper,
    mxfe_list: Sequence[int],
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
) -> Dict[int, bool]:
    if ignore_crc_error_of_mxfe is None:
        ignore_crc_error_of_mxfe = {}

    if ignore_extraordinary_converter_select_of_mxfe is None:
        ignore_extraordinary_converter_select_of_mxfe = {}

    if mxfe_list is None:
        mxfe_list = list(css.get_all_groups())
        mxfe_list.sort()
    link_ok: Dict[int, bool] = {}
    for mxfe_idx in mxfe_list:
        try:
            link_ok[mxfe_idx] = css.configure_mxfe(mxfe_idx, ignore_crc_error=mxfe_idx in ignore_crc_error_of_mxfe)
            if not link_ok[mxfe_idx]:
                logger.error(f"AD9082-#{mxfe_idx} is not working, check power and link status before retrying")
            else:
                css.validate_chip_id(mxfe_idx)

            validate_configuration_integrity(
                css.get_virtual_adc_select(mxfe_idx),
                wss.fw_type,
                ignore_extraordinary_converter_select=mxfe_idx in ignore_extraordinary_converter_select_of_mxfe,
            )

        except RuntimeError:
            logger.error(f"failed to establish a configuration link with AD9082-#{mxfe_idx}")
            link_ok[mxfe_idx] = False
    return link_ok
