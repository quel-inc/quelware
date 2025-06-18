import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, Union, cast

import numpy as np
import numpy.typing as npt
from e7awghal import AwgParam, CapIqDataReader, CapParam, CapSection, WaveChunk

from quel_ic_config import (
    AbstractStartAwgunitsTask,
    BoxStartCapunitsByTriggerTask,
    Quel1Box,
    Quel1PortType,
    QuelClockMasterV1,
)

logger = logging.getLogger(__name__)

BoxSettingType = Mapping[str, Any]
VportTypicalSettingType = Mapping[str, Any]
VportSimpleParamtersSettingType = Mapping[int, Mapping[str, Any]]
VportSettingType = Union[VportTypicalSettingType, VportSimpleParamtersSettingType]


class BoxPool:
    SYSREF_PERIOD: Final[int] = 2000
    # TODO: tried to find the best value, but the best value changes link-up by link-up. so, calibration is required.
    TIMING_OFFSET: Final[int] = 0
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(
        self,
        cm_settings: Mapping[str, Any],
        box_settings: Mapping[str, Mapping[str, BoxSettingType]],
        cap_settings: Mapping[str, Mapping[str, VportSettingType]],
        gen_settings: Mapping[str, Mapping[str, VportSettingType]],
    ):
        self._cm_settings: Mapping[str, Any] = cm_settings
        self._box_settings: Mapping[str, Mapping[str, Any]] = box_settings
        self._cap_settings: Mapping[str, Mapping[str, VportSettingType]] = cap_settings
        self._gen_settings: Mapping[str, Mapping[str, VportSettingType]] = gen_settings

        self._clock_master: Union[QuelClockMasterV1, None] = None

        self._boxes: dict[str, Quel1Box] = {}
        self._channels: dict[str, tuple[str, Quel1PortType, int]] = {}  # Notes: genname -> (boxname, port, channel)
        self._runits: dict[str, tuple[str, Quel1PortType, set[int]]] = {}  # Notes: capname -> (boxname, port, runits)
        self._linkstatus: dict[str, bool] = {}

        self._estimated_timediff: dict[str, int] = {boxname: 0 for boxname in self._boxes}
        self._cap_sysref_time_offset: int = 0

    def initialize(
        self, recreate_box=False, reconnect: bool = True, config_css: bool = True, allow_resync: bool = True
    ):
        if recreate_box or len(self._boxes) == 0:
            self.create_box(self._box_settings)
        if len(self._cm_settings) > 0:
            self._clock_master = QuelClockMasterV1(**self._cm_settings)
        self.scan_link_status(reconnect=reconnect)
        self.reset_wss()
        self.config_channels(config_css, True)
        self.config_runits(config_css, True)
        sync_status = self.check_synchronization()
        if not sync_status:
            if allow_resync:
                self.resync()
                sync_status = self.check_synchronization()
        if not sync_status:
            raise RuntimeError("synchronization error")

    def create_box(self, settings: Mapping[str, Mapping[str, Any]]):
        for k, v in settings.items():
            self._boxes[k] = Quel1Box.create(**v)
            self._linkstatus[k] = False

    def check_synchronization(self) -> bool:
        if len(self._boxes) < 2:
            return True

        fsync_ok = True
        sysref_counters = {name: box.get_averaged_sysref_offset() for name, box in self._boxes.items()}
        diff = max(sysref_counters.values()) - min(sysref_counters.values())
        if diff > 1000:
            diff = 2000 - diff  # Notes: abs(diff - 2000)
        fsync_ok &= diff < 100
        logger.info(f"synchronization status: {fsync_ok}, max_delta = {diff}")
        return fsync_ok

    def scan_link_status(self, reconnect=False):
        for name, box in self._boxes.items():
            link_status: bool = True
            if reconnect:
                if not all(box.reconnect().values()):
                    if all(box.reconnect(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()):
                        logger.warning(f"crc error has been detected on MxFEs of {name}")
                    else:
                        logger.error(f"datalink between MxFE and FPGA of {name} is not working")
                        link_status = False
            else:
                if not all(box.link_status().values()):
                    if all(box.link_status(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()):
                        logger.warning(f"crc error has been detected on MxFEs of {name}")
                    else:
                        logger.error(f"datalink between MxFE and FPGA of {name} is not working")
                        link_status = False
            self._linkstatus[name] = link_status

    def reset_wss(self):
        for name, box in self._boxes.items():
            box.initialize_all_awgunits()
            box.initialize_all_capunits()

    def config_channels(self, config_css: bool, config_wss: bool):
        for k, v in self._gen_settings.items():
            if "create" not in v:
                raise ValueError("no 'create' parameters for wavegen '{k}'")

            c = cast(VportTypicalSettingType, v["create"])
            boxname: str = c["boxname"]
            port: Quel1PortType = c["port"]
            channel: int = c["channel"]
            self._channels[k] = (boxname, port, channel)

            if config_css and "config" in v:
                self._config_channel(k, **cast(VportTypicalSettingType, v["config"]))
            if config_wss and "cw_parameter" in v:
                self._load_awgparam_cw(k, **cast(VportTypicalSettingType, v["cw_parameter"]))

    def config_runits(self, config_css: bool, config_wss: bool):
        for k, v in self._cap_settings.items():
            if "create" not in v:
                raise ValueError("no 'create' parameters for wavegen '{k}'")

            cr = cast(VportTypicalSettingType, v["create"])
            boxname: str = cr["boxname"]
            port: Quel1PortType = cr["port"]
            runits: set[int] = set(cr["runits"])
            self._runits[k] = (boxname, port, runits)

            if config_css and "config" in v:
                self._config_runits(k, **cast(VportTypicalSettingType, v["config"]))
            if config_wss and "simple_parameters" in v:
                self._load_capparam_simple(
                    k,
                    cast(
                        Mapping[int, Mapping[str, Any]], cast(VportSimpleParamtersSettingType, v["simple_parameters"])
                    ),
                )

    def _config_channel(self, gname: str, *, fnco_freq: Union[float, None] = None, **kwargs) -> None:
        logger.info(f"reconfigureing {gname}")
        boxname, port, channel = self._channels[gname]
        box = self._boxes[boxname]

        box.config_port(port=port, **kwargs)
        if fnco_freq is not None:
            box.config_channel(port=port, channel=channel, fnco_freq=fnco_freq)

    def _load_awgparam_cw(
        self,
        gname: str,
        amplitude: float,
        num_wave_sample: int,
        num_repeats: tuple[int, int],
        num_wait_samples: tuple[int, int],
        null_chunk_sample: Union[int, None] = None,
    ) -> None:
        boxname, port, channel = self._channels[gname]
        box = self._boxes[boxname]

        if num_wave_sample % 64 != 0:
            raise ValueError("num_wave_sample must be multiples of 64")
        if num_wait_samples[0] % 16 != 0:
            raise ValueError("both num_wait_samples[0] must be multiples of 16")
        if num_wait_samples[1] % 4 != 0:
            raise ValueError("both num_wait_samples must be multiples of 4")
        if null_chunk_sample is not None and null_chunk_sample < 64:
            raise ValueError("null_chunk_sample must not be less than 64")
        if null_chunk_sample is not None and null_chunk_sample % 4 != 0:
            raise ValueError("null_chunk_sample must be multiples of 4")

        cw_iq = np.zeros(num_wave_sample, dtype=np.complex64)
        cw_iq[:] = amplitude * (1.0 + 0.0j)
        box.register_wavedata(port, channel, "glc_cw", cw_iq)

        ap = AwgParam(num_wait_word=num_wait_samples[0] // 4, num_repeat=num_repeats[0])
        if null_chunk_sample is not None:
            ap.chunks.append(WaveChunk(name_of_wavedata="null", num_blank_word=(null_chunk_sample - 64) // 4))
        ap.chunks.append(WaveChunk(name_of_wavedata="glc_cw", num_blank_word=num_wait_samples[1] // 4))
        box.config_channel(port, channel, awg_param=ap)

    def _config_runits(self, cname: str, *, fnco_freq: Union[Mapping[int, float], float, None] = None, **kwargs):
        logger.info(f"reconfigureing {cname}")
        boxname, port, runits = self._runits[cname]
        box = self._boxes[boxname]

        box.config_port(port=port, **kwargs)
        if isinstance(fnco_freq, Mapping):
            for runit in runits:
                box.config_runit(port=port, runit=runit, fnco_freq=fnco_freq[runit])
        elif fnco_freq is not None:
            for runit in runits:
                box.config_runit(port=port, runit=runit, fnco_freq=fnco_freq)
        else:
            raise TypeError(f"malformed fnco_freq: {fnco_freq}")

    @staticmethod
    def _make_cwparam_simple(
        *,
        num_delay_sample: int,
        num_integration_section: int,
        num_capture_samples: Sequence[int],
        num_blank_samples: Sequence[int],
    ) -> CapParam:
        if num_delay_sample % 4 != 0:
            raise ValueError(f"num_delay_sample (= {num_delay_sample} is not multiple of 4.")

        cp = CapParam(num_wait_word=num_delay_sample // 4, num_repeat=num_integration_section)
        for idx in range(len(num_capture_samples)):
            if num_capture_samples[idx] % 4 != 0:
                raise ValueError(f"num_capture_samples[{idx}] (= {num_capture_samples[idx]}) is not multiple of 4")
            if num_blank_samples[idx] % 4 != 0:
                raise ValueError(f"num_blank_samples[{idx}] (= {num_blank_samples[idx]}) is not multiple of 4")
            cp.sections.append(
                CapSection(
                    name="s0",
                    num_capture_word=num_capture_samples[idx] // 4,
                    num_blank_word=num_blank_samples[idx] // 4,
                )
            )
        return cp

    def _load_capparam_simple(self, cname: str, simple_parameters: VportSimpleParamtersSettingType) -> None:
        boxname, port, runits = self._runits[cname]
        box = self._boxes[boxname]

        for runit, v in simple_parameters.items():
            if runit not in runits:
                raise ValueError(f"an invalid runit: {runit}")
            box.config_runit(port=port, runit=runit, capture_param=self._make_cwparam_simple(**v))

    def resync(self):
        if self._clock_master:
            self._clock_master.sync_boxes()

    def get_boxes(self) -> dict[str, Quel1Box]:
        return dict(self._boxes)

    def get_box(self, name: str) -> Quel1Box:
        if name in self._boxes:
            box = self._boxes[name]
            return box
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_linkstatus(self, name: str) -> bool:
        if name in self._boxes:
            return self._linkstatus[name]
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def _mod_by_sysref(self, t: int) -> int:
        h = self.SYSREF_PERIOD // 2
        return (t + h) % self.SYSREF_PERIOD - h
        # return t % self.SYSREF_PERIOD

    def check_background_noise(self, runit_names: set[str]) -> dict[str, float]:
        bgnoise: dict[str, float] = {}

        runits_by_box: dict[str, set[tuple[Quel1PortType, int]]] = {}
        for ru in runit_names:
            boxname, port, runits = self._runits[ru]
            if boxname not in runits_by_box:
                runits_by_box[boxname] = set()
            for runit in runits:
                runits_by_box[boxname].add((port, runit))

        for boxname, fqrunits in runits_by_box.items():
            ctask = self._boxes[boxname].start_capture_now(fqrunits)
            rdr = ctask.result()
            for fqrunit in fqrunits:
                bg = np.abs(rdr[fqrunit].as_wave_dict()["s0"][0])
                bgavg = np.mean(bg)
                bgmax = np.max(bg)
                logger.info(
                    f"background noise of port-#{fqrunit[0]:02d} of {boxname}: {bgavg:.1f} (mean), {bgmax:.1f} (max)"
                )

                # TODO: revise this!
                for rn, (bn, p, us) in self._runits.items():
                    for u in us:
                        if bn == boxname and p == fqrunit[0] and u == fqrunit[1]:
                            bgnoise[rn] = bgmax
        return bgnoise

    def measure_timediff(self, cname: str, num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS) -> None:
        counter_at_sysref_clk: dict[str, int] = {boxname: 0 for boxname in self._boxes}

        for i in range(num_iters):
            for boxname, box in self._boxes.items():
                last_sysref_time = box.get_latest_sysref_timecounter()
                counter_at_sysref_clk[boxname] += self._mod_by_sysref(last_sysref_time)

        avg: dict[str, int] = {boxname: round(cntr / num_iters) for boxname, cntr in counter_at_sysref_clk.items()}
        # Notes: averaged timecounter of capturing box since self._runits[cname]: tuple[str, int, set[int]]
        self._cap_sysref_time_offset = avg[self._runits[cname][0]]
        self._estimated_timediff = {boxname: cntr - self._cap_sysref_time_offset for boxname, cntr in avg.items()}
        logger.info(f"estimated time difference: {self._estimated_timediff}")

    def start_at(
        self,
        runit_name: str,
        channel_names: Collection[str],
        min_time_offset: int,
        displacement: int = 0,
    ) -> tuple[int, dict[str, BoxStartCapunitsByTriggerTask], dict[str, AbstractStartAwgunitsTask]]:
        if len(channel_names) == 0:
            logger.warning("no pulse generator to activate")

        capture_tasks: dict[str, BoxStartCapunitsByTriggerTask] = {}
        wavegen_tasks: dict[str, AbstractStartAwgunitsTask] = {}

        channels_by_box: dict[str, set[tuple[Quel1PortType, int]]] = {}
        for ch in channel_names:
            boxname, port, channel = self._channels[ch]
            if boxname not in channels_by_box:
                channels_by_box[boxname] = set()
            channels_by_box[boxname].add((port, channel))

        runits_by_box: dict[str, set[tuple[Quel1PortType, int]]] = {}
        for ru in (runit_name,):  # Notes: for future extention
            boxname, port, runits = self._runits[ru]
            if boxname not in channels_by_box:
                raise RuntimeError(f"impossible to trigger the runit of box '{boxname}'")
            if boxname not in runits_by_box:
                runits_by_box[boxname] = set()
            for runit in runits:
                runits_by_box[boxname].add((port, runit))

        # Notes: showing timecounters of each box (just for information)
        for boxname, chs in channels_by_box.items():
            box = self._boxes[boxname]
            if len(chs) == 0:
                continue
            ct = box.get_current_timecounter()
            lst = box.get_latest_sysref_timecounter()
            logger.info(
                f"boxname: {boxname}, current time: {ct}, "
                f"sysref offset: {self._mod_by_sysref(lst)}, "
                f"have_capture: {boxname in runits_by_box}"
            )

        cpbox0 = list(runits_by_box.keys())[0]
        current_time = self._boxes[cpbox0].get_current_timecounter()  # Notes: monitoring box provides standard time.
        last_sysref_time = self._boxes[cpbox0].get_latest_sysref_timecounter()

        # Notes: checking the fluctuation of sysref trigger (just for information).
        fluctuation = self._mod_by_sysref(last_sysref_time - self._cap_sysref_time_offset)
        if abs(fluctuation) > 4:
            logger.warning(
                f"large fluctuation (= {fluctuation}) of sysref is detected from the previous timing measurement"
            )

        # Notes: time adjustment among the boxes
        base_time = current_time + min_time_offset
        offset = (16 - (base_time - self._cap_sysref_time_offset) % 16) % 16
        base_time += offset
        base_time += displacement  # inducing clock displacement for performance evaluation (must be 0 usually).
        base_time += self.TIMING_OFFSET  # Notes: the safest timing to issue trigger, at the middle of two AWG block.

        # Notes: scheduling!
        for boxname in channels_by_box:
            ts = base_time + self._estimated_timediff[boxname]
            if boxname in runits_by_box:
                ctask, gtask = self._boxes[boxname].start_capture_by_awg_trigger(
                    runits_by_box[boxname], channels_by_box[boxname], timecounter=ts
                )
                capture_tasks[boxname] = ctask
                wavegen_tasks[boxname] = gtask
                logger.info(f"reserving capture and wavegen of {boxname} at {ts}")
            else:
                gtask = self._boxes[boxname].start_wavegen(channels_by_box[boxname], timecounter=ts)
                wavegen_tasks[boxname] = gtask
                logger.info(f"reserving wavegen of {boxname} at {ts}")
        logger.info("scheduling completed")

        return base_time, capture_tasks, wavegen_tasks


def single_schedule(
    cp: str, pgs: Collection[str], boxpool: BoxPool, power_thr: float, displacement: int = 0
) -> tuple[int, npt.NDArray[np.complex64], tuple[tuple[int, int], ...]]:
    base_time, c_tasks, g_tasks = boxpool.start_at(
        runit_name=cp, channel_names=pgs, min_time_offset=125_000_000 // 10, displacement=displacement
    )
    for boxname, g_task in g_tasks.items():
        g_task.result()
        logger.info(f"wave generation of box {boxname} is completed")

    rdrs: dict[str, dict[tuple[Quel1PortType, int], CapIqDataReader]] = {}
    for boxname, c_task in c_tasks.items():
        rdrs[boxname] = c_task.result()
        logger.info(f"capture of box {boxname} is completed")

    if len(rdrs) != 1:
        raise AssertionError("too much reader objects...")

    cp_box, cp_port, cp_runits = boxpool._runits[cp]

    rdr = rdrs[cp_box]
    iq = rdr[cp_port, list(cp_runits)[0]].as_wave_dict()["s0"][0]
    chunks = find_chunks(iq, power_thr=power_thr)
    return base_time, iq, chunks


def find_chunks(
    iq: npt.NDArray[np.complex64], power_thr=1000.0, space_thr=16, minimal_length=16
) -> tuple[tuple[int, int], ...]:
    chunk = (abs(iq) > power_thr).nonzero()[0]
    if len(chunk) == 0:
        logger.info("no pulse!")
        return ()

    gaps = (chunk[1:] - chunk[:-1]) > space_thr
    start_edges = list(chunk[1:][gaps])
    start_edges.insert(0, chunk[0])
    last_edges = list(chunk[:-1][gaps])
    last_edges.append(chunk[-1])
    chunks = tuple([(s, e) for s, e in zip(start_edges, last_edges) if e - s >= minimal_length])

    n_chunks = len(chunks)
    logger.info(f"number_of_chunks: {n_chunks}")
    for i, chunk in enumerate(chunks):
        s, e = chunk
        iq0 = np.average(iq[s:e])
        angle = round(np.arctan2(iq0.real, iq0.imag) * 180.0 / np.pi, 1)
        logger.info(f"  chunk {i}: {e - s} samples, ({s} -- {e}),  mean phase = {angle:.1f}")
    return chunks


def calc_angle(iq) -> tuple[float, float, float]:
    angle = np.angle(iq)
    min_angle = min(angle)
    max_angle = max(angle)
    if max_angle - min_angle > 6.0:
        angle = (angle + 2 * np.pi) % np.pi

    avg = np.mean(angle) * 180.0 / np.pi
    sd = np.sqrt(np.var(angle)) * 180.0 / np.pi
    delta = (max(angle) - min(angle)) * 180.0 / np.pi
    return avg, sd, delta
