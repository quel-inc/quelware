import collections.abc
import logging
from concurrent.futures import Future
from typing import Any, Collection, Dict, Final, List, Mapping, Sequence, Set, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence

from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import CaptureReturnCode, Quel1Box

logger = logging.getLogger(__name__)

VportSettingType = Union[Mapping[str, Any], Mapping[int, Mapping[str, Any]]]


class BoxPool:
    SYSREF_PERIOD: Final[int] = 2000
    # TODO: tried to find the best value, but the best value changes link-up by link-up. so, calibration is required.
    TIMING_OFFSET: Final[int] = 0
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(self, settings: Mapping[str, Mapping[str, Any]]):
        self._clock_master = QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        self._boxes: Dict[str, Tuple[Quel1Box, SequencerClient]] = {}
        self._linkstatus: Dict[str, bool] = {}
        self._parse_settings(settings)
        self._estimated_timediff: Dict[str, int] = {boxname: 0 for boxname in self._boxes}
        self._cap_sysref_time_offset: int = 0

    def _parse_settings(self, settings: Mapping[str, Mapping[str, Any]]):
        for k, v in settings.items():
            if k.startswith("BOX"):
                box = Quel1Box.create(**v)
                sqc = SequencerClient(v["ipaddr_sss"])
                self._boxes[k] = (box, sqc)
                self._linkstatus[k] = False

    def init(self, reconnect: bool = True, resync: bool = True):
        self.scan_link_status(reconnect=reconnect)
        self.reset_awg()
        if resync:
            self.resync()
        if not self.check_clock():
            raise RuntimeError("failed to acquire time count from some clocks")

    def scan_link_status(self, reconnect=False):
        for name, (box, sqc) in self._boxes.items():
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

    def reset_awg(self):
        for name, (box, _) in self._boxes.items():
            box.easy_stop_all(control_port_rfswitch=True)
            box.initialize_all_awgs()
            box.initialize_all_capunits()

    def resync(self):
        self._clock_master.reset()  # TODO: confirm whether it is harmless or not.
        self._clock_master.kick_clock_synch([sqc.ipaddress for _, (_, sqc) in self._boxes.items()])

    def check_clock(self) -> bool:
        valid_m, cntr_m = self._clock_master.read_clock()
        t = {}
        for name, (_, sqc) in self._boxes.items():
            t[name] = sqc.read_clock()

        flag = True
        if valid_m:
            logger.info(f"master: {cntr_m:d}")
        else:
            flag = False
            logger.info("master: not found")

        for name, (valid, cntr, cntr_last_sysref) in t.items():
            if valid:
                logger.info(f"{name:s}: {cntr:d} {cntr_last_sysref:d}")
            else:
                flag = False
                logger.info(f"{name:s}: not found")
        return flag

    def get_box(self, name: str) -> Tuple[Quel1Box, SequencerClient]:
        if name in self._boxes:
            box, sqc = self._boxes[name]
            return box, sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_linkstatus(self, name: str) -> bool:
        if name in self._boxes:
            return self._linkstatus[name]
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def emit_at(
        self,
        cp: "PulseCap",
        pgs: Set["PulseGen"],
        min_time_offset: int,
        time_counts=Sequence[int],
        displacement: int = 0,
    ) -> Dict[str, List[int]]:
        if len(pgs) == 0:
            logger.warning("no pulse generator to activate")

        pg_by_box: Dict[str, Set["PulseGen"]] = {box: set() for box in self._boxes}
        bitmap_by_box: Dict[str, int] = {box: 0 for box in self._boxes}
        for pg in pgs:
            pg_by_box[pg.boxname].add(pg)
            # TODO: move the following logic to the right place
            b = self._boxes[pg.boxname][0]
            awg_idx = b.rmap.get_awg_of_channel(*(b._convert_output_channel(pg.awg_spec)))
            bitmap_by_box[pg.boxname] |= 1 << awg_idx

        if cp.boxname not in pg_by_box:
            raise RuntimeError("impossible to trigger the capturer")

        # initialize awgs
        targets: Dict[str, SequencerClient] = {}
        for boxname, pgs in pg_by_box.items():
            box, sqc = self._boxes[boxname]
            if len(pgs) == 0:
                continue
            targets[boxname] = sqc
            box.prepare_for_emission({pg.awg_spec for pg in pgs})
            # Notes: the following is not required actually, just for debug purpose.
            valid_read, current_time, last_sysref_time = sqc.read_clock()
            if valid_read:
                logger.info(
                    f"boxname: {boxname}, current time: {current_time}, "
                    f"sysref offset: {last_sysref_time % self.SYSREF_PERIOD}"
                )
            else:
                raise RuntimeError("failed to read current clock")

        valid_read, current_time, last_sysref_time = targets[cp.boxname].read_clock()
        logger.info(
            f"sysref offset: average: {self._cap_sysref_time_offset},  latest: {last_sysref_time % self.SYSREF_PERIOD}"
        )
        if abs(last_sysref_time % self.SYSREF_PERIOD - self._cap_sysref_time_offset) > 4:
            logger.warning("large fluctuation of sysref is detected on the FPGA")
        base_time = current_time + min_time_offset
        offset = (16 - (base_time - self._cap_sysref_time_offset) % 16) % 16
        base_time += offset
        base_time += displacement  # inducing clock displacement for performance evaluation (must be 0 usually).
        base_time += self.TIMING_OFFSET  # Notes: the safest timing to issue trigger, at the middle of two AWG block.
        schedule: Dict[str, List[int]] = {boxname: [] for boxname in targets}
        for i, time_count in enumerate(time_counts):
            for boxname, sqc in targets.items():
                t = base_time + time_count + self._estimated_timediff[boxname]
                valid_sched = sqc.add_sequencer(t, awg_bitmap=bitmap_by_box[boxname])
                if not valid_sched:
                    raise RuntimeError("failed to schedule AWG start")
                schedule[boxname].append(t)
        logger.info("scheduling completed")
        return schedule

    def measure_timediff(self, cp: "PulseCap", num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS) -> None:
        counter_at_sysref_clk: Dict[str, int] = {boxname: 0 for boxname in self._boxes}

        for i in range(num_iters):
            for name, (_, sqc) in self._boxes.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError(f"firmware of {name} doesn't support this measurement")
                counter_at_sysref_clk[name] += m[2] % self.SYSREF_PERIOD

        avg: Dict[str, int] = {boxname: round(cntr / num_iters) for boxname, cntr in counter_at_sysref_clk.items()}
        adj = avg[cp.boxname]
        self._estimated_timediff = {boxname: cntr - adj for boxname, cntr in avg.items()}
        logger.info(f"estimated time difference: {self._estimated_timediff}")

        self._cap_sysref_time_offset = avg[cp.boxname]


class PulseGen:
    @classmethod
    def create(cls, settings: Mapping[str, Mapping[str, VportSettingType]], boxpool: BoxPool) -> Dict[str, "PulseGen"]:
        settings_ = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], settings)
        pgs: Dict[str, PulseGen] = {}
        for sender, setting in settings_.items():
            pgs[sender] = PulseGen(name=sender, **setting["create"], boxpool=boxpool)

            if "config" in setting:
                pgs[sender].config(**setting["config"])

            if "cw_parameter" in setting:
                pgs[sender].load_cw(**setting["cw_parameter"])
            elif "raw_parameter" in setting:
                raw_parameter = setting["raw_parameter"]
                if isinstance(raw_parameter, WaveSequence):
                    pgs[sender].load_wave_parameter(raw_parameter)
                else:
                    raise TypeError("unexpected raw_parameter, it is expected to be an e7awgsw.WaveSequence object")
        return pgs

    def __init__(
        self,
        name: str,
        *,
        boxname: str,
        port: Union[int, Tuple[int, int]],
        channel: int,
        boxpool: BoxPool,
    ):
        # TODO: eliminate the necessity of boxpool by adding sqc to Quel1Box.
        #       status can be obtained by box.linkstatus()
        box, sqc = boxpool.get_box(boxname)

        self.name: str = name
        self.boxname: str = boxname
        self.box: Quel1Box = box
        self.sqc: SequencerClient = sqc
        self.port, self.subport = self.box.decode_port(port)
        self.channel: int = channel  # TODO: better to check the validity
        self.awg_spec: Tuple[Union[int, Tuple[int, int]], int] = (port, channel)

    def config(self, *, fnco_freq: Union[float, None] = None, **kwargs):
        self.box.config_port(port=self.port, subport=self.subport, **kwargs)
        if fnco_freq is not None:
            self.box.config_channel(port=self.port, subport=self.subport, channel=self.channel, fnco_freq=fnco_freq)

    def load_cw(
        self, amplitude: float, num_wave_sample: int, num_repeats: Tuple[int, int], num_wait_samples: Tuple[int, int]
    ) -> None:
        self.box.load_cw_into_channel(
            port=self.port,
            subport=self.subport,
            channel=self.channel,
            amplitude=amplitude,
            num_wave_sample=num_wave_sample,
            num_repeats=num_repeats,
            num_wait_samples=num_wait_samples,
        )

    def _set_param_awg(self, wave_param: WaveSequence):
        group, line = self.box._convert_output_port_decoded(self.port, self.subport)
        awgidx = self.box.rmap.get_awg_of_channel(group, line, self.channel)
        self.box.wss.set_param_awg(awgidx, wave_param)

    def load_wave_parameter(self, wave_param: WaveSequence):
        """loading detailed parameters of wave to be generated. this API is tentative and its wave_param argument will
        be replaced with more convenient one in the near future.
        :param wave_param: a raw object (WaveSeuqence) describing the wave to be generated
        :return: None
        """
        self.box.config_channel(port=self.port, subport=self.subport, channel=self.channel)
        self._set_param_awg(wave_param)

    def prepare_for_emission(self):
        self.box.prepare_for_emission({self.awg_spec})

    def emit_now(self) -> None:
        self.box.start_emission({self.awg_spec})

    def emit_at(self, min_time_offset: int, time_counts: Sequence[int]) -> None:
        self.prepare_for_emission()
        valid_read, current_time, last_sysref_time = self.sqc.read_clock()
        if valid_read:
            logger.info(f"current time: {current_time},  last sysref time: {last_sysref_time}")
        else:
            raise RuntimeError("failed to read current clock")

        base_time = current_time + min_time_offset  # TODO: implement constraints of the start timing
        for i, time_count in enumerate(time_counts):
            valid_sched = self.sqc.add_sequencer(base_time + time_count)
            if not valid_sched:
                raise RuntimeError("failed to schedule AWG start")
        logger.info("scheduling completed")

    def stop_now(self) -> None:
        self.box.stop_emission({self.awg_spec})


class PulseCap:
    @classmethod
    def create(cls, settings: Mapping[str, Mapping[str, VportSettingType]], boxpool: BoxPool) -> Dict[str, "PulseCap"]:
        cps: Dict[str, PulseCap] = {}
        for capturer, setting in settings.items():
            cps[capturer] = PulseCap(name=capturer, **cast(Mapping[str, Any], setting["create"]), boxpool=boxpool)
            # Notes: you can call the following methods to reconfigure the CP anytime you need.
            if "config" in setting:
                cps[capturer].config(**cast(Mapping[str, Any], setting["config"]))

            if "simple_parameters" in setting:
                simple_params = cast(Mapping[int, Mapping[str, Any]], setting["simple_parameters"])
                for runit, simple_param in simple_params.items():
                    cps[capturer].load_capture_parameter(runit, cls.make_simple_capture_param(**simple_param))
            elif "raw_parameter" in setting:
                raw_params = cast(Mapping[int, Any], setting["raw_parameters"])
                for runit, raw_param in raw_params.items():
                    if isinstance(raw_param, CaptureParam):
                        cps[capturer].load_capture_parameter(runit, raw_param)
                    else:
                        raise TypeError("raw_parameter of each runit is expected to be an e7awgsw.CaptureParam object")
        return cps

    @staticmethod
    def make_simple_capture_param(
        num_delay_sample: int,
        num_integration_section: int,
        num_capture_samples: Sequence[int],
        num_blank_samples: Sequence[int],
    ):
        capprm = CaptureParam()

        if num_delay_sample % 4 != 0:
            raise ValueError(f"num_delay_sample (= {num_delay_sample} is not multiple of 4.")
        capprm.capture_delay = num_delay_sample // 4

        capprm.num_integ_sections = num_integration_section
        for idx in range(len(num_capture_samples)):
            if num_capture_samples[idx] % 4 != 0:
                raise ValueError(f"num_capture_samples[{idx}] (= {num_capture_samples[idx]}) is not multiple of 4")
            if num_blank_samples[idx] % 4 != 0:
                raise ValueError(f"num_blank_samples[{idx}] (= {num_blank_samples[idx]}) is not multiple of 4")
            capprm.add_sum_section(
                num_words=num_capture_samples[idx] // 4, num_post_blank_words=num_blank_samples[idx] // 4
            )

        return capprm

    def __init__(
        self,
        name: str,
        *,
        boxname: str,
        port: int,
        runits: Set[int],
        boxpool: BoxPool,
    ):
        box, _ = boxpool.get_box(boxname)
        if not boxpool.get_linkstatus(boxname):
            raise RuntimeError(f"sender '{name}' is not available due to link problem of '{boxname}'")

        self.name: str = name
        self.boxname: str = boxname
        self.box: Quel1Box = box
        self.port = port
        self.runits = runits
        self.capture_parameter: Dict[int, CaptureParam] = {}

    def config(self, *, fnco_freq: Union[Mapping[int, float], float, None] = None, **kwargs):
        # TODO: DSP setting will be added here somehow.
        self.box.config_port(port=self.port, **kwargs)
        if isinstance(fnco_freq, collections.abc.Mapping):
            for runit in self.runits:
                self.box.config_runit(port=self.port, runit=runit, fnco_freq=fnco_freq[runit])
        elif fnco_freq is not None:
            for runit in self.runits:
                self.box.config_runit(port=self.port, runit=runit, fnco_freq=fnco_freq)
        else:
            raise TypeError(f"malformed fnco_freq: {fnco_freq}")

    def capture_now(self, *, num_samples: int, delay_samples: int = 0):
        thunk = self.box.simple_capture_start(
            port=self.port, runits=self.runits, num_samples=num_samples, delay_samples=delay_samples
        )
        status, iqs = thunk.result()
        self.reload_capture_parameter()
        return status, iqs

    def measure_background_noise(self) -> Tuple[float, float, npt.NDArray[np.complex64]]:
        thunk = self.box.simple_capture_start(port=self.port, runits={0}, num_samples=4096)
        status, iq = thunk.result()
        self.reload_capture_parameter({0})
        if status == CaptureReturnCode.SUCCESS:
            noise_avg, noise_max = np.average(abs(iq[0])), max(abs(iq[0]))
            logger.info(f"background noise: max = {noise_max:.1f}, avg = {noise_avg:.1f}")
            return float(noise_avg), float(noise_max), iq[0]
        else:
            raise RuntimeError(f"capture failure due to {status}")

    def _set_param_capunit(self, runit: int, cap_param: CaptureParam):
        group, rline = self.box._convert_input_port(self.port)
        capmod = self.box.rmap.get_capture_module_of_rline(group, rline)
        self.box.wss.set_param_capunit(capmod=capmod, capunit=runit, capprm=cap_param)

    def load_capture_parameter(self, runit: int, cap_param: CaptureParam) -> None:
        """loading detailed parameters describing how to capture the wave data from the specified runit. this API is
        tentative and its cap_param argument will be replaced with more convenient one in the near future.

        :param runit: an index of runit
        :param cap_param: a raw object (CaptureParam) describing the details of the signal capture to be conducted.
        :return: None
        """
        if runit not in self.runits:
            raise ValueError(f"an invalid runit: {runit}")
        self.box.config_runit(port=self.port, runit=runit)
        self._set_param_capunit(runit, cap_param)
        self.capture_parameter[runit] = cap_param

    def reload_capture_parameter(self, runits: Union[Collection[int], None] = None) -> None:
        if runits is None:
            runits = self.runits
        for runit, capprm in self.capture_parameter.items():
            if runit in runits:
                self.box.config_runit(port=self.port, runit=runit)
                self._set_param_capunit(runit, capprm)

    def capture_at_single_trigger_of(self, *, pg: PulseGen) -> Future:
        if pg.box != self.box:
            raise ValueError("can not be triggered by an awg of the other box")
        return self.box.capture_start(
            port=self.port,
            runits=self.runits,
            triggering_channel=pg.awg_spec,
        )


def make_pulses_wave_param(
    num_delay_sample: int,
    num_global_repeat: int,
    num_wave_samples: Sequence[int],
    num_blank_samples: Sequence[int],
    num_repeats: Sequence[int],
    amplitudes: Sequence[complex],
) -> WaveSequence:
    if num_delay_sample % 64 != 0:
        raise ValueError(f"num_delay_sample (= {num_delay_sample}) is not multiple of 64")
    wave = WaveSequence(num_wait_words=num_delay_sample // 4, num_repeats=num_global_repeat)

    for idx in range(len(num_wave_samples)):
        if num_wave_samples[idx] % 64 != 0:
            raise ValueError(f"num_wave_samples[{idx}] (= {num_wave_samples[idx]}) is not multiple of 64")
        if num_blank_samples[idx] % 4 != 0:
            raise ValueError(f"num_blank_samples[{idx}] (= {num_blank_samples[idx]}) is not multiple of 4")
        iq = np.zeros(num_wave_samples[idx], dtype=np.complex64)
        iq[:] = (1 + 0j) * amplitudes[idx]
        block_assq: List[Tuple[int, int]] = list(zip(iq.real.astype(int), iq.imag.astype(int)))
        wave.add_chunk(iq_samples=block_assq, num_blank_words=num_blank_samples[idx] // 4, num_repeats=num_repeats[idx])
    return wave


def find_chunks(
    iq: npt.NDArray[np.complex64], power_thr=1000.0, space_thr=16, minimal_length=16
) -> Tuple[Tuple[int, int], ...]:
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
        logger.info(f"  chunk {i}: {e - s} samples, ({s} -- {e})")
    return chunks


def calc_angle(iq) -> Tuple[float, float, float]:
    angle = np.angle(iq)
    min_angle = min(angle)
    max_angle = max(angle)
    if max_angle - min_angle > 6.0:
        angle = (angle + 2 * np.pi) % np.pi

    avg = np.mean(angle) * 180.0 / np.pi
    sd = np.sqrt(np.var(angle)) * 180.0 / np.pi
    delta = (max(angle) - min(angle)) * 180.0 / np.pi
    return avg, sd, delta


def plot_iqs(iq_dict, t_offset: int = 0) -> None:
    n_plot = len(iq_dict)

    m = 0
    for _, iq in iq_dict.items():
        m = max(m, np.max(abs(np.real(iq))))
        m = max(m, np.max(abs(np.imag(iq))))

    fig, axs = plt.subplots(n_plot, sharex="col")
    if n_plot == 1:
        axs = [axs]
    fig.set_size_inches(10.0, 2.0 * n_plot)
    fig.subplots_adjust(bottom=max(0.025, 0.125 / n_plot), top=min(0.975, 1.0 - 0.05 / n_plot))
    for idx, (title, iq) in enumerate(iq_dict.items()):
        t = np.arange(0, len(iq)) - t_offset
        axs[idx].plot(t, np.real(iq))
        axs[idx].plot(t, np.imag(iq))
        axs[idx].set_ylim((-m * 1.1, m * 1.1))
        axs[idx].text(0.05, 0.1, title, transform=axs[idx].transAxes)
    plt.show()
