import logging
from concurrent.futures import Future
from typing import Any, Dict, Final, List, Mapping, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import CaptureReturnCode, Quel1Box

logger = logging.getLogger(__name__)


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

    def get_box(self, name: str) -> [Quel1Box, SequencerClient]:
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
        for pg in pgs:
            pg_by_box[pg.boxname].add(pg)

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
                valid_sched = sqc.add_sequencer(t)
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
    NUM_SAMPLES_IN_WAVE_BLOCK: Final[int] = 64  # this should be taken from e7awgsw

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
        if not all(box.link_status().values()):
            raise RuntimeError(f"sender '{name}' is not available due to link problem of '{boxname}'")

        self.name: str = name
        self.boxname: str = boxname
        self.box: Quel1Box = box
        self.sqc: SequencerClient = sqc
        self.port, self.subport = self.box.decode_port(port)
        self.channel: int = channel  # TODO: better to check the validity
        self.awg_spec: Tuple[Union[int, Tuple[int, int]], int] = (port, channel)

    def config(self, *, fnco_freq: Union[float, None] = None, **kwargs):
        self.box.config_port(port=self.port, subport=self.subport, **kwargs)
        self.box.config_channel(port=self.port, subport=self.subport, channel=self.channel, fnco_freq=fnco_freq)

    def load_cw(
        self, amplitude: float, num_wave_sample: int, num_repeats: Tuple[int, int], num_wait_samples: Tuple[int, int]
    ) -> None:
        # TODO: define default values
        self.box.load_cw_into_channel(
            port=self.port,
            subport=self.subport,
            channel=self.channel,
            amplitude=amplitude,
            num_wave_sample=num_wave_sample,
            num_repeats=num_repeats,
            num_wait_samples=num_wait_samples,
        )

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
    def __init__(
        self,
        name: str,
        *,
        boxname: str,
        port: int,
        runits: Set[int],
        background_noise_threshold: float,
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
        self.background_noise_threshold = background_noise_threshold
        # self.capmod = self.box.rmap.get_capture_module_of_rline(self.group, self.rline)

    def config(self, *, fnco_freq: Union[float, None] = None, **kwargs):
        # TODO: DSP setting will be added here somehow.
        self.box.config_port(port=self.port, **kwargs)
        self.box.config_runit(port=self.port, runit=0, fnco_freq=fnco_freq)  # should convert runit -> rchannel

    def capture_now(self, *, num_samples: int, delay_samples: int = 0):
        thunk = self.box.simple_capture_start(port=self.port, runits=self.runits, num_samples=num_samples, delay_samples=delay_samples)
        status, iqs = thunk.result()
        return status, iqs

    def check_noise(self, show_graph: bool = True):
        status, iq = self.capture_now(num_samples=1024)
        if status == CaptureReturnCode.SUCCESS:
            noise_avg, noise_max = np.average(abs(iq[0])), max(abs(iq[0]))
            logger.info(f"background noise: max = {noise_max:.1f}, avg = {noise_avg:.1f}")
            judge = noise_max < self.background_noise_threshold
            if show_graph:
                plot_iqs({"test": iq[0]})
            if not judge:
                raise RuntimeError(
                    "the capture port is too noisy, check the output ports connected to the capture port"
                )
        else:
            raise RuntimeError(f"capture failure due to {status}")

    def capture_at_single_trigger_of(self, *, pg: PulseGen, num_samples: int, delay_samples: int = 0) -> Future:
        if pg.box != self.box:
            raise ValueError("can not be triggered by an awg of the other box")
        return self.box.simple_capture_start(
            port=self.port,
            runits=self.runits,
            num_samples=num_samples,
            delay_samples=delay_samples,
            triggering_channel=pg.awg_spec,
        )

    # TODO: activate it later again.
    """
    def capture_at_multiple_triggers_of(
        self, *, pg: PulseGen, num_iters: int, num_samples: int, delay: int = 0
    ) -> CaptureResults:
        if pg.box != self.box:
            raise ValueError("can not be triggered by an awg of the other box")
        return self.box.wss.capture_start(
            num_iters=num_iters,
            capmod=self.capmod,
            capunits=(self.runit,),
            num_words=num_samples // 4,
            delay=delay,
            triggering_awg=pg.awg,
        )
    """


def create_pulsegen(
    settings: Mapping[str, Mapping[str, Mapping[str, Any]]],
    boxpool: BoxPool,
) -> Dict[str, PulseGen]:
    pgs: Dict[str, PulseGen] = {}
    senders = [s for s in settings if s.startswith("SENDER")]
    for sender in senders:
        pgs[sender] = PulseGen(name=sender, **settings[sender]["create"], boxpool=boxpool)
        # Notes: you can call the following methods to reconfigure the PG anytime you need.
        pgs[sender].config(**settings[sender]["config"])
        pgs[sender].load_cw(**settings[sender]["cw"])
    return pgs


def create_pulsecap(
    settings: Mapping[str, Mapping[str, Mapping[str, Any]]],
    boxpool: BoxPool,
) -> Dict[str, PulseCap]:
    cps: Dict[str, PulseCap] = {}
    capturers = [s for s in settings if s.startswith("CAPTURER")]
    for capturer in capturers:
        cps[capturer] = PulseCap(name=capturer, **settings[capturer]["create"], boxpool=boxpool)
        # Notes: you can call the following methods to reconfigure the CP anytime you need.
        cps[capturer].config(**settings[capturer]["config"])
    return cps


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


def plot_iqs(iq_dict) -> None:
    n_plot = len(iq_dict)
    fig = plt.figure()

    m = 0
    for _, iq in iq_dict.items():
        m = max(m, np.max(abs(np.real(iq))))
        m = max(m, np.max(abs(np.imag(iq))))

    idx = 0
    for title, iq in iq_dict.items():
        ax = fig.add_subplot(n_plot, 1, idx + 1)
        ax.plot(np.real(iq))
        ax.plot(np.imag(iq))
        ax.text(0.05, 0.1, f"{title}", transform=ax.transAxes)
        ax.set_ylim((-m * 1.1, m * 1.1))
        idx += 1
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib

    from quel_ic_config import Quel1BoxType

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    matplotlib.use("Qt5agg")

    def simple_trigger(cp1: PulseCap, pg1: PulseGen):
        thunk = cp1.capture_at_single_trigger_of(pg=pg1, num_samples=1024, delay_samples=0)
        pg1.emit_now()
        s0, iq = thunk.result()
        iq0 = iq[0]
        assert s0 == CaptureReturnCode.SUCCESS
        chunks = find_chunks(iq0)
        return iq0, chunks

    def single_schedule(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool):
        if pg_trigger not in pgs:
            raise ValueError("trigerring pulse generator is not included in activated pulse generators")
        thunk = cp.capture_at_single_trigger_of(pg=pg_trigger, num_samples=1024, delay_samples=0)
        boxpool.emit_at(cp=cp, pgs=pgs, min_time_offset=125_000_000, time_counts=(0,))

        s0, iqs = thunk.result()
        iq0 = iqs[0]
        assert s0 == CaptureReturnCode.SUCCESS
        chunks = find_chunks(iq0)
        return iq0, chunks

    COMMON_SETTINGS: Mapping[str, Any] = {
        "lo_freq": 11500e6,
        "cnco_freq": 1500.0e6,
        "fnco_freq": 0,
        "sideband": "L",
        "amplitude": 6000.0,
    }

    DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
        "CLOCK_MASTER": {
            "ipaddr": "10.3.0.13",
            "reset": True,
        },
        "BOX0": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.QuEL1_TypeA,
            "config_root": None,
            "config_options": [],
        },
        "BOX1": {
            "ipaddr_wss": "10.1.0.58",
            "ipaddr_sss": "10.2.0.58",
            "ipaddr_css": "10.5.0.58",
            "boxtype": Quel1BoxType.QuEL1_TypeA,
            "config_root": None,
            "config_options": [],
        },
        "BOX2": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.QuEL1_TypeB,
            "config_root": None,
            "config_options": [],
        },
    }

    VPORT_SETTINGS: Dict[str, Mapping[str, Mapping[str, Any]]] = {
        "CAPTURER": {
            "create": {
                "boxname": "BOX0",
                "port": 0,  # (0, "r")
                "runits": {0},
                "background_noise_threshold": 200.0,
            },
            "config": {
                "lo_freq": COMMON_SETTINGS["lo_freq"],
                "cnco_freq": COMMON_SETTINGS["cnco_freq"],
                "fnco_freq": COMMON_SETTINGS["fnco_freq"],
                "rfswitch": "open",
            },
        },
        "SENDER0": {
            "create": {
                "boxname": "BOX0",
                "port": 8,  # (1, 2)
                "channel": 0,
            },
            "config": {
                "lo_freq": COMMON_SETTINGS["lo_freq"],
                "cnco_freq": COMMON_SETTINGS["cnco_freq"],
                "fnco_freq": COMMON_SETTINGS["fnco_freq"],
                "sideband": COMMON_SETTINGS["sideband"],
                "vatt": 0xA00,
            },
            "cw": {
                "amplitude": COMMON_SETTINGS["amplitude"],
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (0, 80),
            },
        },
    }

    boxpool0 = BoxPool(DEVICE_SETTINGS)
    boxpool0.init(resync=True)
    pgs0 = create_pulsegen(VPORT_SETTINGS, boxpool0)
    cps0 = create_pulsecap(VPORT_SETTINGS, boxpool0)
    cp0 = cps0["CAPTURER"]

    boxpool0.measure_timediff(cp0)

    box0, sqc0 = boxpool0.get_box("BOX0")

    # Notes: close loop before checking the noise
    box0.config_rfswitch(port=0, rfswitch="loop")  # TODO: capturer should control its loop switch
    box0.config_rfswitch(port=7, rfswitch="loop")
    box0.activate_monitor_loop(0)
    box0.activate_monitor_loop(1)
    cp0.check_noise(show_graph=False)

    # Notes: monitor should be
    box0.config_rfswitch(port=0, rfswitch="open")
    box0.config_rfswitch(port=7, rfswitch="open")
    box0.deactivate_monitor_loop(0)
    box0.deactivate_monitor_loop(1)

    iqs0 = simple_trigger(cp0, pgs0["SENDER0"])
    plot_iqs({"cap0": iqs0[0]})

    iqs1 = single_schedule(cp0, pgs0["SENDER0"], {pgs0["SENDER0"]}, boxpool0)
    plot_iqs({"cap0": iqs1[0]})
