import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Set, Tuple, Union, cast

from quel_ic_config.exstickge_coap_tempctrl_client import Quel1seTempctrlState, _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd7490Mixin,
    Quel1ConfigSubsystemBaseSlot,
    Quel1ConfigSubsystemPowerboardPwmMixin,
)
from quel_ic_config.quel1_thermistor import Quel1Thermistor

logger = logging.getLogger(__name__)


class Quel1ConfigSubsystemTempctrlMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    @property
    def tempctrl_auto_start_at_linkup(self):
        return False

    def get_tempctrl_state(self) -> Quel1seTempctrlState:
        # Notes: temperature control is running independently of the control subsystem.
        return Quel1seTempctrlState.RUN

    def get_tempctrl_state_count(self) -> int:
        # Notes: temperature control is running independently of the control subsystem.
        return 0

    def start_tempctrl(self, new_count: Union[int, None] = None) -> None:
        # Notes: this method should not be called, just exists for the purpose of static analysis.
        logger.warning(f"temperature control starts at power-on in this boxtype (={self._boxtype}), do nothing")


class Quel1seConfigSubsystemTempctrlMixin(Quel1ConfigSubsystemTempctrlMixin):
    __slots__ = ()

    _THERMISTORS: dict[tuple[int, int], Quel1Thermistor]
    _ACTUATORS: dict[str, tuple[str, int]]
    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool
    _TEMPCTRL_POLLING_PERIOD: float = 0.5  # second
    _TEMPCTRL_POLLING_MAX_RETRY: int = 3  # times

    @classmethod
    def get_thermistor_desc(cls) -> dict[tuple[int, int], Quel1Thermistor]:
        return cls._THERMISTORS

    @classmethod
    def get_actuator_desc(cls) -> dict[str, tuple[str, int]]:
        return cls._ACTUATORS

    def _construct_tempctrl(self):
        self._tempctrl_auto_start_at_linkup: bool = self._DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP
        self._tempctrl_watcher = ThreadPoolExecutor(max_workers=1)

    def init_tempctrl(self, param: Dict[str, Any]):
        # Notes: currently nothing to do, define it for future.
        pass

    @property
    def tempctrl_auto_start_at_linkup(self):
        return self._tempctrl_auto_start_at_linkup

    @tempctrl_auto_start_at_linkup.setter
    def tempctrl_auto_start_at_linkup(self, v: bool):
        self._tempctrl_auto_start_at_linkup = v

    @property
    def tempctrl_loop_state(self) -> Dict[str, Union[int, str]]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        loop_state: Quel1seTempctrlState = proxy.read_tempctrl_state()
        retval: Dict[str, Union[int, str]] = {
            "loop_count": proxy.read_tempctrl_loop_count(),
            "loop_state": loop_state.tostr(),
        }
        if loop_state == Quel1seTempctrlState.PRERUN:
            retval["time_to_start_running"] = proxy.read_tempctrl_state_count()
        return retval

    @property
    def tempctrl_error_count(self) -> Dict[str, int]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        retval: Dict[str, int] = {
            "skip": proxy.read_tempctrl_skip_count(),  # Notes: number of skipped trials of thermal control
            "error": proxy.read_tempctrl_error_count(),  # Notes: SPI error
            "timeout": proxy.read_tempctrl_timeout_count(),  # Notes: SPI timeout
            "broken": proxy.read_tempctrl_broken_count(),  # Notes: broken channel index
            "mismatch": proxy.read_tempctrl_mismatch_count(),  # Notes: large mismatch between two consecutive readings
            "mismatch_max": proxy.read_tempctrl_mismatch_max(),
        }
        return retval

    def _convert_temperatures_index(self, tll: List[List[int]]) -> Dict[Tuple[int, int], float]:
        temperatures: Dict[Tuple[int, int], float] = {}

        for i in range(self._NUM_IC["ad7490"]):
            for j, t in enumerate(tll[i]):
                if (i, j) in self._THERMISTORS:
                    try:
                        temperatures[i, j] = self._THERMISTORS[i, j].convert(t)
                    except ValueError:
                        logger.error(f"unexpected reading {t} at AD7490[{i}], {j}-th input")

        return temperatures

    def _convert_temperatures_name(self, tll: List[List[int]]) -> Dict[str, float]:
        temperatures: Dict[str, float] = {}

        for i in range(self._NUM_IC["ad7490"]):
            for j, t in enumerate(tll[i]):
                if (i, j) in self._THERMISTORS:
                    try:
                        th = self._THERMISTORS[i, j]
                        temperatures[th.name] = th.convert(t)
                    except ValueError:
                        logger.error(f"unexpected reading {t} at AD7490[{i}], {j}-th input")

        return temperatures

    def _get_ad7490_readings_thread_main(self) -> List[List[int]]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        cnt0 = proxy.read_tempctrl_loop_count()
        interval = proxy.read_tempctrl_loop_interval()
        num_iter = int(interval / self._TEMPCTRL_POLLING_PERIOD * 1.5) + 1
        for i in range(num_iter):
            cnt1 = proxy.read_tempctrl_loop_count()
            if cnt1 == cnt0:
                time.sleep(self._TEMPCTRL_POLLING_PERIOD)
            elif cnt1 == cnt0 + 1:
                break
            else:
                raise RuntimeError(f"an unexpected loop count {cnt1} comes after {cnt0}")
        else:
            raise TimeoutError(f"timeout occurred, loop count doesn't go up from {cnt0} too long")

        cnt2 = -1
        tll: List[List[int]] = []
        for _ in range(self._TEMPCTRL_POLLING_MAX_RETRY):
            # Notes: waiting for the firmware completes scanning temperatures.
            time.sleep(self._TEMPCTRL_POLLING_PERIOD)
            cnt2 = -1
            tll.clear()
            for i in range(self._NUM_IC["ad7490"]):
                cnt2, tl = proxy.read_tempctrl_rawdata_ad7490(i)
                if cnt2 != cnt1:
                    break
                tll.append(tl)
            if cnt2 != cnt1:
                logger.warning("the acquisition of temperature data is not completed yet")
                continue
            break
        else:
            raise RuntimeError(
                f"temperature data with an unexpected loop count {cnt2} (!= {cnt1}) is received "
                f"despite of {self._TEMPCTRL_POLLING_MAX_RETRY} trials"
            )
        return tll

    def get_tempctrl_temperature(self) -> Future[Dict[str, float]]:
        return self._tempctrl_watcher.submit(
            lambda: self._convert_temperatures_name(self._get_ad7490_readings_thread_main())
        )

    def get_tempctrl_temperature_now(self) -> Dict[str, float]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)

        tll: List[List[int]] = []
        cnt0: int = -1
        cnt2: int = -1
        for j in range(2):
            cnt0 = proxy.read_tempctrl_loop_count()
            tll.clear()
            for i in range(self._NUM_IC["ad7490"]):
                cnt2, tl = proxy.read_tempctrl_rawdata_ad7490(i)
                if cnt0 != cnt2:
                    # Notes: retry from the first because jump over the tick, unfortunately.
                    break
                tll.append(tl)
            if len(tll) == self._NUM_IC["ad7490"]:
                break
        else:
            raise RuntimeError(f"temperature data with an unexpected loop count {cnt2} (!= {cnt0}) is received")

        return self._convert_temperatures_name(tll)

    def start_tempctrl(self, new_count: Union[int, None] = None) -> None:
        cur_state = self.get_tempctrl_state()
        if cur_state in {Quel1seTempctrlState.INIT, Quel1seTempctrlState.IDLE}:
            logger.info("starting temperature control")
        elif cur_state in {Quel1seTempctrlState.PRERUN, Quel1seTempctrlState.RUN, Quel1seTempctrlState.DRYRUN}:
            logger.info("re-starting temperature control")
        else:
            raise AssertionError
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_state(Quel1seTempctrlState.PRERUN)
        if new_count is not None:
            proxy.write_tempctrl_state_count(new_count)

    def stop_tempctrl(self) -> None:
        cur_state = self.get_tempctrl_state()
        if cur_state == Quel1seTempctrlState.INIT:
            logger.info("re-initializing temperature control")
        elif cur_state in {Quel1seTempctrlState.PRERUN, Quel1seTempctrlState.RUN, Quel1seTempctrlState.DRYRUN}:
            logger.info("stopping and initializing temperature control")
        elif cur_state == Quel1seTempctrlState.IDLE:
            logger.info("initializing temperature control")
        else:
            raise AssertionError

        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_state(Quel1seTempctrlState.INIT)

    def start_tempctrl_external(self) -> None:
        logger.info("entering manual temperature control mode")
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_state(Quel1seTempctrlState.DRYRUN)

    def get_tempctrl_state(self) -> Quel1seTempctrlState:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        return proxy.read_tempctrl_state()

    def set_tempctrl_state(self, new_state: Union[str, Quel1seTempctrlState]) -> None:
        if isinstance(new_state, str):
            new_state = Quel1seTempctrlState.fromstr(new_state)
        if not isinstance(new_state, Quel1seTempctrlState):
            raise TypeError(f"unexpected new_state: {new_state}")
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_state(new_state)

    def get_tempctrl_state_count(self) -> int:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        return proxy.read_tempctrl_state_count()

    def set_tempctrl_state_count(self, new_count: int) -> None:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        return proxy.write_tempctrl_state_count(new_count)

    # TODO: fix it to return name-temp dictionary
    def get_tempctrl_setpoint(self) -> Dict[str, List[float]]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        return proxy.read_tempctrl_feedback_setpoint()

    # TODO: fix it to take name-temp dictionary
    def set_tempctrl_setpoint(self, fan: Sequence[float], heater: Sequence[float]):
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_feedback_setpoint(fan, heater)

    def get_tempctrl_gain(self) -> Dict[str, Dict[str, float]]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        return proxy.read_tempctrl_feedback_coeffcient()

    def set_tempctrl_gain(
        self, fan: Union[Dict[str, float], None] = None, heater: Union[Dict[str, float], None] = None
    ) -> None:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_feedback_coeffcient(fan, heater)

    def get_tempctrl_actuator_output(self) -> Dict[str, Dict[str, float]]:
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        outputs = proxy.read_tempctrl_rawdata_pwrpwm_current()
        fans: Dict[str, float] = {}
        heaters: Dict[str, float] = {}
        for name, (atype, aidx) in self.get_actuator_desc().items():
            if atype == "fan":
                fans[name] = outputs["fan"][aidx] / 1000
            elif atype == "heater":
                heaters[name] = outputs["heater"][aidx] / 1000
            else:
                raise AssertionError("never happens")
        return {
            "fan": fans,
            "heater": heaters,
        }

    def set_tempctrl_actuator_output(self, fan: Dict[str, float], heater: Dict[str, float]) -> None:
        fan_i: list[int] = [0 for _ in range(2)]
        heater_i: list[int] = [0 for _ in range(40)]
        for name, (atype, aidx) in self.get_actuator_desc().items():
            if atype == "fan":
                fan_i[aidx] = round(fan[name] * 1000)
            elif atype == "heater":
                heater_i[aidx] = round(heater[name] * 1000)
            else:
                raise AssertionError("never happens")
        proxy = cast(_ExstickgeCoapClientQuel1seTempctrlBase, self._proxy)
        proxy.write_tempctrl_rawdata_pwrpwm_next(fan_i, heater_i)


class Quel1seConfigSubsystemTempctrlDebugMixin(
    Quel1seConfigSubsystemTempctrlMixin, Quel1ConfigSubsystemAd7490Mixin, Quel1ConfigSubsystemPowerboardPwmMixin
):
    __slots__ = ()

    def _construct_tempctrl_debug(self):
        self._construct_tempctrl()
        self._construct_ad7490()
        self._construct_powerboard_pwm()

    def init_tempctrl_debug(self, param: Dict[str, Any]) -> None:
        self.init_tempctrl(param)
        for idx in range(self._NUM_IC["ad7490"]):
            self.init_ad7490(idx, param["ad7490"][idx])

    def _get_ad7490_readings_direct(self) -> List[List[int]]:
        tll: List[List[int]] = []

        for i in range(self._NUM_IC["ad7490"]):
            for k in range(3):
                tl = self.ad7490[i].read_adcs()
                if tl is not None:
                    break
            else:
                raise RuntimeError(f"failed to read temperature of {i}-th board repeatedly.")
            tll.append(tl)

        return tll

    def get_temperatures(self) -> Dict[Tuple[int, int], float]:
        return self._convert_temperatures_index(self._get_ad7490_readings_direct())

    def get_heater_master(self) -> bool:
        return self.powerboard_pwm[0].get_heater_master()

    def set_heater_master(self, en: bool) -> None:
        self.powerboard_pwm[0].set_heater_master(en)

    def get_heater_outputs(self, indices: Union[Set[int], None] = None) -> Dict[int, float]:
        ratios: Dict[int, float] = {}
        if indices is None:
            indices = {aidx for at, aidx in self.get_actuator_desc().values() if at == "heater"}
        for idx in range(self.powerboard_pwm[0].NUM_HEATER):
            if idx in indices:
                ratios[idx] = self.powerboard_pwm[0].get_heater_settings(idx)["high_ratio"]
        return ratios

    def set_heater_outputs(self, ratios: Dict[int, float]):
        indices = {aidx for at, aidx in self.get_actuator_desc().values() if at == "heater"}
        for idx, ratio in ratios.items():
            if idx not in indices:
                logger.warning(f"trying to configure an unavailable heater: {idx}")
            self.powerboard_pwm[0].set_heater_settings(idx, high_ratio=ratio)

    def get_fan_speed(self) -> float:
        s: List[float] = []
        for idx in {0, 1}:
            s.append(self.powerboard_pwm[0].get_fan_speed(idx))

        if s[0] == s[1]:
            return s[0]
        else:
            logger.warning(f"two fans have different speed: {s[0]:.3f} and {s[1]:.3f}, returns their average")
            return (s[0] + s[1]) / 2

    def set_fan_speed(self, ratio: float):
        for idx in {0, 1}:
            self.powerboard_pwm[0].set_fan_speed(idx, ratio)
