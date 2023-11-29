import logging
from abc import ABCMeta
from typing import Tuple, Union

from pydantic import BaseModel, ConfigDict
from windfreak import SynthHD  # type: ignore

logger = logging.getLogger(__name__)


class SynthHDSweepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sweep_freq_low: Union[None, float] = None
    sweep_freq_high: Union[None, float] = None
    sweep_freq_step: Union[None, float] = None
    sweep_time_step: Union[None, float] = None
    sweep_power_high: Union[None, float] = None
    sweep_power_low: Union[None, float] = None
    sweep_direction: Union[None, int] = None
    sweep_type: Union[None, int] = None
    sweep_cont: Union[None, bool] = None

    def update_device_parameter(self, obj: "SynthHDChannel") -> bool:
        fields = SynthHDSweepParams.model_fields
        if not isinstance(fields, dict):
            raise RuntimeError("unexpected field data")
        for k in fields.keys():
            v0 = getattr(self, k)
            if v0 is None:
                continue
            setattr(obj, k, v0)
        return True

    @classmethod
    def from_synthHD(cls, obj: "SynthHDChannel") -> "SynthHDSweepParams":
        model = SynthHDSweepParams()
        fields = cls.model_fields
        if not isinstance(fields, dict):
            raise RuntimeError("unexpected field data")
        for k in fields.keys():
            setattr(model, k, getattr(obj, k))

        # TODO: validate the model before returning it.
        return model


class SynthHDChannel(metaclass=ABCMeta):
    __slots__ = (
        "_synth_channel",
        "_frequency",
        "_power",
        "_enable",
        "_sweep_freq_low",
        "_sweep_freq_high",
        "_sweep_freq_step",
        "_sweep_time_step",
        "_sweep_power_low",
        "_sweep_power_high",
        "_sweep_direction",
        "_sweep_type",
        "_sweep_cont",
        "_run_sweep",
    )

    def __init__(self, dev: SynthHD):
        self._synth_channel = dev
        self._frequency: Union[None, float] = None
        self._power: Union[None, float] = None
        self._enable: Union[None, bool] = None
        self._sweep_freq_low: Union[None, float] = None
        self._sweep_freq_high: Union[None, float] = None
        self._sweep_freq_step: Union[None, float] = None
        self._sweep_time_step: Union[None, float] = None
        self._sweep_power_low: Union[None, float] = None
        self._sweep_power_high: Union[None, float] = None
        self._sweep_direction: Union[None, int] = None
        self._sweep_type: Union[None, int] = None
        self._sweep_cont: Union[None, bool] = None
        self._run_sweep: Union[None, bool] = None

    def _cache_flush(self):
        self._frequency = None
        self._power = None
        self._enable = None
        self._sweep_freq_low = None
        self._sweep_freq_high = None
        self._sweep_freq_step = None
        self._sweep_time_step = None
        self._sweep_power_low = None
        self._sweep_power_high = None
        self._sweep_direction = None
        self._sweep_type = None
        self._sweep_cont = None
        self._run_sweep = None

    @property
    def frequency(self) -> float:  # in Hz
        if self._frequency is None:
            self._frequency = self._synth_channel.frequency
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: float) -> None:  # in Hz
        self._frequency = frequency
        self._synth_channel.frequency = self._frequency
        self._cache_flush()

    @property
    def power(self) -> float:  # in dB
        if self._power is None:
            self._power = self._synth_channel.power
        return self._power

    @power.setter
    def power(self, power: float) -> None:  # in dB
        self._power = power
        self._synth_channel.power = self._power
        self._cache_flush()

    @property
    def enable(self) -> bool:
        if self._enable is None:
            self._enable = self._synth_channel.enable
        return self._enable

    @enable.setter
    def enable(self, enable: bool) -> None:
        self._enable = enable
        self._synth_channel.enable = self._enable
        self._cache_flush()

    @property
    def sweep_freq_low(self) -> float:  # in Hz
        if self._sweep_freq_low is None:
            self._sweep_freq_low = self._synth_channel.read("sweep_freq_low") * 1.0e6
        return self._sweep_freq_low

    @sweep_freq_low.setter
    def sweep_freq_low(self, sweep_freq_low: float) -> None:  # in Hz
        self._sweep_freq_low = sweep_freq_low
        self._synth_channel.write("sweep_freq_low", self._sweep_freq_low * 1.0e-6)
        self._cache_flush()

    @property
    def sweep_freq_high(self) -> float:  # in Hz
        if self._sweep_freq_high is None:
            self._sweep_freq_high = self._synth_channel.read("sweep_freq_high") * 1.0e6
        return self._sweep_freq_high

    @sweep_freq_high.setter
    def sweep_freq_high(self, sweep_freq_high: float) -> None:  # in Hz
        self._sweep_freq_high = sweep_freq_high
        self._synth_channel.write("sweep_freq_high", self._sweep_freq_high * 1.0e-6)
        self._cache_flush()

    @property
    def sweep_freq_step(self) -> float:  # in Hz
        if self._sweep_freq_step is None:
            self._sweep_freq_step = self._synth_channel.read("sweep_freq_step") * 1.0e6
        return self._sweep_freq_step

    @sweep_freq_step.setter
    def sweep_freq_step(self, sweep_freq_step: float) -> None:  # in Hz
        self._sweep_freq_step = sweep_freq_step
        self._synth_channel.write("sweep_freq_step", self._sweep_freq_step * 1.0e-6)
        self._cache_flush()

    @property
    def sweep_time_step(self) -> float:  # [4,10000] millisecond
        if self._sweep_time_step is None:
            self._sweep_time_step = self._synth_channel.read("sweep_time_step")
        return self._sweep_time_step

    @sweep_time_step.setter
    def sweep_time_step(self, sweep_time_step: float) -> None:  # [4,10000] millisecond
        self._sweep_time_step = sweep_time_step
        self._synth_channel.write("sweep_time_step", self._sweep_time_step)
        self._cache_flush()

    @property
    def sweep_power_low(self) -> float:  # Sweep lower power [-60,+20] dBm
        if self._sweep_power_low is None:
            self._sweep_power_low = self._synth_channel.read("sweep_power_low")
        return self._sweep_power_low

    @sweep_power_low.setter
    def sweep_power_low(self, sweep_power_low: float) -> None:  # Sweep lower power [-60,+20] dBm
        self._sweep_power_low = sweep_power_low
        self._synth_channel.write("sweep_power_low", self._sweep_power_low)
        self._cache_flush()

    @property
    def sweep_power_high(self) -> float:  # Sweep higher power [-60,+20] dBm
        if self._sweep_power_high is None:
            self._sweep_power_high = self._synth_channel.read("sweep_power_high")
        return self._sweep_power_high

    @sweep_power_high.setter
    def sweep_power_high(self, sweep_power_high: float) -> None:  # Sweep higher power [-60,+20] dBm
        self._sweep_power_high = sweep_power_high
        self._synth_channel.write("sweep_power_high", self._sweep_power_high)
        self._cache_flush()

    @property
    def sweep_direction(self) -> int:  # up:1, down:0
        if self._sweep_direction is None:
            self._sweep_direction = self._synth_channel.read("sweep_direction")
        return self._sweep_direction

    @sweep_direction.setter
    def sweep_direction(self, sweep_direction: int) -> None:  # up:1, down:0
        self._sweep_direction = sweep_direction
        self._synth_channel.write("sweep_direction", self._sweep_direction)
        self._cache_flush()

    @property
    def sweep_type(self) -> int:  # linear:0, tabular:
        if self._sweep_type is None:
            self._sweep_type = self._synth_channel.read("sweep_type")
        return self._sweep_type

    @sweep_type.setter
    def sweep_type(self, sweep_type: int) -> None:  # linear:0, tabular:
        self._sweep_type = sweep_type
        self._synth_channel.write("sweep_type", self._sweep_type)
        self._cache_flush()

    @property
    def sweep_cont(self) -> bool:  # Continuous:True, Single:False
        if self._sweep_cont is None:
            self._sweep_cont = self._synth_channel.read("sweep_cont")
        return self._sweep_cont

    @sweep_cont.setter
    def sweep_cont(self, sweep_cont: bool) -> None:
        self._sweep_cont = sweep_cont
        self._synth_channel.write("sweep_cont", self._sweep_cont)
        self._cache_flush()

    @property
    def run_sweep(self) -> bool:  # start:True, stop:False
        if self._run_sweep is None:
            self._run_sweep = self._synth_channel.read("sweep_single")
        return self._run_sweep

    @run_sweep.setter
    def run_sweep(self, run_sweep: bool) -> None:
        self._run_sweep = run_sweep
        self._synth_channel.write("sweep_single", self._run_sweep)
        self._cache_flush()


class SynthHDMaster(metaclass=ABCMeta):
    def __init__(self):
        self._synth = SynthHD("/dev/ttyACM0")
        self._synth.init()
        self.channel: Tuple[SynthHDChannel, ...] = tuple(SynthHDChannel(self._synth[i]) for i in range(2))
