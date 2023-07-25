import logging
from enum import IntEnum
from typing import Collection, List, Set, Tuple

import numpy as np
import numpy.typing as npt

from quel_inst_tool.spectrum_analyzer import SpectrumAnalyzer

logger = logging.getLogger(__name__)


class AbstractSpectrumPeak:
    def __init__(self, freq: float, power: float):
        self._freq: float = freq
        self._power: float = power
        self._freq_error_margin: float = 0.0

    def __repr__(self):
        return f"peak@({self._freq/1e6:.1f}MHz: {self._power:.1f}dBm)"

    def _freq_match(self, other: "AbstractSpectrumPeak") -> bool:
        return self._freq_diff(other) <= (self._freq_error_margin + other._freq_error_margin)

    def _freq_diff(self, other: "AbstractSpectrumPeak"):
        return abs(self._freq - other._freq)

    @property
    def freq(self):
        return self._freq

    @property
    def power(self):
        return self._power


class MeasuredSpectrumPeak(AbstractSpectrumPeak):
    @classmethod
    def from_spectrumanalyzer(cls, sa: SpectrumAnalyzer, minimum_power: float) -> List["MeasuredSpectrumPeak"]:
        return cls.from_numpy(sa.peak_get(minimum_power=minimum_power), sa.max_freq_error_get())

    @classmethod
    def from_spectrumanalyzer_with_trace(
        cls, sa: SpectrumAnalyzer, minimum_power: float
    ) -> Tuple[List["MeasuredSpectrumPeak"], npt.NDArray[np.float_]]:
        tr, pk = sa.trace_and_peak_get(minimum_power=minimum_power)
        return cls.from_numpy(pk, sa.max_freq_error_get()), tr

    @classmethod
    def from_numpy(cls, peaks: npt.NDArray[np.float_], max_freq_error: float) -> List["MeasuredSpectrumPeak"]:
        return [MeasuredSpectrumPeak(fp[0], fp[1], max_freq_error) for fp in peaks]

    def __init__(self, freq: float, power: float, freq_error_margin: float):
        super().__init__(freq, power)
        self._freq_error_margin = freq_error_margin
        if self._freq_error_margin < 0.0:
            raise ValueError("negative frequency error margin is not allowed")


class ExpectedSpectrumPeak(AbstractSpectrumPeak):
    def match_with_expected(self, other: "ExpectedSpectrumPeak", freq_error_margin: float) -> bool:
        # ExpextedSpectrumPeak doesn't keep measurement specific information. It should be given as arguments.
        if not isinstance(other, ExpectedSpectrumPeak):
            raise TypeError("a expected spectrum peak is required")
        return self._freq_diff(other) <= freq_error_margin

    def match_with_measurement(self, target: MeasuredSpectrumPeak) -> Tuple[bool, bool]:
        if not isinstance(target, MeasuredSpectrumPeak):
            raise TypeError("a measured spectrum peak is required")
        return self._freq_match(target), self._power <= target._power


class _SpectrumPeakCategory(IntEnum):
    DESIRED = 0
    TOO_WEAK = 1
    SPURIOUS = 2


class ExpectedSpectrumPeaks:
    def __init__(self, peaks: List[Tuple[float, float]]):
        """Checking the congruency of the expected spectrum peaks and actual spectrum peaks
        :param peaks: a set of pairs of expected frequency and minimum power of the peaks.
        """
        self._peaks: List[ExpectedSpectrumPeak] = [ExpectedSpectrumPeak(peak[0], peak[1]) for peak in peaks]

    def __repr__(self):
        return "{" + ", ".join([p.__repr__() for p in self._peaks]) + "}"

    def __getitem__(self, k):
        return self._peaks[k]

    def validate_with_measurement_condition(self, freq_error_margin) -> bool:
        valid = True
        # notes: the order of self._peaks should be preserved for the user's convenience for identifying missing
        #        expected peaks in the actual measurement data.
        sorted_peaks = sorted(self._peaks, key=lambda peak: peak.freq)
        for i in range(len(sorted_peaks) - 1):
            if sorted_peaks[i].match_with_expected(sorted_peaks[i + 1], freq_error_margin):
                valid = False
                logger.warning(f"peaks[{i}] and peaks[{i+1}] is indistiguishable under the given frequency margin")
        return valid

    def _matches1(self, measured: MeasuredSpectrumPeak) -> Tuple[_SpectrumPeakCategory, int]:
        matched: List[int] = []
        enough_power: bool = False
        for i, p in enumerate(self._peaks):
            freq_ok, power_ok = p.match_with_measurement(measured)
            if freq_ok:
                matched.append(i)
                enough_power = power_ok

        n_matched = len(matched)
        if n_matched == 0:
            return _SpectrumPeakCategory.SPURIOUS, -1
        elif n_matched == 1:
            if enough_power:
                return _SpectrumPeakCategory.DESIRED, matched[0]
            else:
                return _SpectrumPeakCategory.TOO_WEAK, matched[0]
        else:
            raise ValueError("a measured peak matches with multiple expected peaks, validation condition looks wrong.")

    def match(
        self, measured: Collection[MeasuredSpectrumPeak]
    ) -> Tuple[Tuple[bool, ...], List[MeasuredSpectrumPeak], List[MeasuredSpectrumPeak]]:
        matched: Set[int] = set()
        too_weak: List[MeasuredSpectrumPeak] = []
        spurious: List[MeasuredSpectrumPeak] = []
        for m in measured:
            cat, e = self._matches1(m)
            if cat == _SpectrumPeakCategory.SPURIOUS:
                spurious.append(m)
                logger.warning(f"unexpected peak of {m.power:.1f}dBm is detected at {m.freq/1e6:.1f}MHz")
            elif cat == _SpectrumPeakCategory.TOO_WEAK:
                too_weak.append(m)
                logger.warning(
                    f"too weak peak of {m.power:.1f}dBm (< {self._peaks[e].power:.1f}dBm) is "
                    f"detected at {m.freq/1e6:.1f}MHz"
                )
            elif cat == _SpectrumPeakCategory.DESIRED:
                if e in matched:
                    raise ValueError("multiple measured peak matches with the same expected peak")
                else:
                    matched.add(e)
            else:
                raise AssertionError

        desired_detection = tuple((i in matched) for i in range(len(self._peaks)))
        return desired_detection, spurious, too_weak

    def extract_matched(self, measured: Collection[MeasuredSpectrumPeak]) -> Set[MeasuredSpectrumPeak]:
        matched: Set[MeasuredSpectrumPeak] = set()
        for m in measured:
            cat, e = self._matches1(m)
            if cat == _SpectrumPeakCategory.DESIRED:
                matched.add(m)
        return matched
