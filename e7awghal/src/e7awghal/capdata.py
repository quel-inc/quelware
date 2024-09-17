import logging
from typing import Union

import numpy as np
import numpy.typing as npt

from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager, E7awgMemoryObj
from e7awghal.hbmctrl import HbmCtrl

logger = logging.getLogger(__name__)


class CapIqParser:
    def __init__(
        self,
        *,
        total_size_in_sample: int,
        num_repeat: int,
        section_sizes_in_sample: tuple[int, ...],
        section_names: Union[tuple[str, ...], None],
        total_exponent_offset: int,
        classification_enable: bool,
    ):
        self._total_size_in_sample = total_size_in_sample
        self._num_repeat: int = num_repeat
        self._section_sizes_in_sample: tuple[int, ...] = section_sizes_in_sample
        self._section_names: Union[tuple[str, ...], None] = section_names
        self._total_exponent_offset = total_exponent_offset
        self._classification_enable = classification_enable

    @property
    def total_size_in_sample(self) -> int:
        return self._total_size_in_sample

    @property
    def classification_enable(self) -> bool:
        return self._classification_enable

    def parse_wave_as_dict(self, iqdata: npt.NDArray[np.complex64]) -> dict[str, npt.NDArray[np.complex64]]:
        if self._classification_enable:
            raise RuntimeError("no wave data because classification is enabled")
        if self._section_names is None:
            raise ValueError(
                "no valid list of section names are available, use parse_as_list() instead (it looks like that names "
                "are crashed among the sections)"
            )
        if len(iqdata) != self._total_size_in_sample:
            raise ValueError(
                f"the size of the given data (= {len(iqdata)}) is different "
                f"from the expected (= {self._total_size_in_sample})"
            )
        v = {
            name: np.zeros((self._num_repeat, size_in_sample), dtype=np.complex64)
            for name, size_in_sample in zip(self._section_names, self._section_sizes_in_sample)
        }

        pos = 0
        for i in range(self._num_repeat):
            for name, size_in_sample in zip(self._section_names, self._section_sizes_in_sample):
                v[name][i, :] = iqdata.data[pos : pos + size_in_sample]
                pos += size_in_sample

        if self._total_exponent_offset != 0:
            for name in self._section_names:
                v[name] /= 1 << self._total_exponent_offset

        return v

    def parse_wave_as_list(self, iqdata: npt.NDArray[np.complex64]) -> list[npt.NDArray[np.complex64]]:
        if self._classification_enable:
            raise RuntimeError("no wave data because classification is enabled")
        if len(iqdata) != self._total_size_in_sample:
            raise ValueError(
                f"the size of the given data (= {len(iqdata)}) is different "
                f"from the expected (= {self._total_size_in_sample})"
            )
        v = [
            np.zeros((self._num_repeat, size_in_sample), dtype=np.complex64)
            for size_in_sample in self._section_sizes_in_sample
        ]

        pos = 0
        for i in range(self._num_repeat):
            for j, size_in_sample in enumerate(self._section_sizes_in_sample):
                v[j][i, :] = iqdata.data[pos : pos + size_in_sample]
                pos += size_in_sample

        if self._total_exponent_offset != 0:
            for j in range(len(self._section_sizes_in_sample)):
                v[j] /= 1 << self._total_exponent_offset

        return v

    def parse_class_as_dict(self, classdata: npt.NDArray[np.uint64]) -> dict[str, npt.NDArray[np.uint8]]:
        if not self._classification_enable:
            raise RuntimeError("no class data because classification is disabled")
        if self._section_names is None:
            raise ValueError(
                "no valid list of section names are available, use parse_as_list() instead (it looks like that names "
                "are crashed among the sections)"
            )
        if len(classdata) != np.ceil(self._total_size_in_sample / 32):
            raise ValueError(
                f"the size of the given data (= {len(classdata)}) is different "
                f"from the expected (= {np.ceil(self._total_size_in_sample / 32)})"
            )

        p = np.zeros(self._total_size_in_sample, dtype=np.uint8)
        v = {
            name: np.zeros((self._num_repeat, size_in_sample), dtype=np.uint8)
            for name, size_in_sample in zip(self._section_names, self._section_sizes_in_sample)
        }

        ii = 0
        for i in range(len(classdata)):
            x = classdata.data[i]
            for j in range(min(self._total_size_in_sample - i * 32, 32)):
                p[ii] = x & 0x3
                x >>= 2
                ii += 1

        pos = 0
        for i in range(self._num_repeat):
            for name, size_in_sample in zip(self._section_names, self._section_sizes_in_sample):
                v[name][i, :] = p.data[pos : pos + size_in_sample]
                pos += size_in_sample

        return v

    def parse_class_as_list(self, classdata: npt.NDArray[np.uint64]) -> list[npt.NDArray[np.uint8]]:
        if not self._classification_enable:
            raise RuntimeError("no class data because classification is disabled")
        if len(classdata) != np.ceil(self._total_size_in_sample / 32):
            raise ValueError(
                f"the size of the given data (= {len(classdata)}) is different "
                f"from the expected (= {np.ceil(self._total_size_in_sample / 32)})"
            )
        p = np.zeros(self._total_size_in_sample, dtype=np.uint8)
        v = [
            np.zeros((self._num_repeat, size_in_sample), dtype=np.uint8)
            for size_in_sample in self._section_sizes_in_sample
        ]

        ii = 0
        for i in range(len(classdata)):
            x = classdata.data[i]
            for j in range(min(self._total_size_in_sample - i * 32, 32)):
                p[ii] = x & 0x3
                x >>= 2
                ii += 1
        pos = 0
        for i in range(self._num_repeat):
            for j, size_in_sample in enumerate(self._section_sizes_in_sample):
                v[j][i, :] = p.data[pos : pos + size_in_sample]
                pos += size_in_sample

        return v


class CapIqDataReader:
    __slots__ = (
        "_parser",
        "_mobj",
        "_wavedata",
        "_classdata",
        "_mm",
        "_hbmctrl",
    )

    def __init__(self, parser: CapIqParser, mobj: E7awgMemoryObj, mm: E7awgAbstractMemoryManager, hbmctrl: HbmCtrl):
        self._parser: CapIqParser = parser
        self._mobj: Union[E7awgMemoryObj, None] = mobj
        self._wavedata: Union[npt.NDArray[np.complex64], None] = None
        self._classdata: Union[npt.NDArray[np.uint64], None] = None
        self._mm = mm
        self._hbmctrl = hbmctrl

    @property
    def total_size_in_sample(self) -> int:
        return self._parser.total_size_in_sample

    def read(self) -> None:
        if self._mobj is not None:
            if self._parser.classification_enable:
                self._classdata = self._hbmctrl.read_u64(
                    self._mm._address_offset + self._mobj.address_top, int(np.ceil(self._mobj._size / 8))
                )
            else:
                self._wavedata = self._hbmctrl.read_iq64(
                    self._mm._address_offset + self._mobj.address_top, self._mobj._size // 8
                )
            self._mobj = None

    def rawwave(self) -> npt.NDArray[np.complex64]:
        if self._parser.classification_enable:
            raise RuntimeError("no wave data is available because classification is enabled")
        if self._wavedata is None:
            self.read()
            assert self._wavedata is not None  # Notes: for mypy
        return self._wavedata

    def as_wave_dict(self) -> dict[str, npt.NDArray[np.complex64]]:
        if self._parser.classification_enable:
            raise RuntimeError("no wave data is available because classification is enabled")
        if self._wavedata is None:
            self.read()
            assert self._wavedata is not None  # Notes: for mypy
        return self._parser.parse_wave_as_dict(self._wavedata)

    def as_wave_list(self) -> list[npt.NDArray[np.complex64]]:
        if self._parser.classification_enable:
            raise RuntimeError("no wave data is available because classification is enabled")
        if self._wavedata is None:
            self.read()
            assert self._wavedata is not None  # Notes: for mypy
        return self._parser.parse_wave_as_list(self._wavedata)

    def rawclass(self) -> npt.NDArray[np.uint64]:
        if not self._parser.classification_enable:
            raise RuntimeError("no class data is available because classification is disabled")
        if self._classdata is None:
            self.read()
            assert self._classdata is not None  # Notes: for mypy
        return self._classdata

    def as_class_dict(self) -> dict[str, npt.NDArray[np.uint8]]:
        if not self._parser.classification_enable:
            raise RuntimeError("no class data is available because classification is disabled")
        if self._classdata is None:
            self.read()
            assert self._classdata is not None  # Notes: for mypy
        return self._parser.parse_class_as_dict(self._classdata)

    def as_class_list(self) -> list[npt.NDArray[np.uint8]]:
        if not self._parser.classification_enable:
            raise RuntimeError("no class data is available because classification is disabled")
        if self._classdata is None:
            self.read()
            assert self._classdata is not None  # Notes: for mypy
        return self._parser.parse_class_as_list(self._classdata)
