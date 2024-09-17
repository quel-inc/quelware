import logging

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from e7awghal.common_defs import _AWG_CHUNK_SIZE_UNIT_IN_SAMPLE, _AWG_MINIMUM_ALIGN
from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager, E7awgMemoryObj
from e7awghal.hbmctrl import HbmCtrl

logger = logging.getLogger(__name__)


def iq32v_to_complex64v(v: npt.NDArray[np.int16]) -> npt.NDArray[np.complex64]:
    return v[:, 0] + np.complex64(1j) * v[:, 1]


def complex64v_to_iq32v(v: npt.NDArray[np.complex64]) -> npt.NDArray[np.int16]:
    vr = np.round(v)
    re = np.array(vr.real, dtype=np.int16)
    im = np.array(vr.imag, dtype=np.int16)
    return np.vstack((re, im)).transpose()


class _WaveMemIq32:
    @classmethod
    def allocate(cls, mm: E7awgAbstractMemoryManager, size_in_word: int, **kwargs) -> "_WaveMemIq32":
        # Notes: 1 word = 4 samples = 16 bytes
        #        1 block = 16 words = 64 samples = 256 bytes
        rawobj = mm.allocate(size_in_word * 16, minimum_align=_AWG_MINIMUM_ALIGN, **kwargs)
        data = np.zeros((size_in_word * 4, 2), dtype=np.int16)
        return _WaveMemIq32(rawobj, data)

    def __init__(self, rawobj: E7awgMemoryObj, data: npt.NDArray[np.int16]):
        self._rawobj = rawobj
        self._data = data

    @property
    def physical_ptr(self) -> int:
        return self._rawobj._manager._address_offset + self._rawobj._address_top

    @property
    def size_in_word(self) -> int:
        return self.data.shape[0] // 4

    @property
    def size_in_sample(self) -> int:
        return self.data.shape[0]

    @property
    def size_in_byte(self) -> int:
        return self._rawobj._size

    @property
    def data(self) -> npt.NDArray[np.int16]:
        return self._data

    def get_data_as_complex64_vector(self) -> npt.NDArray[np.complex64]:
        return iq32v_to_complex64v(self.data)

    def set_data_from_complex64_vector(self, v: npt.NDArray[np.complex64]) -> None:
        if not isinstance(v, np.ndarray) and np.dtype == np.complex64:
            raise TypeError("data must be numpy array of complex64")
        if not (len(v.shape) == 1 and v.shape[0] == self.size_in_sample):
            raise ValueError("wrong shape of data")
        self.data[:, :] = complex64v_to_iq32v(v)


class WaveLibrary:
    def __init__(self, hbmctrl: HbmCtrl, mm: E7awgAbstractMemoryManager):
        self._hbmctrl = hbmctrl
        self._mm = mm
        self._lib: dict[str, _WaveMemIq32] = {}

    def register_wavedata_from_iq32vector(self, name: str, data: npt.NDArray[np.int16], **kwargs) -> None:
        if not (isinstance(name, str) and len(name) > 0):
            raise ValueError(f"invalid name of wavedata: '{name}'")

        if name in self._lib:
            raise ValueError(f"duplicated name of wavedata: '{name}'")

        if not (
            isinstance(data, np.ndarray) and np.dtype == np.int16 and len(data.shape) == 2 and (data.shape[1] == 2)
        ):
            raise TypeError("data must be numpy array of int16 pairs")

        size_in_sample = data.shape[0]
        if size_in_sample % _AWG_CHUNK_SIZE_UNIT_IN_SAMPLE == 0:
            raise ValueError(f"wrong length of data: ({size_in_sample} % {_AWG_CHUNK_SIZE_UNIT_IN_SAMPLE} != 0)")

        wavedata = _WaveMemIq32.allocate(mm=self._mm, size_in_word=size_in_sample // 4, **kwargs)
        wavedata.data[:, :] = data
        self._hbmctrl.write_iq32(wavedata.physical_ptr, size_in_sample, wavedata.data)
        self._lib[name] = wavedata

    def register_wavedata_from_complex64vector(self, name: str, data: npt.NDArray[np.complex64], **kwargs) -> None:
        if not (isinstance(name, str) and len(name) > 0):
            raise ValueError(f"invalid name of wavedata: '{name}'")

        if name in self._lib:
            raise ValueError(f"duplicated name of wavedata: '{name}'")

        if not (isinstance(data, np.ndarray) and data.dtype == np.complex64 and len(data.shape) == 1):
            raise TypeError("data must be one-dimensional numpy array of complex64")

        size_in_sample = data.shape[0]
        if size_in_sample % _AWG_CHUNK_SIZE_UNIT_IN_SAMPLE != 0:
            raise ValueError(f"wrong length of data: ({size_in_sample} % {_AWG_CHUNK_SIZE_UNIT_IN_SAMPLE} != 0)")

        wavedata = _WaveMemIq32.allocate(self._mm, size_in_word=size_in_sample // 4, **kwargs)
        wavedata.data[:] = complex64v_to_iq32v(data)
        self._hbmctrl.write_iq32(wavedata.physical_ptr, size_in_sample, wavedata.data)
        self._lib[name] = wavedata

    def has_wavedata(self, name: str) -> bool:
        return name in self._lib

    def get_names_of_wavedata(self) -> set[str]:
        return set(self._lib.keys())

    def get_pointer_to_wavedata(self, name: str) -> int:
        if not self.has_wavedata(name):
            raise ValueError(f"no wavedata named '{name}' is found")
        return self._lib[name].physical_ptr

    def get_size_in_word_of_wavedata(self, name: str) -> int:
        if not self.has_wavedata(name):
            raise ValueError(f"no wavedata named '{name}' is found")
        return self._lib[name].size_in_word

    def get_wavedata_as_complex64vector(self, name: str) -> npt.NDArray[np.complex64]:
        if not self.has_wavedata(name):
            raise ValueError(f"no wavedata named '{name}' is found")
        return iq32v_to_complex64v(self._lib[name].data)

    def get_wavedata_as_iq32vector(self, name: str) -> npt.NDArray[np.int16]:
        if not self.has_wavedata(name):
            raise ValueError(f"no wavedata named '{name}' is found")
        return np.array(self._lib[name].data)

    def delete_wavedata(self, name: str):
        if not self.has_wavedata(name):
            raise ValueError(f"no wavedata named '{name}' is found")
        del self._lib[name]


class WaveChunk(BaseModel, validate_assignment=True):
    name_of_wavedata: str = Field()
    num_blank_word: int = Field(ge=0x0000_0000, le=0xFFFF_FFFF, default=0x0000_0000)
    num_repeat: int = Field(ge=0x0000_0001, le=0xFFFF_FFFF, default=0x0000_0001)

    def __repr__(self):
        return (
            f"<WaveChunk -- {self.name_of_wavedata}, num_blank_word: {self.num_blank_word}, "
            f"repeat: {self.num_repeat}>"
        )


class AwgParam(BaseModel, validate_assignment=True):
    num_wait_word: int = Field(ge=0x0000_0000, le=0xFFFF_FFFF, default=0x0000_0000, multiple_of=16)
    num_repeat: int = Field(ge=0x0000_0001, le=0xFFFF_FFFF, default=0x0000_0001)
    # Notes: do not support the value other than 1.
    start_interval: int = Field(ge=0x0000_0001, le=0x0000_0001, default=0x0000_0001)
    # Notes: validation of length is postponed by the load.
    chunks: list[WaveChunk] = Field(default=[])

    @property
    def num_chunk(self) -> int:
        return len(self.chunks)
