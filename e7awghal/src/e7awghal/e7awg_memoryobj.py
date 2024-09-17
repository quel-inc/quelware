import bisect
import logging
from abc import ABCMeta, abstractmethod
from threading import Lock
from weakref import WeakSet

from e7awghal.common_defs import E7awgMemoryError

logger = logging.getLogger(__name__)


class E7awgAbstractMemoryManager(metaclass=ABCMeta):
    def __init__(self, name: str, address_top: int, size: int):
        self._name = name
        self._address_offset = address_top
        self._size = size

        self._freelist: dict[int, E7awgMemoryObj] = {0: E7awgMemoryObj(0, self._size, self, False)}
        self._freekey: list[int] = [0]
        self._issued_obj: WeakSet = WeakSet()
        self._lock = Lock()

    def reset(self):
        with self._lock:
            for ref in self._issued_obj:
                obj = ref
                if obj is not None:
                    if obj._live:
                        obj._live = False
            self._issued_obj.clear()
            self._freelist = {0: E7awgMemoryObj(0, self._size, self, False)}
            self._freekey = [0]

    def _add_freeobj(self, obj: "E7awgMemoryObj"):
        bisect.insort_left(self._freekey, obj._address_top)
        self._freelist[obj._address_top] = obj

    def _take_freeobj(self, idx: int) -> "E7awgMemoryObj":
        k = self._freekey.pop(idx)
        obj = self._freelist.pop(k)
        return obj

    def _borrow_freeobj(self, idx: int) -> "E7awgMemoryObj":
        return self._freelist[self._freekey[idx]]

    def _allocate_liveobj(self, address_top: int, size: int) -> "E7awgMemoryObj":
        # Notes: lock should be acquired by the caller
        idx = bisect.bisect_right(self._freekey, address_top)
        if idx > 0:
            if self._borrow_freeobj(idx - 1).contains(address_top, size):
                obj0 = self._take_freeobj(idx - 1)
                newobj, obj1 = obj0.split(address_top, size)
                for o in obj1:
                    self._add_freeobj(o)
                self._issued_obj.add(newobj)
                return newobj

        raise ValueError(f"memory region {address_top}:{address_top+size} is not available")

    def _deallocate_liveobj(self, obj: "E7awgMemoryObj"):
        assert obj._live, "deallocation is attempted on a dead memory object"
        with self._lock:
            idx_pre = bisect.bisect_right(self._freekey, obj._address_top) - 1
            merge_pre = False
            if idx_pre >= 0:
                pre_ref = self._borrow_freeobj(idx_pre)
                merge_pre = pre_ref._address_top + pre_ref._size == obj._address_top

            idx_post = bisect.bisect_left(self._freekey, obj._address_top + obj._size)
            merge_post = False
            if idx_post < len(self._freelist):
                post_ref = self._borrow_freeobj(idx_post)
                merge_post = obj._address_top + obj._size == post_ref._address_top

            if merge_pre:
                pre_ref = self._borrow_freeobj(idx_pre)
                if merge_post:
                    post = self._take_freeobj(idx_post)
                    pre_ref.merge(obj)
                    pre_ref.merge(post)
                else:
                    pre_ref.merge(obj)
            else:
                if merge_post:
                    post = self._take_freeobj(idx_post)
                    post.merge(obj)
                    self._add_freeobj(post)
                else:
                    self._add_freeobj(E7awgMemoryObj(obj._address_top, obj._size, self, False))
            obj._live = False
            # Notes: WeakSet eliminated dead object automatically.

    @abstractmethod
    def allocate(self, size: int, **kwargs) -> "E7awgMemoryObj":
        pass


class E7awgMemoryObj:
    __slots__ = ("_address_top", "_size", "_manager", "_live", "__weakref__")

    def __init__(self, address_top: int, size: int, manager: E7awgAbstractMemoryManager, live: bool):
        self._address_top = address_top
        self._size = size
        self._manager = manager
        self._live: bool = live

    def __del__(self):
        if self._live:
            self._manager._deallocate_liveobj(self)

    def __repr__(self):
        return f"<{'livemem' if self._live else 'freemem'} {self._address_top:09x}--{self._address_top+self._size:09x}>"

    @property
    def address_top(self) -> int:
        if self._live:
            return self._address_top
        else:
            raise RuntimeError("memory object is invalidated")

    def contains(self, addr_top: int, sz: int) -> bool:
        if addr_top < 0:
            raise ValueError(f"invalid address: {addr_top}")
        if sz <= 0:
            raise ValueError(f"invalid size: {sz}")
        return (0 <= addr_top - self._address_top < self._size) and (addr_top - self._address_top + sz <= self._size)

    def split(self, addr_top: int, sz: int) -> tuple["E7awgMemoryObj", tuple["E7awgMemoryObj", ...]]:
        # Notes: should confirm that self.contains(addr_top, sz) is True in advance.
        if addr_top == self._address_top and sz == self._size:
            self._live = True
            return self, ()
        else:
            new = E7awgMemoryObj(addr_top, sz, self._manager, True)

        if addr_top == self._address_top:
            self._address_top += sz
            self._size -= sz
            return new, (self,)
        elif addr_top + sz == self._address_top + self._size:
            self._size -= sz
            return new, (self,)
        else:
            pre = E7awgMemoryObj(self._address_top, addr_top - self._address_top, self._manager, False)
            orig_end = self._address_top + self._size
            self._address_top = addr_top + sz
            self._size = orig_end - self._address_top
            return new, (pre, self)

    def merge(self, others: "E7awgMemoryObj") -> bool:
        if others._address_top + others._size == self._address_top:
            self._address_top = others._address_top
            self._size += others._size
            return True
        elif self._address_top + self._size == others._address_top:
            self._size += others._size
            return True
        else:
            return False


class E7awgPrimitiveMemoryManager(E7awgAbstractMemoryManager):
    def _validate_alignment(self, address_top: int, **kwargs):
        if "minimum_align" in kwargs:
            if (address_top % kwargs["minimum_align"]) != 0:
                raise ValueError(
                    f"address_top (= {kwargs['address_top']:09x}) is not aligned to "
                    f"{kwargs['minimum_align']}-byte boundary"
                )

    def _find_first_free(self, size: int, **kwargs) -> int:
        minimum_align = kwargs["minimum_align"] if "minimum_align" in kwargs else 1

        for i, _ in enumerate(self._freekey):
            obj_ref = self._borrow_freeobj(i)
            top0 = obj_ref._address_top
            top1 = (top0 + minimum_align - 1) // minimum_align * minimum_align
            if obj_ref._size - (top1 - top0) >= size:
                return top1

        raise E7awgMemoryError(f"failed to acquire {size} byte in '{self._name}'")

    def allocate(self, size: int, **kwargs) -> E7awgMemoryObj:
        if size <= 0:
            raise ValueError(f"invalid size (= {size})")

        with self._lock:
            # TODO: reconsider the right place to write constraint
            if "address_top" in kwargs:
                address_top: int = kwargs["address_top"]
                self._validate_alignment(**kwargs)
            else:
                address_top = self._find_first_free(size, **kwargs)

            return self._allocate_liveobj(address_top, size)
