from typing import Callable, TypeVar

from e7awghal import BoxLockDelegationError
from typing_extensions import ParamSpec

_trancated_traceback_enable: bool = True


class BoxLockError(Exception):
    pass


def set_trancated_traceback_for_lock_error(v: bool):
    global _trancated_traceback_enable
    _trancated_traceback_enable = v


Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def guarded_by_box_lock(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        try:
            return func(*args, **kwargs)
        except BoxLockError as e:
            # Notes: the source of BoxLockError is apparent, it provides no valuable information.
            if _trancated_traceback_enable:
                raise e.with_traceback(None)
            else:
                raise e
        except BoxLockDelegationError as e:
            # Notes: the source of BoxLockDelegationError is apparent, it provides no valuable information.
            if _trancated_traceback_enable:
                raise BoxLockError(*e.args) from None
            else:
                raise e

    return wrapper
