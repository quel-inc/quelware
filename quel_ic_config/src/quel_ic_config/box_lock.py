from e7awghal import BoxLockDelegationError

_trancated_traceback_enable: bool = True


class BoxLockError(Exception):
    pass


def set_trancated_traceback_for_lock_error(v: bool):
    global _trancated_traceback_enable
    _trancated_traceback_enable = v


def guarded_by_box_lock(func):
    def wrapper(*args, **kwargs):
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
