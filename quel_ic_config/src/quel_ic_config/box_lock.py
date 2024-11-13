from e7awghal import BoxLockDelegationError


class BoxLockError(Exception):
    pass


def guarded_by_box_lock(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BoxLockError as e:
            # Notes: the source of BoxLockError is apparent, it provides no valuable information.
            raise e.with_traceback(None)
        except BoxLockDelegationError as e:
            # Notes: the source of BoxLockDelegationError is apparent, it provides no valuable information.
            raise BoxLockError(*e.args) from None

    return wrapper
