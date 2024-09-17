class DeviceLockException(Exception):
    pass


def guarded_by_device_lock(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DeviceLockException as e:
            # Notes: the source of DeviceLockException is apparent, it provides no valuable information.
            raise e.with_traceback(None)

    return wrapper
