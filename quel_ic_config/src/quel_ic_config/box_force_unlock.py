from quel_ic_config.exstickge_coap_client import AbstractSyncAsyncCoapClient
from quel_ic_config.exstickge_sock_client import AbstractLockKeeper


def force_unlock_all_boxes():
    AbstractLockKeeper.release_lock_all()
    AbstractSyncAsyncCoapClient.release_lock_all()
