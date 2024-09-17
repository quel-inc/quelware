import logging

from e7awghal import AbstractQuel1Au50Hal, ClockmasterAu200Hal, create_quel1au50hal


def show():
    global boxes, cmhal

    print(cmhal.name, (cmhal.ctrl.read_counter(),))
    for box in boxes:
        print(box.name, box.clkcntr.read_counter())


def sync():
    global boxes, cmhal
    syncintfs = {box.syncintf for box in boxes}
    cmhal.ctrl.kick_sync(syncintfs)
    show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxes: set[AbstractQuel1Au50Hal] = {
        create_quel1au50hal(ipaddr_wss=a) for a in ("10.1.0.74", "10.1.0.58", "10.1.0.60")
    }
    cmhal = ClockmasterAu200Hal(ipaddr="10.3.0.13")
    show()
