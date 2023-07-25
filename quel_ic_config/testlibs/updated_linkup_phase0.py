import logging
from pathlib import Path
from typing import Tuple

from quel_ic_config import Quel1BoxType, Quel1ConfigObjects

logger = logging.getLogger(__name__)

# Notes: the total initialization of a QuEL-1 box will be handled by a specific method in Quel1ConfigObjects.
#        the following classes initializes a part of ICs step-by-step due to developmental reasons.


class Quel1ConfigSubSystemReferencePhase0Pre:
    def __init__(self, qco: Quel1ConfigObjects):
        self.qco = qco

    def phase0a(self) -> None:
        # initializes all DAC channel output to zero accordingly to settings/quel-1/ad5328.json.
        for idx in range(self.qco.NUM_IC["ad5328"]):
            logger.info(f"configuring {self.qco.ipaddr_css}:AD5328[{idx}]")
            self.qco.init_ad5328(idx)

    def phase0b(self) -> None:
        # turn all RF switches inside accordingly to settings/quel-1/rfswitch.json.
        for idx in range(self.qco.NUM_IC["gpio"]):
            logger.info(f"configuring {self.qco.ipaddr_css}:GPIO[{idx}]")
            self.qco.init_gpio(idx)

    def run(self, a: bool = True, b: bool = True) -> None:
        if a:
            self.phase0a()
        if b:
            self.phase0b()


class Quel1ConfigSubSystemReferencePhase0:
    def __init__(self, qco: Quel1ConfigObjects, group: int):
        self.qco = qco

        if group not in {0, 1}:
            raise ValueError(f"invalid group '{group}', must be either 0 or 1.")
        self.group = group

    def phase0a(self) -> None:
        logger.info(f"boxtype = {self.qco._boxtype}")  # TODO: refactor

        if self.group == 0:
            lmx2594_idx: Tuple[int, ...] = (0, 1, 2, 3, 8)
        elif self.group == 1:
            lmx2594_idx = (4, 5, 6, 7, 9)
        else:
            raise AssertionError

        for idx in lmx2594_idx:
            logger.info(f"configuring {self.qco.ipaddr_css}:LMX2594[{idx}]")
            self.qco.init_lmx2594(idx)

    def phase0b(self) -> None:
        if self.group == 0:
            adrf6780_idxs: Tuple[int, ...] = (0, 1, 2, 3)
        elif self.group == 1:
            adrf6780_idxs = (4, 5, 6, 7)
        else:
            raise AssertionError

        for idx in adrf6780_idxs:
            logger.info(f"configuring {self.qco.ipaddr_css}:ADRF6780[{idx}]")
            self.qco.init_adrf6780(idx)

    def run(self, a: bool = True, b: bool = True) -> None:
        if a:
            self.phase0a()
        if b:
            self.phase0b()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="configure ICs other than MxFE")
    parser.add_argument("--ipaddr_wss", type=str, required=False, default="non_existent", help="IGNORED")
    parser.add_argument(
        "--ipaddr_css", type=str, required=True, help="IP address of the configuration subsystem of the target box"
    )
    parser.add_argument(
        "--boxtype",
        type=str,
        choices=["quel1-a", "quel1-b", "qube-a", "qube-b"],
        required=True,
        help="a type of the target box: either of quel1-a, quel1-b, qube-a, or qube-b",
    )
    parser.add_argument("--monitor", action="store_true", help="IGNORED")
    parser.add_argument(
        "--config_root",
        type=Path,
        default="settings",
        help="path to configuration file root",
    )
    args = parser.parse_args()

    qco_ = Quel1ConfigObjects(args.ipaddr_css, Quel1BoxType.fromstr(args.boxtype), args.config_root)
    css_p0_pre = Quel1ConfigSubSystemReferencePhase0Pre(qco_)
    css_p0_g0 = Quel1ConfigSubSystemReferencePhase0(qco_, 0)
    css_p0_g1 = Quel1ConfigSubSystemReferencePhase0(qco_, 1)
    print("How to use:  css_p0_pre.run()  # initialize global settings")
    print("          :  css_p0_g0.run()  # initialize frequency converter subsystem of group 0")
    print("          :  css_p0_g1.run()  # initialize frequency converter subsystem of group 1")
