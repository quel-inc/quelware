import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from e7awgsw import AWG, AwgCtrl, CaptureCtrl, CaptureModule, CaptureParam

from quel_ic_config import Quel1BoxType, Quel1ConfigObjects

logger = logging.getLogger(__name__)


class Quel1ConfigSubSystemReferencePhase1:
    def __init__(self, wss_addr: str, qco: Quel1ConfigObjects, group: int):
        self.wss_addr = wss_addr
        self.awg_ctrl = AwgCtrl(self.wss_addr)
        self.cap_ctrl = CaptureCtrl(self.wss_addr)

        self.qco: Quel1ConfigObjects = qco

        self.group = group
        if self.group == 0:
            self.awgs: AWG = [AWG.U15, AWG.U14, AWG.U13, AWG.U12, AWG.U11, AWG.U10, AWG.U9, AWG.U8]
            self.cpmd: CaptureModule = CaptureModule.U1
        elif self.group == 1:
            self.awgs = [AWG.U7, AWG.U6, AWG.U5, AWG.U4, AWG.U3, AWG.U2, AWG.U1, AWG.U0]
            self.cpmd = CaptureModule.U0
        else:
            raise ValueError(f"invalid group '{group}', must be either 0 or 1.")

    def phase1a(self):
        self.awg_ctrl.initialize(*self.awgs)  # TODO: clarify what initialize() do.
        self.awg_ctrl.terminate_awgs(*self.awgs)  # TODO: clarify this is included in the above function or not.
        cap_units = CaptureModule.get_units(self.cpmd)
        self.cap_ctrl.initialize(*cap_units)

    def phase1b(self) -> None:
        logger.info(f"closing all the RF switches of group {self.group}")
        _, helper = self.qco.gpio[0]
        helper.write_field(f"Group{self.group}", path0=True, path1=True, path2=True, path3=True, monitor=True)
        helper.flush()

    def phase1c(self, reset, combine_01=False, combine_23=False) -> bool:
        self.qco.ad9082[self.group].initialize(reset=reset)

        """
        # XXX: will be moved into startup_tx() of ad9081 (this setting is not specific to our boxes)
        dac_select = ad9081.DAC_MODE_SWITCH_GROUP_NONE
        if combine_01:
            dac_select = ad9081.DAC_MODE_SWITCH_GROUP_0
        if combine_23:
            dac_select = ad9081.DAC_MODE_SWITCH_GROUP_1
        if combine_01 & combine_23:
            dac_select = ad9081.DAC_MODE_SWITCH_GROUP_ALL
        ad9081.dac_mode_set(self.ad9082.device, dac_select, ad9081.DAC_MODE_3)
        """

        self.qco.ad9082[self.group].dump_jesd_status()

        link_valid = self.qco.ad9082[self.group].check_link_status()
        if not link_valid:
            logger.warning("link-up failed!")

        return link_valid

    def phase1d(self, num_words: int = 1024, timeout: float = 0.5) -> Dict[int, npt.NDArray[np.complexfloating]]:
        cprm = CaptureParam()
        cprm.num_integ_sections = 1
        cprm.add_sum_section(num_words=num_words, num_post_blank_words=1)
        cprm.capture_delay = 100

        cpuns = CaptureModule.get_units(self.cpmd)
        self.cap_ctrl.initialize(*cpuns)
        for cpun in cpuns:
            # notes: common capture parameter is shared by all the capture units
            self.cap_ctrl.set_capture_params(cpun, cprm)

        self.cap_ctrl.start_capture_units(*cpuns)
        self.cap_ctrl.wait_for_capture_units_to_stop(timeout, *cpuns)
        self.cap_ctrl.check_err(*cpuns)

        data: Dict[int, npt.NDArray[np.complexfloating]] = {}
        for cpun in cpuns:
            n = self.cap_ctrl.num_captured_samples(cpun)
            logger.info(f"the capture unit {int(cpun)} captured {n} samples")
            data_in_assq: List[Tuple[float, float]] = self.cap_ctrl.get_capture_data(cpun, n)
            tmp = np.array(data_in_assq)
            data[cpun] = tmp[:, 0] + tmp[:, 1] * 1j
        return data

    def phase1e(self, cpdt: Dict[int, npt.NDArray[np.complexfloating]], unit_idx: int, thr: float = 400.0):
        cpun = CaptureModule.get_units(self.cpmd)[unit_idx]
        data = cpdt[cpun]

        amp_max = max(abs(data))
        logger.info(f"the maximum amplitude of the captured data is {amp_max:.1f}")
        return amp_max < thr

    def run(
        self,
        a=True,
        b=True,
        c=True,
        d=True,
        e=True,
        reset=True,
        max_trial=10,
        combine_01=False,
        combine_23=False,
    ) -> Tuple[bool, List[Dict[int, npt.NDArray[np.complexfloating]]]]:
        data_list = []
        judge = False

        if a:
            self.phase1a()
        if b:
            self.phase1b()

        for i in range(max_trial):
            if c:
                flag = self.phase1c(reset, combine_01, combine_23)
            else:
                flag = True

            if flag:
                if d:
                    data = self.phase1d(16384)
                    data_list.append(data)
                else:
                    break

                if e:
                    judge = self.phase1e(data, 0)
                else:
                    logger.info("note that no check of captured data is conducted.")
                    judge = True

                if judge:
                    break

        return judge, data_list


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser(description="configure ICs other than MxFE")
    parser.add_argument(
        "--ipaddr_wss",
        type=str,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box",
    )
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
    parser.add_argument(
        "--config_root",
        type=Path,
        default="settings",
        help="path to configuration file root",
    )
    parser.add_argument("--monitor", action="store_true", help="IGNORED")
    args = parser.parse_args()

    qco_ = Quel1ConfigObjects(args.ipaddr_css, Quel1BoxType.fromstr(args.boxtype), args.config_root)
    css_p1_g0 = Quel1ConfigSubSystemReferencePhase1(args.ipaddr_wss, qco_, 0)
    css_p1_g1 = Quel1ConfigSubSystemReferencePhase1(args.ipaddr_wss, qco_, 1)
    print("How to use:  judge, data_list = css_p1_g0.run()  # for group 0")
    print("          :  judge, data_list = css_p1_g1.run()  # for group 1")
