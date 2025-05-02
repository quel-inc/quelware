import argparse
import json
import logging
import pprint
import time
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import requests
import yaml
from quel_pyxsdb import get_jtagterminal_port

from quel_cmod_scripting import FanControlCmod, QuelCmod, SwitchControlCmod

logger = logging.getLogger(__name__)

DEFAULT_XSDB_CMOD_PORT = 36335
DEFAULT_HWSVR_CMOD_PORT = 6121
DEFAULT_LOGSTASH_URL = ""
# DEFAULT_LOGSTASH_URL = "http://localhost:5001"
DEFAULT_BOX_ID = "quel#0-00"
HEADERS = {"Content-Type": "application/json"}
# be aware that max voltage of CP30238 is 8.6V, corresponding to 450 or around.
# for continuous operation 200 seems to be maximum value, based on morisaka-san's code.
DEFAULT_PELTIER_ON_VALUE = 200
DEFAULT_INVERTED_PELTIER_ON_VALUE = 200
DEFAULT_HEATER_ON_VALUE = 300


def parse_host_port(args) -> Tuple[str, int]:
    hp = args.con

    if ":" in hp:
        h, p_s = hp.split(":")
    else:
        h = "localhost"
        p_s = hp
    p = int(p_s)

    if p == 0:
        with open(f".config/{args.box}.yml", "r") as f:
            config = yaml.safe_load(f)
        p = get_jtagterminal_port(
            adapter_id=config["adapter_cmod"],
            host=h,
            xsdb_port=DEFAULT_XSDB_CMOD_PORT,
            hwsvr_port=DEFAULT_HWSVR_CMOD_PORT,
        )
        logger.info(f"auto-detected port of the jtag terminal is {p}")

    return h, p


def common_parser():
    parser = argparse.ArgumentParser(
        prog="upd_temp",
        description="capture thermistor reading from cmod and send it to elasticsearch",
    )
    parser.add_argument(
        "con",
        type=str,
        help="telnet port of cmod, specify 0 for automatic resolution of port from name of box",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_LOGSTASH_URL,
        help="url of logstash pipeline input",
    )
    parser.add_argument(
        "--box", type=str, default=DEFAULT_BOX_ID, help="url of logstash pipeline input"
    )
    return parser


def array2dict(a: npt.NDArray[np.int32]) -> Dict[str, int]:
    cj = {}
    for i in range(len(a)):
        cj[f"th{i:02d}"] = int(a[i])  # numpy.int32 is not JSON serializable
    return cj


def send_telemetry(convd: Dict[str, int], box: str, url: str):
    sent: Dict[str, Any] = dict(convd)
    sent["box_id"] = box
    sent["time"] = int(time.time() * 1000 + 0.5)
    response = requests.post(url, headers=HEADERS, data=json.dumps(sent))
    return response.status_code


def show_th() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = QuelCmod(con[0], con[1])
    data = obj.thall_in_json()
    if data is None:
        return 1
    else:
        pprint.pprint(data)
        if args.url != "":
            sc = send_telemetry(data, args.box, args.url)
            if sc != 200:
                logger.error(f"return status of logstash is {sc} (!= 200)")
                return 1
        return 0


def show_pl() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = QuelCmod(con[0], con[1])
    data = obj.plstat_in_json()
    if data is None:
        return 1
    else:
        pprint.pprint(data)
        if args.url != "":
            sc = send_telemetry(data, args.box, args.url)
            if sc != 200:
                logger.error(f"return status of logstash is {sc} (!= 200)")
                return 1
        return 0


def stop_tmp_control() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = QuelCmod(con[0], con[1])
    _ = obj.execute("mode xp")
    rep2 = obj.execute("mode")
    current_mode = rep2.strip().split()[-1]
    if current_mode != "xp":
        logger.error(f"failed to enter into 'xp' mode. (mode = '{current_mode}')")
        return 1

    _ = obj.execute("plalloff!")
    data = obj.plstat_in_json()
    if data is None:
        logger.error("failed to execute a command.")
        return 1

    for k, v in data.items():
        if v != 0:
            logger.error("failed to turn off some Peltier devices and heater")
            return 1

    return 0


def restart_tmp_control() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = QuelCmod(con[0], con[1])
    _ = obj.execute("mode xp")
    time.sleep(1)
    _ = obj.execute("mode wup")
    rep2 = obj.execute("mode")
    current_mode = rep2.strip().split()[-1]
    if current_mode != "wup":
        logger.error(f"failed to enter into 'wup' mode. (mode = '{current_mode})'")
        return 1

    return 0


def show_fan() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = FanControlCmod(con[0], con[1])
    data = obj.fan_get_in_json()
    if data is None:
        return 1
    else:
        pprint.pprint(data)
        if args.url != "":
            sc = send_telemetry(data, args.box, args.url)
            if sc != 200:
                logger.error(f"return status of logstash is {sc} (!= 200)")
                return 1
        return 0


def show_switches() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    obj = SwitchControlCmod(con[0], con[1])
    data = obj.switch_get_in_json()
    if data is not None:
        pprint.pprint(data)
        return 0
    else:
        return 1


def set_switch() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    parser.add_argument("--channel", type=int, help="channel index")
    parser.add_argument("ctrl", type=int, help="1 (ON) or 0 (OFF)")
    args = parser.parse_args()
    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    if args.ctrl != 1 and args.ctrl != 0:
        raise ValueError("need to specify the channel state with 0 or 1")

    obj = SwitchControlCmod(con[0], con[1])
    ret = obj.switch_set(int(args.channel), bool(args.ctrl))
    if ret is True:
        return 0
    else:
        return 1


def set_all_switches() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    parser.add_argument("ctrl", type=str, help="specify 0 - 15")
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1
    all_ctrl = int(args.ctrl, 16) if "0x" in args.ctrl else int(args.ctrl)
    if all_ctrl < 0 or all_ctrl > 15:
        raise ValueError("need to specify all channel states using 0 - 15")

    obj = SwitchControlCmod(con[0], con[1])
    ret = obj.all_switches_set(all_ctrl)
    if ret is True:
        return 0
    else:
        return 1


def capture_and_send(obj, args):
    data1 = obj.thall()
    if data1 is None:
        logger.error("failed to get thermistor readings")
        return None
    if args.url != "":
        sc = send_telemetry(array2dict(data1), args.box, args.url)
        if sc != 200:
            logger.warning("failed to update data on dashboard")
    return data1


def waiting(obj, wait, args, take_min=False):
    data0 = capture_and_send(obj, args)
    for _ in range(wait):
        time.sleep(1)
        data1 = capture_and_send(obj, args)
        if take_min:
            data0[0:22] = np.minimum(data0[0:22], data1[0:22])
            data0[22:28] = np.maximum(data0[22:28], data1[22:28])
        else:
            data0 = data1
    return data0


def check(
    peltier_on_value=DEFAULT_PELTIER_ON_VALUE,
    inverted_peltier_on_value=DEFAULT_INVERTED_PELTIER_ON_VALUE,
    heater_on_value=DEFAULT_HEATER_ON_VALUE,
    wait0=10,
    wait1=5,
    wait2=20,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )

    parser = common_parser()
    parser.add_argument(
        "--full",
        action="store_true",
        default="",
        help="testing of all 28 thermistors instead of 22 thermistors for the on-board components.",
    )
    parser.add_argument(
        "--type_b",
        action="store_true",
        default="",
        help="excluding two thermistors 24 and 25 for unimplemented RF milling cases for RI",
    )
    parser.add_argument(
        "--skipoff",
        action="store_true",
        help="skip confirming to turn off all the peltier devices",
    )
    parser.add_argument(
        "--th",
        type=str,
        default="",
        help="comma-separated list of thermistors to be checked",
    )
    args = parser.parse_args()

    try:
        con = parse_host_port(args)
    except Exception as e:  # noqa: E722
        logger.error(f"invalid telnet port: {e}")
        return 1

    logger.info("starting...")
    obj = QuelCmod(con[0], con[1])

    max_thermistor_idx = 28 if args.full else 22
    targets = [i for i in range(0, max_thermistor_idx)]
    if args.th != "":
        try:
            targets = [int(th) for th in args.th.split(",")]
        except ValueError:
            logger.error("invalid list of thermistors")
            return 1

    if args.type_b:
        targets = [i for i in targets if i not in (24, 25)]

    for t in targets:
        if not (0 <= t < max_thermistor_idx):
            logger.error(f"invalid index of thermistor {t}")
            return 1

    # Turning off all the peltiers and heaters
    if args.skipoff:
        logger.info("skipping to turn off all peltier devices")
    else:
        logger.info("turn off all peltier devices")
        for idx in range(28):
            if not obj.pl(idx, 0):
                logger.error("failed to set peltier power")
                return 1
            if capture_and_send(obj, args) is None:
                return 1

    # Capturing the temperature at equilibirium
    data0 = capture_and_send(obj, args)
    if data0 is None:
        return 1

    # Checking one by one
    error_flag = False
    wait_coeff = 1.0
    wait_coeff_recover = 1.0
    num_targets = len(targets)
    for i, idx in enumerate(targets):
        # Turn on!
        logger.info(f"checking thermistor {idx}")
        if (0 <= idx < 8) or (10 <= idx < 22):
            obj.pl(idx, peltier_on_value)
            wait_coeff = 1.5
            wait_coeff_recover = 2.0
        elif 8 <= idx < 10:
            obj.pl(idx, peltier_on_value)
            wait_coeff = 2.0
            wait_coeff_recover = 4.0
        elif 22 <= idx < 26:
            obj.pl(idx, inverted_peltier_on_value)
            wait_coeff = 3.0
            wait_coeff_recover = 6.0
        elif 26 <= idx < 28:
            obj.pl(idx, heater_on_value)
            wait_coeff = 6.0
            wait_coeff_recover = 3.0
        else:
            return 1

        data1 = waiting(obj, int(wait0 * wait_coeff), args)
        if data1 is None:
            logging.error(
                "quitting due to commnunication error! TURN OFF all peltier MANUALLY!! ***"
            )
            return 1

        # Turn off and waiting for the coduction from heat source to sensor
        obj.pl(idx, 0)
        data1 = waiting(obj, int(wait1 * wait_coeff_recover), args, True)
        if data1 is None:
            return 1
        diff = data1 - data0
        logger.info(f"dth[{idx:02d}] = {diff[idx]}")
        if idx < 22:
            maxdiff = np.argmin(diff)
        else:
            maxdiff = np.argmax(diff)
        if maxdiff != idx:
            logger.error(
                f"dth[{maxdiff:02d}] is {diff[maxdiff]} and smaller than the dth[{idx:02d}], something wrong!"
            )
            error_flag = True

        # Waiting until equilibirium is approached.
        if i < num_targets - 1:
            logger.info("recovering...")
            if waiting(obj, int(wait2 * wait_coeff_recover), args) is None:
                return 1

    # Display judgement
    if error_flag:
        print("*** FAILED ***")
        return 1
    else:
        print("*** PASSED ***")
        return 0
