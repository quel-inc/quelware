import argparse
import logging
import sys

from quel_inst_tool import Pe6108ava, PeSwitchState

logger = logging.getLogger()


def common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--ipaddr", type=str, required=True, help="IP address or hostname of PE6108AVA switch box")
    parser.add_argument("--idx", type=int, required=True, choices=(1, 2, 3, 4, 5, 6, 7, 8), help="index of switch")


def pe_switch_check():
    parser = argparse.ArgumentParser(description="show the status of the specified switch")
    common_args(parser)
    args = parser.parse_args()

    try:
        obj = Pe6108ava(hostname=args.ipaddr)
        print(obj.check_switch(args.idx).value)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def pe_switch_control():
    parser = argparse.ArgumentParser(description="show the status of the specified switch")
    common_args(parser)
    parser.add_argument("--state", choices=("on", "off"), required=True, help="'on' or 'off'")
    args = parser.parse_args()

    try:
        obj = Pe6108ava(hostname=args.ipaddr)
        obj.turn_switch(args.idx, PeSwitchState(args.state))
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def pe_switch_powercycle():
    parser = argparse.ArgumentParser(description="show the status of the specified switch")
    common_args(parser)
    args = parser.parse_args()

    try:
        obj = Pe6108ava(hostname=args.ipaddr)
        obj.powercycle_switch(args.idx)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
