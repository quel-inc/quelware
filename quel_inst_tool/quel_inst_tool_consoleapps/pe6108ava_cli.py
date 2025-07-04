import argparse
import logging
import sys

from quel_inst_tool import Pe4104aj, Pe6108ava, PeSwitchState, Pexxxx

logger = logging.getLogger()


def parse_indices(optstr: str) -> set[int]:
    ss = {int(s) for s in optstr.split(",")}
    for s in ss:
        if not (1 <= s <= 8):
            raise ValueError(f"invalid index: {s}")
    return ss


def common_args(parser: argparse.ArgumentParser, multiple_indices: bool = False):
    parser.add_argument("--rbtype", choices=("PE6108AVA", "PE4104AJ"), help="rebooter type PE6108AVA or PE4104AJ")
    parser.add_argument("--ipaddr", type=str, required=True, help="IP address or hostname of the switch box")
    if multiple_indices:
        parser.add_argument(
            "--idx", type=parse_indices, required=True, help="comma separated list of indices of switch"
        )
    else:
        parser.add_argument("--idx", type=int, required=True, choices=(1, 2, 3, 4, 5, 6, 7, 8), help="index of switch")


def pe_switch_check():
    parser = argparse.ArgumentParser(description="show the status of the specified switch")
    common_args(parser)
    args = parser.parse_args()

    try:
        if args.rbtype == "PE6108AVA":
            obj: Pexxxx = Pe6108ava(hostname=args.ipaddr)
        elif args.rbtype == "PE4104AJ":
            obj = Pe4104aj(hostname=args.ipaddr)
        else:
            raise AssertionError
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
        if args.rbtype == "PE6108AVA":
            obj: Pexxxx = Pe6108ava(hostname=args.ipaddr)
        elif args.rbtype == "PE4104AJ":
            obj = Pe4104aj(hostname=args.ipaddr)
        else:
            raise AssertionError
        obj.turn_switch(args.idx, PeSwitchState(args.state))
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def pe_switch_powercycle():
    parser = argparse.ArgumentParser(description="show the status of the specified switch")
    common_args(parser, multiple_indices=True)
    args = parser.parse_args()
    try:
        if args.rbtype == "PE6108AVA":
            obj: Pexxxx = Pe6108ava(hostname=args.ipaddr)
        elif args.rbtype == "PE4104AJ":
            obj = Pe4104aj(hostname=args.ipaddr)
        else:
            raise AssertionError
        obj.powercycle_switch(args.idx)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
