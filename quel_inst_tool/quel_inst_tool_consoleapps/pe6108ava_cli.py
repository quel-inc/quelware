import argparse
import logging
import sys

from quel_inst_tool import Pe4104aj, Pe6108ava, PeSwitchState, Pexxxx

logger = logging.getLogger()


def common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--rbtype", choices=("PE6108AVA", "PE4104AJ"), help="rebooter type PE6108AVA or PE4104AJ")
    parser.add_argument("--ipaddr", type=str, required=True, help="IP address or hostname of the switch box")
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
    common_args(parser)
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
