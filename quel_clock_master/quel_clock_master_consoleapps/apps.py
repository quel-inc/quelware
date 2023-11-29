import argparse
import logging
import sys

from quel_clock_master import QuBEMasterClient, SequencerClient

logger = logging.getLogger(__name__)


def init_parser_for_master(descr: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("ipaddr_master", type=str, help="IP address of the clock master")
    parser.add_argument("--master_port", type=int, default="16384")
    parser.add_argument("--master_reset_port", type=int, default="16385")
    return parser


def init_parser_for_seqr(descr: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descr)  # no arguments about master is required.
    parser.add_argument("ipaddr_targets", type=str, nargs="+", help="IP addresses of the target boxes")
    parser.add_argument("--seqr_port", type=int, default=SequencerClient.DEFAULT_SEQR_PORT)
    parser.add_argument("--synch_port", type=int, default=SequencerClient.DEFAULT_SYNCH_PORT)
    return parser


def reset_master_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_master("resetting the master node")
    args = parser.parse_args()
    proxy = QuBEMasterClient(args.ipaddr_master, args.master_port, args.master_reset_port)

    retcode = proxy.reset()
    if retcode:
        logger.info("reset successfully")
        sys.exit(0)
    else:
        logger.error("failure in reset")
        sys.exit(1)


def reset_target_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_seqr("resetting the FPGA of the given client nodes")
    args = parser.parse_args()

    flag = True
    for ipaddr_target in args.ipaddr_targets:
        q = SequencerClient(ipaddr_target, args.seqr_port, args.synch_port)
        retcode = q.kick_softreset()
        if not retcode:
            flag = False

    sys.exit(0 if flag else 1)


def read_master_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_master("retrieving the clock counter of the given master node")
    args = parser.parse_args()
    proxy = QuBEMasterClient(args.ipaddr_master, args.master_port, args.master_reset_port)

    retcode, clock = proxy.read_clock(0)
    if retcode:
        logger.info(f"{args.ipaddr_master}: {clock:d}")
        sys.exit(0)
    else:
        logger.error("failure in reading the clock")
        sys.exit(1)


def clear_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_master(
        "zero'ing the clock counter of the master node specified by the first argument, "
        "and then clearing the client nodes if provided."
    )
    parser.add_argument("ipaddr_targets", type=str, nargs="*", help="IP addresses of target boxes to kick")
    args = parser.parse_args()
    proxy = QuBEMasterClient(args.ipaddr_master, args.master_port, args.master_reset_port)

    retcode = proxy.clear_clock(0)
    if retcode:
        logger.info("cleared successfully")
    else:
        logger.error("failure in cleaning")
        sys.exit(1)

    if len(args.ipaddr_targets) > 0:
        retcode = proxy.kick_clock_synch(args.ipaddr_targets)
        if retcode:
            logger.info("kicked successfully")
        else:
            logger.error("failure in kicking the targets")
            sys.exit(1)
    else:
        logger.info("no kick is conducted because no targets are given")


def kick_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_master(
        "triggering the synchronization protocol between the master node given as the first argument "
        "and the client nodes given as the remaining arguments"
    )
    parser.add_argument("ipaddr_targets", type=str, nargs="+", help="IP addresses of target boxes to kick")
    args = parser.parse_args()
    proxy = QuBEMasterClient(args.ipaddr_master, args.master_port, args.master_reset_port)

    retcode = proxy.kick_clock_synch(args.ipaddr_targets)
    if retcode:
        logger.info("kicked successfully")
    else:
        logger.error("failure in kicking the targets")
        sys.exit(1)


def read_target_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_seqr("retrieving the clock counter of the given client nodes")
    args = parser.parse_args()

    flag = True
    for ipaddr_target in args.ipaddr_targets:
        q = SequencerClient(ipaddr_target, args.seqr_port, args.synch_port)
        retcode, clock, last_sysref = q.read_clock()
        if retcode:
            if last_sysref > 0:
                logger.info(f"{ipaddr_target}: {clock:d} {last_sysref:d}")
            else:
                logger.info(f"{ipaddr_target}: {clock:d}")
        else:
            logger.info(f"{ipaddr_target}: not found")
            flag = False

    sys.exit(0 if flag else 1)


def read_main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = init_parser_for_master(
        "retrieving the clock counters of the master node given as the first arguments "
        "and the client nodes given as the remaining arguments."
    )
    parser.add_argument("ipaddr_targets", type=str, nargs="*", help="IP addresses of the target boxes")
    parser.add_argument("--seqr_port", type=int, default=SequencerClient.DEFAULT_SEQR_PORT)
    parser.add_argument("--synch_port", type=int, default=SequencerClient.DEFAULT_SYNCH_PORT)
    args = parser.parse_args()
    proxy = QuBEMasterClient(args.ipaddr_master, args.master_port, args.master_reset_port)

    flag = True
    retcode, clock = proxy.read_clock(0)
    if retcode:
        logger.info(f"{args.ipaddr_master}: {clock:d}")
    else:
        logger.info(f"{args.ipaddr_master}: not found")
        flag = False

    for ipaddr_target in args.ipaddr_targets:
        q = SequencerClient(ipaddr_target, args.seqr_port, args.synch_port)
        retcode, clock, last_sysref = q.read_clock()
        if retcode:
            if last_sysref > 0:
                logger.info(f"{ipaddr_target}: {clock:d} {last_sysref:d}")
            else:
                logger.info(f"{ipaddr_target}: {clock:d}")
        else:
            logger.info(f"{ipaddr_target}: not found")
            flag = False

    sys.exit(0 if flag else 1)
