import argparse
import logging
import pathlib
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from quel_inst_tool import E440xb, E440xbReadableParams, E440xbWritableParams, E4405b, E4407b, InstDevManager

logger = logging.getLogger("main")

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description="A command line interface for E440xB",
)

parser.add_argument(
    "-t", "--spatype", choices=("E4405B", "E4407B"), help="type of spectrum analyzer to use, either of E4405B or E4407B"
)
parser.add_argument("--reset", action="store_true", help="reset E440xB at initialization")
parser.add_argument("-r", "--resolution", type=float, help="resolution bandwidth")
parser.add_argument("-a", "--average", type=int, help="average count")
parser.add_argument("-p", "--points", type=int, help="number of sweep points")
parser.add_argument("--peak", type=float, help="minimum power of peaks")
parser.add_argument("-c", "--freq_center", type=float, help="center frequency in Hz")
parser.add_argument("-s", "--freq_span", type=float, help="frequency span in Hz")
parser.add_argument("-d", "--delay", default=0.0, type=float, help="wait before capturing trace in seconds")
parser.add_argument("-o", "--outfile", type=pathlib.Path, help="a path to save an image of the captured graph")


def main():
    wprms: E440xbWritableParams = E440xbWritableParams()
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:4}] {name}: {message}", style="{")

    args = parser.parse_args()
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
    im.scan()
    if args.spatype is not None:
        if args.spatype == "E4405B":
            e440xb: E440xb = E4405b(im.lookup(prod_id="E4405B"))
        elif args.spatype == "E4407B":
            e440xb = E4407b(im.lookup(prod_id="E4407B"))
        else:
            raise AssertionError
    else:
        raise ValueError("Specify the name of SPA (E4405B or E4407B) using --spatype option")

    if args.reset:
        e440xb.reset()

    if args.freq_center is not None:
        wprms.freq_center = args.freq_center
    if args.freq_span is not None:
        wprms.freq_span = args.freq_span
    if args.points is not None:
        wprms.sweep_points = args.points
    if args.average is not None:
        if args.average <= 0:
            e440xb.average_clear()
            wprms.average_enable = False
        else:
            e440xb.average_clear()
            wprms = args.average
            wprms.average_enable = True
    if args.resolution is not None:
        wprms.resolution_bandwidth = args.resolution
    if args.delay > 0:
        logger.info(f"wainting for {args.delay:f} seconds...")
        time.sleep(args.delay)

    wprms.update_device_parameter(e440xb)
    time.sleep(1)  # TODO: shorten it!
    rprms: E440xbReadableParams = e440xb.check_prms(wprms)

    logger.info(f"average_enable, average_count = {rprms.average_enable}, {rprms.average_count}")
    if args.peak:
        fd0, p0 = e440xb.trace_and_peak_get(minimum_power=args.peak)
        logger.info(p0)
    else:
        fd0 = e440xb.trace_get()

    if args.outfile is None:
        mpl.use("Gtk3Agg")
        logger.info("push Q to update.")
        logger.info("push Ctrl+C to quit.")
        while True:
            plt.cla()
            plt.plot(fd0[:, 0], fd0[:, 1])
            if args.delay:
                plt.pause(args.delay)
            else:
                plt.show()

            e440xb._cache_flush()  # TODO: remove it after test

            if args.peak:
                fd0, p0 = e440xb.trace_and_peak_get(minimum_power=args.peak)
                logger.info(p0)
            else:
                fd0 = e440xb.trace_get()
    else:
        ext = args.outfile.suffix
        if ext == ".png":
            plt.plot(fd0[:, 0], fd0[:, 1])
            plt.savefig(args.outfile)
        elif ext == ".npz":
            np.savez(args.outfile, hz=fd0[:, 0], db=fd0[:, 1])
        elif ext == ".csv":
            np.savetxt(args.outfile, fd0, fmt="%g", delimiter=",")
        else:
            logger.error(f"unsupported extention of outpuf file '{ext}', not saved")


if __name__ == "__main__":
    main()
