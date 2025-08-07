import argparse
import dataclasses
import h5py
import numpy as np
import os
import sys
import textwrap
import warnings

from .. import _parser


@dataclasses.dataclass
class Stats:
    packets_seen: int
    packets_dropped: int
    last_seq: int


def main(args=None):
    stats = {}

    parser = argparse.ArgumentParser(
        prog="parser",
        description="Data acquisition / capture tool for rfmux",
        add_help=False,
    )

    parser.add_argument(
        "-i", "--interface", required=True, help="network interface to use (e.g. eth0)"
    )

    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=None,
        help="stop after seeing NUM_SAMPLES packets from any CRS module",
    )

    parser.add_argument(
        "-m",
        "--module",
        default=None,
        type=int,
        action="append",
        help="only collect data for the specified module",
    )

    parser.add_argument(
        "-c",
        "--channel",
        default=None,
        type=int,
        action="append",
        help="only collect data for the specified channel",
    )

    parser.add_argument(
        "-s",
        "--serial",
        default=None,
        type=int,
        action="append",
        help="only collect data for the specified CRS board",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite capture files, if they exist",
    )

    parser.add_argument(
        "-h", "--hdf5", default=None, type=str, help="store data to this HDF5 file"
    )

    parser.add_argument(
        "-t",
        "--text",
        default=False,
        action="store_true",
        help="dump streamed data to the screen (very verbose!)",
    )

    parser.add_argument(
        "--stats",
        default=False,
        action="store_true",
        help="Emit statistics when exiting",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Be chatty about dropped packets, etc.",
    )

    parser.add_argument(
        "--sniff-time",
        default=1,
        type=int,
        help="Collect packets for this many seconds before proceeding (necessary for HDF5 SWMR mode)",
    )

    parser.add_argument(
        "--chunk_size",
        default=128,
        type=int,
        help="Receive this many packets at once (using recvmmsg)",
    )

    parser.add_argument(
        "--reorder_buffer",
        default=256,
        type=int,
        help="Leave this many packets in a reorder queue to combat out-of-sequence UDP reception",
    )

    parser.add_argument("--help", action="help")

    args = parser.parse_args(args)
    p = _parser.Parser(args.interface)

    if args.hdf5:
        if os.path.exists(args.hdf5) and not args.force:
            sys.exit(f"hdf5 file {args.hdf5} exists, and --force was not present")

        h5 = h5py.File(args.hdf5, "w", libver="latest")

        # Create metadata - we need to do this before SWMR mode is turned on, sadly.
        # This constraint will be relaxed some time after HDF5 2.0.0.
        # (see https://zenodo.org/records/15722625)
        streamers = p.sniff(args.sniff_time)
        for serial, module in streamers:
            path = f"/raw/{serial:04d}/{module:1d}"
            dset = h5.require_dataset(
                path,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=_parser.ReadoutFrame,
                compression="lzf",
            )

        h5.swmr_mode = True

    channels = range(1024) if args.channel is None else np.array(args.channel) - 1

    try:
        while True:
            p.receive(args.chunk_size)
            for serial, module in p.get_queues():
                # Drain queues unconditionally (i.e. without filtering),
                # because otherwise they accumulate indefinitely in RAM
                if (xs := p.drain((serial, module)), args.reorder_buffer) is None:
                    continue

                #
                # Filter based on serial or module. TODO: upstream filtering
                # (eBPF, source-specific multicasting, ...)
                #

                if args.serial is not None and serial not in args.serial:
                    continue

                if args.module is not None and module not in args.module:
                    continue

                #
                # Collect statistics
                #

                s = stats.setdefault(
                    (serial, module),
                    Stats(
                        packets_seen=0,
                        packets_dropped=0,
                        last_seq=None,
                    ),
                )
                s.packets_seen += len(xs)

                # check if we dropped packets within this drain() call
                internal_drops = sum(np.diff(xs["seq"]) - 1)

                # check if we dropped packets since the last drain() call
                if (last_seq := s.last_seq) is not None:
                    boundary_drops = xs[0]["seq"] - last_seq - 1
                else:
                    boundary_drops = 0

                s.last_seq = xs[-1]["seq"]

                if internal_drops or boundary_drops:
                    s.packets_dropped += internal_drops + boundary_drops
                    if verbose:
                        print(
                            stderr,
                            f"Dropped {internal_drops + boundary_drops} packets! "
                            f"({internal_drops}, {boundary_drops})",
                        )

                #
                # Dump to text
                #

                if args.text:
                    for x in xs:
                        # Timestamp
                        ts = x["ts"]
                        print(
                            f"\tTimestamp(y={ts['y']} d={ts['d']:03} h:m:s={ts['h']:02}:{ts['m']:02}:{ts['s']:02} ss={ts['ss']}"
                        )

                        # Channel data
                        for n in channels:
                            sample = x["samples"][n]
                            print(
                                f"\tMC({module}.{n+1})"
                                f"\ti({np.real(sample):.6})"
                                f"\tq({np.imag(sample):.6})"
                                f"\tabs({np.abs(sample):.6})"
                            )

                #
                # Dump to HDF5
                #

                if args.hdf5:
                    path = f"/raw/{serial:04d}/{module:1d}"

                    # SWMR (single-writer multiple-reader) sounds nice, but we
                    # aren't allowed to alter metadata in this mode (yet).
                    if path not in h5:
                        warnings.warn(
                            f"Can't stream into nonexistent HDF5 path {path}! (SWMR limitation)"
                        )
                        continue

                    dset = h5[path]
                    old_len = dset.len()
                    dset.resize(old_len + len(xs), axis=0)
                    dset[old_len:] = np.array(xs)
                    dset.flush()

                # bail if num_samples provided and we've seen enough traffic
                if args.num_samples is not None and s.packets_seen >= args.num_samples:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if args.stats:
            for (serial, module), s in stats.items():
                frac_dropped = s.packets_dropped / (s.packets_seen + s.packets_dropped)
                print(
                    textwrap.dedent(
                        f"""
                    Serial {serial} module {module}:
                        packets seen: {s.packets_seen}
                        packets dropped: {s.packets_dropped} ({100*frac_dropped:.02%})
                """
                    ).strip()
                )


if __name__ == "__main__":
    main()
