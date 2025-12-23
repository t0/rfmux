#!/usr/bin/env python3
"""
CRS Packet Parser

Receives and processes packets from CRS boards, with options to:
- Dump packets to text (for debugging)
- Write to GetData dirfiles (for analysis)
- Filter by serial, module, and channel
- Collect drop statistics
"""

import argparse
import signal
import socket
import sys
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional

import psutil
import pygetdata as gd
import numpy as np

from rfmux.streamer import (
    ReadoutPacketReceiver,
    PFBPacketReceiver,
    get_multicast_socket,
    STREAMER_PORT,
    PFB_STREAMER_PORT,
    PFBPACKET_NSAMP_MAX,
)


def resolve_interface(interface_name: str) -> str:
    """
    Resolve network interface name to IPv4 address.

    Args:
        interface_name: Interface name (e.g., 'eth0', 'en0', 'Ethernet')

    Returns:
        IPv4 address as string

    Raises:
        ValueError: If interface not found or has no IPv4 address
    """
    addrs = psutil.net_if_addrs()
    if interface_name not in addrs:
        available = ", ".join(sorted(addrs.keys()))
        raise ValueError(
            f"Interface '{interface_name}' not found. Available interfaces: {available}"
        )

    # Find IPv4 address
    for addr in addrs[interface_name]:
        if addr.family == socket.AF_INET:
            return addr.address

    raise ValueError(f"No IPv4 address found for interface '{interface_name}'")


def parse_ranges(spec: str, min_val: int, max_val: int, name: str) -> list[range]:
    """
    Parse comma-separated ranges like "1,5-10,20-30" into list of range objects.

    Args:
        spec: Comma-separated range specification (1-indexed)
        min_val: Minimum allowed value (1-indexed)
        max_val: Maximum allowed value (1-indexed)
        name: Name for error messages (e.g., "channel", "module")

    Returns:
        List of range objects (0-indexed, exclusive end as per Python convention)

    Examples:
        >>> parse_ranges("1,5-10", 1, 100, "channel")
        [range(0, 1), range(4, 10)]
    """
    ranges = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(token)

        if not (min_val <= start <= end <= max_val):
            raise ValueError(
                f"{name.capitalize()} range {start}-{end} out of bounds "
                f"[{min_val}, {max_val}]"
            )

        # Convert to 0-indexed range (exclusive end)
        ranges.append(range(start - 1, end))

    return ranges


@dataclass
class ModuleStats:
    """Statistics for a single readout module"""

    packets_seen: int = 0
    last_seq: Optional[int] = None
    packets_dropped: int = 0
    dirfile_fields: Optional[dict[str, str]] = None
    dirfile_frame: int = 0  # Per-module frame position for writing


@dataclass
class BoardStats:
    """Statistics and state for a CRS board"""

    dirfile: Optional[gd.dirfile] = None
    module_stats: dict[int, ModuleStats] = field(
        default_factory=lambda: defaultdict(ModuleStats)
    )
    packets_indexed: int = 0


def setup_dirfile_for_board(
    board_stats: BoardStats,
    serial: int,
    main_dirfile: gd.dirfile,
    channels: list[range],
):
    """Create a subdirfile for a specific CRS board"""
    board_ns = f"serial_{serial:04d}"
    board_path = f"{main_dirfile.name}/{board_ns}"

    board_stats.dirfile = gd.dirfile(
        board_path, gd.CREAT | gd.RDWR | gd.EXCL | gd.PRETTY_PRINT
    )

    num_channels = sum(len(r) for r in channels)

    # Add index field for multiplexing
    e = gd.entry(gd.RAW_ENTRY, "mplex_idx", 0, (gd.UINT16, 2 * num_channels))
    board_stats.dirfile.add(e)
    board_stats.dirfile.hide("mplex_idx")
    board_stats.dirfile.metaflush()

    # Link from main dirfile
    main_dirfile.include(f"{board_ns}/format", 0, 0, board_ns)
    main_dirfile.metaflush()


def setup_dirfile_for_module(
    board_stats: BoardStats,
    module_stats: ModuleStats,
    module: int,
    channels: list[range],
):
    """Create dirfile fields for a specific module"""
    df = board_stats.dirfile
    num_channels = sum(len(r) for r in channels)

    # Field names
    raw_field = f"m{module+1:02d}_raw32"
    ts_sbs_field = f"m{module+1:02d}_ts_sbs"
    ts_ss_field = f"m{module+1:02d}_ts_ss"

    module_stats.dirfile_fields = {
        "raw": raw_field,
        "ts_sbs": ts_sbs_field,
        "ts_ss": ts_ss_field,
    }

    # Add timestamp fields
    df.add(gd.entry(gd.RAW_ENTRY, ts_sbs_field, 0, (gd.INT32, 1)))
    df.add(gd.entry(gd.RAW_ENTRY, ts_ss_field, 0, (gd.INT32, 1)))

    # Add raw multiplexed field
    df.add(gd.entry(gd.RAW_ENTRY, raw_field, 0, (gd.INT32, 2 * num_channels)))
    df.hide(raw_field)

    # Create demultiplexed I/Q fields for each channel
    # Build a flat list of channels with their offsets
    ch_offset = 0
    for r in channels:
        for ch in r:
            i_field = f"m{module+1:02d}_c{ch+1:04d}_i"
            q_field = f"m{module+1:02d}_c{ch+1:04d}_q"

            # MPLEX entry: (in_field, count_field, value, max)
            df.add(
                gd.entry(
                    gd.MPLEX_ENTRY,
                    i_field,
                    0,
                    (raw_field, "mplex_idx", 2 * ch_offset, 2 * num_channels),
                )
            )

            df.add(
                gd.entry(
                    gd.MPLEX_ENTRY,
                    q_field,
                    0,
                    (raw_field, "mplex_idx", 2 * ch_offset + 1, 2 * num_channels),
                )
            )

            # Add combined IQ field (scaled)
            iq_field = f"m{module+1:02d}_c{ch+1:04d}"
            # LINCOM entry: (in_fields, m, b)
            df.add(
                gd.entry(
                    gd.LINCOM_ENTRY,
                    iq_field,
                    0,
                    ((i_field, q_field), (1 / 256.0 + 0j, 1j / 256.0), (0, 0)),
                )
            )

            ch_offset += 1

    # Add timebase field
    timebase_field = f"m{module+1:02d}_timebase"
    df.add(
        gd.entry(
            gd.LINCOM_ENTRY,
            timebase_field,
            0,
            ((ts_sbs_field, ts_ss_field), (1.0, 1 / 125e6), (0, 0)),
        )
    )

    # Flush metadata so dirfile can be read immediately
    df.metaflush()


def main(*args):
    parser = argparse.ArgumentParser(
        description="CRS packet parser - receives and processes CRS readout packets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump readout packets to text (by hostname)
  %(prog)s -H rfmux0033.local -t

  # Write readout packets to dirfile, filter specific channels (by interface name)
  %(prog)s -i eth0 -d ~/data/test.dirfile -c 1-10,50-100

  # Parse PFB packets instead of readout packets
  %(prog)s -H rfmux0033.local --pfb -d ~/data/pfb.dirfile

  # Write to dirfile using interface IP directly
  %(prog)s --interface-ip 192.168.1.100 -d ~/data/test.dirfile

  # Collect drop statistics for specific serial
  %(prog)s -H rfmux0033.local --drop-stats -s 33

  # Capture N frames then exit
  %(prog)s -i eth0 -n 1000 -d ~/data/capture.dirfile
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-H",
        "--hostname",
        metavar="HOST",
        help="CRS board hostname or IP (auto-discovers interface)",
    )
    group.add_argument(
        "-i",
        "--interface",
        metavar="IFACE",
        help="Local interface name for multicast (e.g., 'eth0', 'en0', 'Ethernet')",
    )
    group.add_argument(
        "--interface-ip",
        metavar="IP",
        help="Local interface IP address for multicast (receives from all sources)",
    )

    parser.add_argument(
        "--pfb",
        action="store_true",
        help="Parse PFB packets instead of readout packets (port 9877)",
    )

    parser.add_argument(
        "-d",
        "--dirfile",
        metavar="PATH",
        help="Write packets to GetData dirfile at PATH",
    )

    parser.add_argument(
        "-t", "--text", action="store_true", help="Dump packets to text (for debugging)"
    )

    parser.add_argument(
        "--drop-stats",
        action="store_true",
        help="Collect and report dropped packet statistics",
    )

    parser.add_argument(
        "-c",
        "--channel",
        metavar="RANGE",
        help='Filter by channel ranges (1-indexed, e.g., "1,5-10,100-200")',
    )

    parser.add_argument(
        "-m",
        "--module",
        metavar="RANGE",
        help='Filter by module ranges (1-indexed, e.g., "1-4")',
    )

    parser.add_argument(
        "-s", "--serial", type=int, metavar="N", help="Filter by board serial number"
    )

    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        metavar="N",
        help="Number of frames to capture (default: unlimited)",
    )

    parser.add_argument(
        "--reorder-window",
        type=int,
        default=256,
        metavar="N",
        help="Packet reorder window size (default: 256)",
    )

    args = parser.parse_args(args or None)

    # Validate arguments
    if not (args.text or args.dirfile or args.drop_stats):
        parser.error(
            "Must specify at least one of --text, --dirfile, or --drop-stats is necessary"
        )

    if args.num_frames is not None and args.num_frames < 0:
        parser.error("--num-frames must be non-negative")

    # Parse channel and module ranges
    TOTAL_CHANNELS = 1024
    TOTAL_MODULES = 4

    channels = (
        parse_ranges(args.channel, 1, TOTAL_CHANNELS, "channel")
        if args.channel
        else [range(0, TOTAL_CHANNELS)]
    )

    modules = (
        parse_ranges(args.module, 1, TOTAL_MODULES, "module")
        if args.module
        else [range(0, TOTAL_MODULES)]
    )

    serials = {args.serial} if args.serial else set()

    # Resolve interface name to IP if needed
    interface_ip = None
    if args.interface:
        interface_ip = resolve_interface(args.interface)
    elif args.interface_ip:
        interface_ip = args.interface_ip

    # Statistics tracking
    board_stats: dict[int, BoardStats] = defaultdict(BoardStats)

    # Dispatch to appropriate mode
    if args.pfb:
        return main_pfb(args, serials, modules, channels, interface_ip, board_stats)
    else:
        return main_readout(args, serials, modules, channels, interface_ip, board_stats)


def main_readout(args, serials, modules, channels, interface_ip, board_stats):
    """Main loop for readout packet parsing"""
    # Signal handler for clean exit
    def signal_handler(signum, frame):
        if args.drop_stats:
            print("\n=== Drop Statistics ===", file=sys.stderr)
            for serial, bstats in board_stats.items():
                for module, mstats in bstats.module_stats.items():
                    print(
                        f"Serial {serial} module {module}: "
                        f"{mstats.packets_seen} packets seen "
                        f"({mstats.packets_dropped} dropped)",
                        file=sys.stderr,
                    )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Main receive loop
    chunk_size = 256
    num_channels = sum(len(r) for r in channels)
    index = np.arange(2 * num_channels, dtype=np.uint16)

    # Setup packet receiver with large buffer to reduce drops
    with (
        closing(
            get_multicast_socket(
                crs_hostname=args.hostname,
                port=STREAMER_PORT,
                interface=interface_ip,
                buffer_size=67108864,
            )
        ) as sock,
        closing(
            gd.dirfile(args.dirfile, gd.CREAT | gd.RDWR | gd.EXCL | gd.PRETTY_PRINT)
            if args.dirfile
            else tempfile.TemporaryFile()
        ) as main_dirfile,
    ):
        receiver = ReadoutPacketReceiver(sock, reorder_window=args.reorder_window)

        while True:
            # Receive batch of packets
            receiver.receive_batch(chunk_size, timeout_ms=1000)

            # Process all queues
            for serial, module, queue in receiver.get_all_queues():
                # Filter by serial
                if serials and serial not in serials:
                    continue

                # Filter by module
                if not any(module in r for r in modules):
                    continue

                # Get/create statistics
                bstats = board_stats[serial]
                mstats = bstats.module_stats[module]

                # Process packets from this queue
                while packet := queue.try_pop():
                    pkt = packet.to_python()

                    # Track statistics
                    mstats.packets_seen += 1

                    if args.num_frames and mstats.packets_seen > args.num_frames:
                        signal_handler(signal.SIGTERM, None)

                    # Drop detection
                    if args.drop_stats and mstats.last_seq is not None:
                        expected_seq = (mstats.last_seq + 1) & 0xFFFFFFFF
                        if pkt.seq != expected_seq:
                            gap = (pkt.seq - expected_seq) & 0xFFFFFFFF
                            print(
                                f"Dropped packet: module {module}, "
                                f"seq {mstats.last_seq:08x} -> {pkt.seq:08x} "
                                f"({gap} lost)"
                            )
                            mstats.packets_dropped += gap
                    mstats.last_seq = pkt.seq

                    # Text dump
                    if args.text:
                        flags = []
                        if pkt.flags & 0x1:
                            flags.append("OR")
                        if pkt.flags & 0x2:
                            flags.append("OV")
                        flags_str = " ".join(flags)

                        print(
                            f"Packet(magic=0x{pkt.magic:08x} version={pkt.version} "
                            f"serial={pkt.serial} num_modules={pkt.num_modules} "
                            f"flags='{flags_str}' fir_stage={pkt.fir_stage} "
                            f"module={pkt.module} seq={pkt.seq})"
                        )

                        ts = pkt.ts
                        print(
                            f"  Timestamp(source={ts.get_source()} y={ts.y} d={ts.d} "
                            f"h:m:s={ts.h}:{ts.m}:{ts.s} ss={ts.ss} "
                            f"sbs={ts.sbs} c={ts.c & 0x1fffffff} "
                            f"recent={'Y' if ts.is_recent() else 'N'})"
                        )

                        # Sample channels
                        num_pkt_channels = pkt.get_num_channels()
                        for r in channels:
                            for ch in r:
                                if ch >= num_pkt_channels:
                                    break
                                sample = pkt.get_channel(ch)
                                print(
                                    f"  CH({module}.{ch+1}) "
                                    f"i={sample.real:.3f} q={sample.imag:.3f} abs={abs(sample):.3f}"
                                )

                    # Dirfile output
                    if args.dirfile:
                        # Create board dirfile if needed
                        if bstats.dirfile is None:
                            setup_dirfile_for_board(
                                bstats, serial, main_dirfile, channels
                            )

                        # Create module fields if needed
                        if mstats.dirfile_fields is None:
                            setup_dirfile_for_module(bstats, mstats, module, channels)

                        df = bstats.dirfile
                        fields = mstats.dirfile_fields

                        # Use per-module frame counter
                        frame = mstats.dirfile_frame

                        # Write timestamp
                        ts = pkt.ts
                        df.putdata(
                            fields["ts_sbs"],
                            np.array([ts.sbs], dtype=np.int32),
                            first_frame=frame,
                        )
                        df.putdata(
                            fields["ts_ss"],
                            np.array([ts.ss], dtype=np.int32),
                            first_frame=frame,
                        )

                        # Write channel data - extract selected channels and interleave I/Q
                        samples = pkt.samples
                        num_pkt_channels = pkt.get_num_channels()

                        # Gather selected channel data
                        selected = []
                        for r in channels:
                            for ch in r:
                                if ch < num_pkt_channels:
                                    selected.append(samples[ch])

                        # Convert to interleaved I/Q format (scaled by 256)
                        if selected:
                            complex_array = np.array(selected, dtype=np.complex64)
                            raw_data = np.empty(2 * len(selected), dtype=np.int32)
                            raw_data[0::2] = (complex_array.real * 256).astype(np.int32)
                            raw_data[1::2] = (complex_array.imag * 256).astype(np.int32)

                            df.putdata(fields["raw"], raw_data, first_frame=frame)

                            # Extend mplex_idx to match the longest module
                            # Write index if this module is ahead of the shared index
                            if frame >= bstats.packets_indexed:
                                df.putdata(
                                    "mplex_idx",
                                    index,
                                    first_frame=bstats.packets_indexed,
                                )
                                bstats.packets_indexed = frame + 1

                            # Increment per-module frame counter
                            mstats.dirfile_frame += 1


def main_pfb(args, serials, modules, channels, interface_ip, board_stats):
    """Main loop for PFB packet parsing"""
    # Signal handler for clean exit
    def signal_handler(signum, frame):
        print("\n=== Receiver Statistics ===", file=sys.stderr)
        if args.drop_stats:
            for serial, bstats in board_stats.items():
                for module, mstats in bstats.module_stats.items():
                    print(
                        f"Serial {serial} module {module}: "
                        f"{mstats.packets_seen} packets seen "
                        f"({mstats.packets_dropped} dropped)",
                        file=sys.stderr,
                    )

        # Print receiver stats on exit
        stats = receiver.get_stats()
        print(
            f"Total packets received: {stats.total_packets_received}\n"
            f"Total bytes received: {stats.total_bytes_received}\n"
            f"Invalid packets: {stats.invalid_packets}\n"
            f"Wrong magic: {stats.wrong_magic}",
            file=sys.stderr,
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Main receive loop
    chunk_size = 256
    receiver = None

    # Setup packet receiver with large buffer to reduce drops
    with (
        closing(
            get_multicast_socket(
                crs_hostname=args.hostname,
                port=PFB_STREAMER_PORT,
                interface=interface_ip,
                buffer_size=67108864,
            )
        ) as sock,
        closing(
            gd.dirfile(args.dirfile, gd.CREAT | gd.RDWR | gd.EXCL | gd.PRETTY_PRINT)
            if args.dirfile
            else tempfile.TemporaryFile()
        ) as main_dirfile,
    ):
        receiver = PFBPacketReceiver(sock, reorder_window=args.reorder_window)

        while True:
            # Receive batch of packets
            n = receiver.receive_batch(chunk_size, timeout_ms=1000)

            # Process all queues
            for serial, module, queue in receiver.get_all_queues():
                # Filter by serial
                if serials and serial not in serials:
                    continue

                # Filter by module
                if not any(module in r for r in modules):
                    continue

                # Get/create statistics
                bstats = board_stats[serial]
                mstats = bstats.module_stats[module]

                # Process packets from this queue
                while packet := queue.try_pop():
                    pkt = packet.to_python()

                    # Track statistics
                    mstats.packets_seen += 1

                    if args.num_frames and mstats.packets_seen > args.num_frames:
                        signal_handler(signal.SIGTERM, None)

                    # Drop detection
                    if args.drop_stats and mstats.last_seq is not None:
                        expected_seq = (mstats.last_seq + 1) & 0xFFFFFFFF
                        if pkt.seq != expected_seq:
                            gap = (pkt.seq - expected_seq) & 0xFFFFFFFF
                            print(
                                f"Dropped packet: module {module}, "
                                f"seq {mstats.last_seq:08x} -> {pkt.seq:08x} "
                                f"({gap} lost)"
                            )
                            mstats.packets_dropped += gap
                    mstats.last_seq = pkt.seq

                    # Text dump
                    if args.text:
                        mode_names = {0: "PFB1", 1: "PFB2", 2: "PFB4"}
                        mode_str = mode_names.get(pkt.mode, f"UNKNOWN({pkt.mode})")

                        print(
                            f"PFBPacket(magic=0x{pkt.magic:08x} version={pkt.version} "
                            f"mode={mode_str} serial={pkt.serial} module={pkt.module} "
                            f"seq={pkt.seq} num_samples={pkt.num_samples})"
                        )
                        print(
                            f"  Slots: slot1={pkt.slot1} slot2={pkt.slot2} "
                            f"slot3={pkt.slot3} slot4={pkt.slot4}"
                        )

                        ts = pkt.ts
                        print(
                            f"  Timestamp(source={ts.get_source()} y={ts.y} d={ts.d} "
                            f"h:m:s={ts.h}:{ts.m}:{ts.s} ss={ts.ss} "
                            f"sbs={ts.sbs} c={ts.c & 0x1fffffff} "
                            f"recent={'Y' if ts.is_recent() else 'N'})"
                        )

                        # Determine number of active slots based on mode
                        num_slots = {0: 1, 1: 2, 2: 4}.get(pkt.mode, 4)
                        slot_channels = [pkt.slot1, pkt.slot2, pkt.slot3, pkt.slot4][:num_slots]

                        # Sample a few interleaved values
                        samples = pkt.samples
                        max_display = min(8, pkt.num_samples)
                        for i in range(max_display):
                            # Samples are interleaved across slots
                            slot_idx = i % num_slots
                            ch = slot_channels[slot_idx]
                            sample = samples[i]
                            print(
                                f"  Sample[{i}] slot{slot_idx+1}->ch{ch+1}: "
                                f"i={sample.real:.3f} q={sample.imag:.3f} abs={abs(sample):.3f}"
                            )

                        if pkt.num_samples > max_display:
                            print(f"  ... ({pkt.num_samples - max_display} more samples)")

                    # Dirfile output
                    if args.dirfile:
                        # Create board dirfile if needed
                        if bstats.dirfile is None:
                            setup_pfb_dirfile_for_board(
                                bstats, serial, main_dirfile, channels
                            )

                        # Create module fields if needed
                        if mstats.dirfile_fields is None:
                            setup_pfb_dirfile_for_module(bstats, mstats, module, channels)

                        df = bstats.dirfile
                        fields = mstats.dirfile_fields

                        # Use per-module frame counter
                        frame = mstats.dirfile_frame

                        # Write timestamp
                        ts = pkt.ts
                        df.putdata(
                            fields["ts_sbs"],
                            np.array([ts.sbs], dtype=np.int32),
                            first_frame=frame,
                        )
                        df.putdata(
                            fields["ts_ss"],
                            np.array([ts.ss], dtype=np.int32),
                            first_frame=frame,
                        )

                        # Write slot headers (which channels are in each slot)
                        slots = np.array([pkt.slot1, pkt.slot2, pkt.slot3, pkt.slot4], dtype=np.uint16)
                        df.putdata(fields["slots"], slots, first_frame=frame)

                        # Write mode
                        df.putdata(
                            fields["mode"],
                            np.array([pkt.mode], dtype=np.uint8),
                            first_frame=frame,
                        )

                        # Determine number of active slots based on mode
                        num_slots = {0: 1, 1: 2, 2: 4}.get(pkt.mode, 4)

                        # Create slot_idx for MPLEX indexing (similar to readout mplex_idx)
                        # For PFB4: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, ...]
                        # Where: 0=slot1_I, 1=slot1_Q, 2=slot2_I, 3=slot2_Q, ...
                        # This repeats with period = 2 * num_slots
                        num_samples = len(pkt.samples)
                        period = 2 * num_slots
                        slot_idx = np.tile(np.arange(period, dtype=np.uint16), (num_samples * 2 + period - 1) // period)[:2 * num_samples]

                        df.putdata(fields["slot_idx"], slot_idx, first_frame=frame)

                        # Write samples (interleaved I/Q format, scaled by 256)
                        samples = pkt.samples
                        raw_data = np.empty(2 * len(samples), dtype=np.int32)
                        raw_data[0::2] = (np.real(samples) * 256).astype(np.int32)
                        raw_data[1::2] = (np.imag(samples) * 256).astype(np.int32)

                        df.putdata(fields["raw"], raw_data, first_frame=frame)

                        # Increment per-module frame counter
                        mstats.dirfile_frame += 1


def setup_pfb_dirfile_for_board(
    board_stats: BoardStats,
    serial: int,
    main_dirfile: gd.dirfile,
    channels: list[range],
):
    """Create a subdirfile for a specific CRS board (PFB mode)"""
    board_ns = f"serial_{serial:04d}"
    board_path = f"{main_dirfile.name}/{board_ns}"

    board_stats.dirfile = gd.dirfile(
        board_path, gd.CREAT | gd.RDWR | gd.EXCL | gd.PRETTY_PRINT
    )

    # Link from main dirfile
    main_dirfile.include(f"{board_ns}/format", 0, 0, board_ns)
    main_dirfile.metaflush()


def setup_pfb_dirfile_for_module(
    board_stats: BoardStats,
    module_stats: ModuleStats,
    module: int,
    channels: list[range],
):
    """Create dirfile fields for a specific module (PFB mode)"""
    df = board_stats.dirfile

    # Field names
    raw_field = f"m{module+1:02d}_pfb_raw32"
    slots_field = f"m{module+1:02d}_pfb_slots"
    slot_idx_field = f"m{module+1:02d}_pfb_slot_idx"
    mode_field = f"m{module+1:02d}_pfb_mode"
    ts_sbs_field = f"m{module+1:02d}_ts_sbs"
    ts_ss_field = f"m{module+1:02d}_ts_ss"

    module_stats.dirfile_fields = {
        "raw": raw_field,
        "slots": slots_field,
        "slot_idx": slot_idx_field,
        "mode": mode_field,
        "ts_sbs": ts_sbs_field,
        "ts_ss": ts_ss_field,
    }

    # Add timestamp fields
    df.add(gd.entry(gd.RAW_ENTRY, ts_sbs_field, 0, (gd.INT32, 1)))
    df.add(gd.entry(gd.RAW_ENTRY, ts_ss_field, 0, (gd.INT32, 1)))

    # Add mode field
    df.add(gd.entry(gd.RAW_ENTRY, mode_field, 0, (gd.UINT8, 1)))

    # Add slot header field (4 uint16 values: slot1, slot2, slot3, slot4)
    # This tells which channel each slot corresponds to
    df.add(gd.entry(gd.RAW_ENTRY, slots_field, 0, (gd.UINT16, 4)))
    df.hide(slots_field)

    # Add slot index field (repeating pattern [0,1,2,3,0,1,2,3,...] for MPLEX)
    # Maximum PFB packet has PFBPACKET_NSAMP_MAX samples
    df.add(gd.entry(gd.RAW_ENTRY, slot_idx_field, 0, (gd.UINT16, PFBPACKET_NSAMP_MAX)))
    df.hide(slot_idx_field)

    # Add raw multiplexed field (variable length, but we'll make it large enough)
    # Each sample is 2 int32 values (I and Q)
    df.add(gd.entry(gd.RAW_ENTRY, raw_field, 0, (gd.INT32, 2 * PFBPACKET_NSAMP_MAX)))
    df.hide(raw_field)

    # Create demultiplexed fields for each slot
    # PFB packets interleave samples across active slots (1, 2, or 4)
    # We create MPLEX fields to extract each slot's samples from the raw field
    # The slot_idx field repeats [0,1,2,3,...] and we MPLEX on that
    # The slot header fields (slot1-slot4) tell which channel each slot corresponds to
    NUM_SLOTS=4

    for slot_idx in range(NUM_SLOTS):
        slot_num = slot_idx + 1

        # Create I and Q fields for this slot using MPLEX
        # slot_idx cycles through [0, 1, 2, ..., 2*num_slots-1]
        # For PFB4: 0=slot1_I, 1=slot1_Q, 2=slot2_I, 3=slot2_Q, ...
        # We extract where slot_idx == 2*slot_idx for I, 2*slot_idx+1 for Q

        i_field_raw = f"m{module+1:02d}_slot{slot_num}_i_raw"
        q_field_raw = f"m{module+1:02d}_slot{slot_num}_q_raw"

        df.add(
            gd.entry(
                gd.MPLEX_ENTRY,
                i_field_raw,
                0,
                (raw_field, slot_idx_field, 2 * slot_idx, 2*NUM_SLOTS),
            )
        )

        df.add(
            gd.entry(
                gd.MPLEX_ENTRY,
                q_field_raw,
                0,
                (raw_field, slot_idx_field, 2 * slot_idx + 1, 2*NUM_SLOTS),
            )
        )

        # Create scaled complex field combining I and Q
        slot_field = f"m{module+1:02d}_slot{slot_num}"
        df.add(
            gd.entry(
                gd.LINCOM_ENTRY,
                slot_field,
                0,
                ((i_field_raw, q_field_raw), (1 / 256.0 + 0j, 1j / 256.0), (0, 0)),
            )
        )

        # Hide raw I/Q fields, keep the complex field visible
        df.hide(i_field_raw)
        df.hide(q_field_raw)

    # Add timebase field
    timebase_field = f"m{module+1:02d}_timebase"
    df.add(
        gd.entry(
            gd.LINCOM_ENTRY,
            timebase_field,
            0,
            ((ts_sbs_field, ts_ss_field), (1.0, 1 / 125e6), (0, 0)),
        )
    )

    # Flush metadata so dirfile can be read immediately
    df.metaflush()


if __name__ == "__main__":
    main()
