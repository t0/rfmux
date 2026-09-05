#!/usr/bin/env python3
"""
Firmware maintenance via netconsole

During boot, CRS boards broadcast FWREQ beacons. Ordinary (flash card) boot can
be interrupted by responding to these with FWRESP packets that redirect the
board's console to the network.  We can then drive U-Boot over that connection.

Subcommands:

  reflash-spi FILE     write a boot.bin to QSPI flash (sf erase / sf write)
  reflash-mmc IMAGE    write a WIC image (raw or gzipped) to MMC (whole card,
                       including /home)
  repl [COMMAND]       run one U-Boot command, or open an interactive console
  write-backplane-eeprom
                       program the crate backplane EEPROM behind each board's
                       slot with an IPMI FRU descriptor (a "--slot SERIAL=N"
                       mapping programs a whole crate in one run)
  read-backplane-eeprom
                       read back and decode the backplane FRU descriptor,
                       optionally verifying slot numbers against a mapping

The target board(s) are selected with --serial on the group (repeatable, or
"any"). --crate (repeatable) further restricts to boards that are within
a particular crate and can be paired with --serial any to target an entire
crate at once. Multiple boards are handled concurrently: each beaconing
board that matches gets its own worker thread. A targeted run exits once
every named serial is done; otherwise it listens until --timeout (default: never).

Every board ends up in the state named by --then (default: reset, i.e. the
board reboots). Subcommands may be chained in a single invocation; boards
reboot between stages, so each stage starts from a fresh U-Boot environment
and a fresh beacon:

    rfmux firmware --serial 0110 reflash-spi boot.bin reflash-mmc image.wic.gz
"""

import click
import datetime
import functools
import gzip
import hashlib
import logging
import os
import pathlib
import re
import select
import socket
import sys
import tempfile
import termios
import threading
import tty
import zlib

# optional dependencies - not mandated in pyproject.toml because they would
# bloat the Yocto build (where they are certainly unnecessary).
try:
    import fru
    import pexpect
    import pexpect.fdpexpect
    import tqdm
    import xmodem
    _HAVE_IMPORTS = True
except ImportError:
    # A proper error message is reported in the cli() callback
    _HAVE_IMPORTS = False


log = logging.getLogger(__name__)

DEFAULT_DISCOVERY_PORT = 9875
BUF_SIZE = 4096
PROMPT = "ZynqMP>"

class UDPStream:
    """A connected UDP socket presented as a file-like stream for pexpect."""

    def __init__(self, crs_ip):
        # Discover which local address routes to the board (sends nothing).
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((crs_ip, 1))
            self.ourip = probe.getsockname()[0]

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.ourip, 0))

        self.ncport = self.s.getsockname()[1]
        self.s.connect((crs_ip, self.ncport))

    def read(self, size):
        return self.s.recv(size)

    def write(self, data):
        return self.s.send(data)

    def fileno(self):
        return self.s.fileno()

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass  # fdspawn.close() may already have closed the underlying fd

class TqdmLogHandler(logging.Handler):
    """Route log records through tqdm.write so they don't corrupt progress bars."""

    def emit(self, record):
        try:
            tqdm.tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

class TqdmStdoutRedirect:
    """Redirects verbose output to tqdm write to not interrupt progress bars."""
    def __init__(self):
        self.buf = ""

    def write(self, data):
        self.buf += data
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            tqdm.tqdm.write(line.rstrip("\r"))

    def flush(self):
        pass

def send_xmodem(stream, child, serial, position, path):
    """Transfer a file to U-Boot's 'loadx' over xmodem on the raw socket."""

    child.sendline("loadx")
    child.flush()

    def getc(size, timeout=1):
        stream.s.settimeout(timeout)
        try:
            return stream.s.recv(size) or None
        except socket.timeout:
            return None

    def putc(data, timeout=1):
        return stream.s.send(data)

    modem = xmodem.XMODEM(getc, putc, mode="xmodem1k")  # 1024-byte packets

    # Transfers are long; show progress so users don't conclude the flash has
    # hung and power-cycle the board mid-write. One packet = 1 KiB.
    npackets = -(-os.path.getsize(path) // 1024)
    with open(path, "rb") as fh, \
            tqdm.tqdm(total=npackets, unit="KiB", unit_scale=True,
                      desc=f"[{serial}] xmodem",
                      position=position, leave=False) as bar:
        def progress(total, success, errors):
            bar.update(success - bar.n)
        if not modem.send(fh, callback=progress):
            return False

    # Consume loadx's post-transfer banner. Its size (and the $filesize env
    # variable) reflect the transfer padded to a whole 1024-byte block, not
    # the true file size, so it isn't worth validating; the transcripts check
    # content instead, and splice in the exact size where it matters.
    child.expect(r"## Total Size")
    child.expect(PROMPT)
    return True


# ---------------------------------------------------------------------------
# Flash transcripts
#
# These are deliberately written out in full rather than factored through a
# shared script runner: there are only a couple of transcripts that matter,
# and keeping them distinct makes each independently editable without
# regressing the other.
# ---------------------------------------------------------------------------

def reflash_qspi(stream, child, serial, position, path, md5):
    """Write a boot.bin to QSPI flash, then reset. Returns True on success."""
    filesize = os.path.getsize(path)
    # Round the erase region up to a whole number of 128 KiB erase blocks.
    erase_size = hex(-(-filesize // 0x20000) * 0x20000)

    log.info("[%s] transferring %s (%d bytes) over xmodem", serial, path, filesize)
    if not send_xmodem(stream, child, serial, position, path):
        log.error("[%s] xmodem transfer failed", serial)
        return False

    # Use the exact size, not $filesize: loadx sets $filesize to the transfer
    # padded to a whole xmodem block, which would break the md5 comparison.
    child.sendline(f"md5sum $loadaddr {filesize:#x}")
    child.expect(md5, timeout=5)
    child.expect(PROMPT)
    log.info("[%s] md5 verified: %s", serial, md5)

    child.sendline("sf probe 0:0")
    child.expect("erase size 128 KiB")
    child.expect(PROMPT)

    child.sendline(f"sf erase 0x0 {erase_size}")
    child.expect("Erased: OK", timeout=30)
    child.expect(PROMPT)
    log.info("[%s] flash erased", serial)

    child.sendline(f"sf write $loadaddr 0x0 {filesize:#x}")
    child.expect("Written: OK", timeout=30)
    child.expect(PROMPT)
    log.info("[%s] flash written", serial)
    return True


def reflash_mmc(stream, child, serial, position, path, size, crc):
    """Write a compressed WIC image to MMC. Returns True on success.

    "size" and "crc" are the image's uncompressed byte count and CRC32,
    computed locally; gzwrite reports both on completion and we require them
    to match.

    The whole card is overwritten, including /home. (Preserving /home would
    mean divining the image's partition geometry somewhere; deliberately
    punted for now.)
    """
    filesize = os.path.getsize(path)

    # Probe the card before the (long) transfer; "mmc dev" re-initializes it
    # and only prints "is current device" on success. A missing or dead card
    # produces an error message and a bare prompt instead.
    child.sendline("mmc dev 0")
    if child.expect(["is current device", PROMPT], timeout=10) != 0:
        log.error("[%s] no usable MMC card detected: %s",
                  serial, child.before.strip())
        return False
    child.expect(PROMPT)
    log.info("[%s] MMC card detected", serial)

    log.info("[%s] transferring %s (%d bytes) over xmodem", serial, path, filesize)
    if not send_xmodem(stream, child, serial, position, path):
        log.error("[%s] xmodem transfer failed", serial)
        return False

    child.sendline(f"gzwrite mmc 0 $loadaddr {filesize:#x}")

    # gzwrite reports progress as "written/total" lines, which we relay to a
    # progress bar, and finishes with e.g. "1158281216 bytes, crc 0xa24d3d10",
    # which must match the local decompression pass so a short or corrupt
    # write can't pass silently. A failed gzwrite skips the completion line
    # and lands straight on the prompt. The timeout only needs to cover the
    # gap between progress reports, so a stalled write fails quickly.
    with tqdm.tqdm(total=size, unit="B", unit_scale=True,
                   desc=f"[{serial}] gzwrite",
                   position=position, leave=False) as bar:
        while True:
            matched = child.expect([r"(\d+)/\d+",
                                    r"(\d+) bytes, crc 0x([0-9a-fA-F]+)",
                                    PROMPT], timeout=60)
            if matched == 0:
                bar.update(max(0, int(child.match.group(1)) - bar.n))
            elif matched == 1:
                written = int(child.match.group(1))
                written_crc = int(child.match.group(2), 16)
                break
            else:
                log.error("[%s] gzwrite failed: %s",
                          serial, child.before.strip())
                return False
    child.expect(PROMPT)
    if (written, written_crc) != (size, crc):
        log.error("[%s] gzwrite mismatch: board reports %d bytes (crc %#010x), "
                  "expected %d bytes (crc %#010x)",
                  serial, written, written_crc, size, crc)
        return False
    log.info("[%s] image written to MMC: %d bytes, crc %#010x",
             serial, size, crc)
    return True


def repl(stream, child, serial, position, cmd=None):
    """Drive U-Boot over netconsole.

    With a command, run it once and return. Without one, relay
    stdin<->netconsole raw until the user presses Ctrl-]: the terminal is put
    in raw mode so keystrokes (including Ctrl-C, which U-Boot needs to
    interrupt autoboot) pass straight through to the board. Ctrl-] is the sole
    local escape (as in telnet); it is caught here before forwarding, so it
    exits even if the board is unresponsive.
    """

    # Ctrl-]
    ESCAPE = 0x1D

    # A bare LF (not already part of a CRLF) needing CRLF fixup on a raw terminal.
    CRLF = re.compile(rb"(?<!\r)\n")

    if cmd is not None:
        log.info("[%s] executing: %s", serial, cmd)
        child.sendline(cmd)
        child.expect(PROMPT, timeout=30)
        log.info("[%s] output:\n%s", serial, child.before.strip())
        return True

    if not sys.stdin.isatty():
        raise click.ClickException("interactive repl needs an interactive terminal")

    sys.stdout.write(f"\r\nInteractive netconsole to CRS {serial}: Ctrl-] exits\r\n")
    sys.stdout.flush()

    sock = stream.s
    sock.setblocking(False)
    stdin_fd = sys.stdin.fileno()
    old = termios.tcgetattr(stdin_fd)
    try:
        tty.setraw(stdin_fd)
        # Nudge U-Boot to reprint its prompt on the now-raw terminal.
        sock.send(b"\n")
        while True:
            readable, _, _ = select.select([stdin_fd, sock], [], [])
            if stdin_fd in readable:
                keys = os.read(stdin_fd, BUF_SIZE)
                if ESCAPE in keys:
                    keys = keys[:keys.index(ESCAPE)]
                    if keys:
                        sock.send(keys)
                    break
                sock.send(keys)
            if sock in readable:
                try:
                    data = sock.recv(BUF_SIZE)
                except BlockingIOError:
                    continue
                if not data:
                    break
                # Raw terminal: no ONLCR, and U-Boot emits bare LFs. Map LF to
                # CRLF (leaving any existing CR alone) so lines don't staircase.
                os.write(sys.stdout.fileno(), CRLF.sub(b"\r\n", data))
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old)
        sys.stdout.write("\r\n--- netconsole closed ---\r\n")
        sys.stdout.flush()
    return True


# ---------------------------------------------------------------------------
# Backplane EEPROM
#
# Each crate slot carries its own EEPROM on the backplane, wired to the i2c0
# bus of whatever CRS board occupies it (the rest of the board's I2C tree
# hangs off i2c1). The EEPROM holds an IPMI FRU descriptor for the crate; the
# slot number is stored as a chassis-area custom field ("slot=N"), which is
# the only way a board can discover which slot it occupies.
# ---------------------------------------------------------------------------

# Minutes since this epoch is the FRU board-area manufacturing-date encoding.
FRU_EPOCH = datetime.datetime(1996, 1, 1, tzinfo=datetime.timezone.utc)

# Known backplane designs, selected with "--backplane".
BACKPLANES = {
    "4sbp": {
        "eeprom": {"address": 0x51, "size": 256}, # 24lc025t (2kbit)
        "chassis": {"type": 0x17}, # rack-mount chassis
        "board": {
            "manufacturer": "t0.technology",
            "product_name": "4-slot backplane",
            "part_number": "4SBP",
        },
    },
}

SLOT_FIELD = re.compile(r"slot=(\d+)$")

# One line of "i2c md" output: a 16-bit offset, then hex byte pairs (the
# trailing ASCII column can't false-match: it is set off by two or more
# consecutive spaces, which the single-space byte separator won't cross).
HEXDUMP_LINE = re.compile(r"^([0-9A-Fa-f]{4}):((?: [0-9A-Fa-f]{2})+)", re.M)


def select_backplane_eeprom(child, serial, bus, chip):
    """Select the backplane I2C bus and confirm the EEPROM answers a probe."""

    child.sendline(f"i2c dev {bus}")
    child.expect(PROMPT)
    if "Failure" in child.before:
        log.error("[%s] cannot select I2C bus %d: %s",
                  serial, bus, child.before.strip())
        return False

    child.sendline("i2c probe")
    child.expect(PROMPT, timeout=30)
    found = re.search(r"Valid chip addresses:((?: [0-9A-Fa-f]{2})*)", child.before)
    chips = [int(c, 16) for c in found.group(1).split()] if found else []
    if chip not in chips:
        log.error("[%s] no EEPROM at %#04x on I2C bus %d (probe found: %s). "
                  "Is the board seated in a crate?",
                  serial, chip, bus,
                  ", ".join(f"{c:#04x}" for c in chips) or "nothing")
        return False
    return True


def write_backplane_eeprom(stream, child, serial, position, images, size,
                           bus, chip, alen):
    """Write a FRU image to the backplane EEPROM and verify it by readback.

    "images" maps each targeted board serial to its (path, md5, slot): the
    boards in a crate are programmed concurrently, each with its own slot.
    """
    path, md5, slot = images[serial]

    if not select_backplane_eeprom(child, serial, bus, chip):
        return False

    log.info("[%s] transferring slot-%d FRU image (%d bytes) over xmodem",
             serial, slot, size)
    if not send_xmodem(stream, child, serial, position, path):
        log.error("[%s] xmodem transfer failed", serial)
        return False

    # Use the exact size, not $filesize: loadx sets $filesize to the transfer
    # padded to a whole xmodem block.
    child.sendline(f"md5sum $loadaddr {size:#x}")
    child.expect(md5, timeout=5)
    child.expect(PROMPT)

    log.info("[%s] writing EEPROM at %#04x on I2C bus %d", serial, chip, bus)
    child.sendline(f"i2c write $loadaddr {chip:#x} 0.{alen} {size:#x}")
    child.expect(PROMPT, timeout=5)
    if "Error" in child.before:
        log.error("[%s] EEPROM write failed: %s", serial, child.before.strip())
        return False

    # Read back through the EEPROM (into scratch memory well past the image)
    # and require the md5 to match, so a write-protected or flaky part can't
    # pass silently.
    child.sendline(f"setexpr bpaddr $loadaddr + {2*size:#x}")
    child.expect(PROMPT)
    child.sendline(f"i2c read {chip:#x} 0.{alen} {size:#x} $bpaddr")
    child.expect(PROMPT, timeout=5)
    child.sendline(f"md5sum $bpaddr {size:#x}")
    if child.expect([md5, PROMPT], timeout=5) != 0:
        log.error("[%s] EEPROM readback mismatch: %s",
                  serial, child.before.strip())
        return False
    child.expect(PROMPT)
    log.info("[%s] EEPROM readback verified (md5 %s)", serial, md5)
    return True


def read_backplane_eeprom(stream, child, serial, position, size, bus, chip,
                          alen, expected):
    """Read the backplane EEPROM and decode its FRU descriptor.

    "expected" maps board serials to the slot number each EEPROM should
    claim; a board present in the map fails unless its EEPROM agrees.
    """
    if not select_backplane_eeprom(child, serial, bus, chip):
        return False

    child.sendline(f"i2c md {chip:#x} 0.{alen} {size:#x}")
    child.expect(PROMPT, timeout=5)
    blob = b"".join(bytes.fromhex(m.group(2).replace(" ", ""))
                    for m in HEXDUMP_LINE.finditer(child.before))
    if len(blob) != size:
        log.error("[%s] short EEPROM dump: expected %d bytes, parsed %d",
                  serial, size, len(blob))
        return False

    try:
        data = fru.load(blob=blob)
    except ValueError as e:
        log.error("[%s] EEPROM contents are not a valid FRU descriptor (%s)",
                  serial, e)
        return False

    slot = next((int(m.group(1))
                 for f in data.get("chassis", {}).get("custom_fields", [])
                 if (m := SLOT_FIELD.match(f))), None)
    if slot is None:
        log.warning('[%s] FRU descriptor has no "slot=N" chassis custom field',
                    serial)

    # Render the decoded FRU as indented text for the log.
    lines = []
    for area in ("chassis", "board", "product"):
        if area not in data:
            continue
        lines.append(f"  {area}:")
        for key, value in data[area].items():
            if key == "format_version" or value in ("", [], 0):
                continue
            if key == "mfg_date_time":
                value = (FRU_EPOCH + datetime.timedelta(minutes=value)
                         ).strftime("%Y-%m-%d %H:%M UTC")
            elif key == "custom_fields":
                value = ", ".join(value)
            lines.append(f"    {key}: {value}")
    log.info("[%s] backplane FRU (slot %s):\n%s",
             serial, slot if slot is not None else "unknown", "\n".join(lines))

    expect = expected.get(serial)
    if expect is not None:
        if slot != expect:
            log.error("[%s] slot mismatch: EEPROM says %s, expected %d",
                      serial, slot if slot is not None else "nothing", expect)
            return False
        log.info("[%s] slot %d verified", serial, slot)
    return True


def validate_boot_bin(data, path):
    """Sanity-check that data looks like a complete ZynqMP boot.bin."""

    def word(offset):
        if offset + 4 > len(data):
            raise click.ClickException(
                f"{path}: truncated (header read past EOF at {offset:#x})")
        return int.from_bytes(data[offset:offset + 4], "little")

    if word(0x20) != (ZYNQMP_WIDTH_DETECTION := 0xaa995566):
        raise click.ClickException(
            f"{path}: bad width detection word at 0x20: "
            f"{word(0x20):#010x} (expected {ZYNQMP_WIDTH_DETECTION:#010x})")

    if word(0x24) != (ZYNQMP_XLNX_ID := 0x584c4e58):
        raise click.ClickException(
            f"{path}: bad identification word at 0x24: "
            f"{word(0x24):#010x} (expected {ZYNQMP_XLNX_ID:#010x})")

    # There is no total-size field, but the boot header (0x98, bytes) points
    # at an image header table, which points at a chain of partition headers
    # whose payloads must all lie within the file - walk them to catch
    # truncation. Layout per U-Boot tools/zynqmpimage.h; offsets and lengths
    # are in 32-bit words.
    iht = word(0x98)
    part = word(iht + 0x08) * 4
    seen = set()
    while part and part not in seen:
        seen.add(part)
        end = (word(part + 0x20) + word(part + 0x08)) * 4
        if end > len(data):
            raise click.ClickException(
                f"{path}: truncated: partition (header at {part:#x}) ends at "
                f"{end:#x} but the file is only {len(data):#x} bytes")
        part = word(part + 0x0c) * 4


def parse_fwreq(data):
    """Parse an FWREQ datagram into (version, headers). Raise ValueError if bad."""
    text = data.decode("utf-8")
    head, _, _ = text.partition("\n\n")
    lines = head.split("\n")

    status = lines[0]
    if not status.startswith("FWREQ/"):
        raise ValueError(f"not an FWREQ message: {status!r}")
    version = int(status.split("/", 1)[1])

    headers = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        headers[key.strip()] = value.strip().strip('"')

    return version, headers


def handle_board(crs_ip, crs_port, beacon_version, serial, action, position,
                 verbose, then, on_done):
    """Ack one board, run the requested action over netconsole, and log it.

    "action" is a callable taking (stream, child, serial, position) and
    returning bool. After a successful action, "then" names the board's
    terminal state: "reset" reboots it, "prompt" leaves it at U-Boot. Runs
    in its own thread, so it reports its outcome through "on_done" rather
    than raising.
    """
    try:
        stream = UDPStream(crs_ip)
        try:
            # Redirect the board's netconsole to our socket. The FWRESP goes
            # to the board's beacon source port; the console conversation then
            # flows over the connected socket.
            fwresp = (
                f"FWRESP/{beacon_version}\n"
                f"ncip:{stream.ourip}\n"
                f"ncinport:{stream.ncport}\n"
                f"ncoutport:{stream.ncport}\n\n"
            ).encode("utf-8")
            stream.s.sendto(fwresp, (crs_ip, crs_port))

            child = pexpect.fdpexpect.fdspawn(stream.fileno(), timeout=10,
                                      encoding="utf-8")
            if verbose:
                # Everything the board sends (which echoes our input back)
                # lands on stdout. xmodem traffic bypasses pexpect and stays
                # out of the log.
                child.logfile_read = TqdmStdoutRedirect()

            # Wake U-Boot and land on a fresh prompt before the transcript.
            stream.s.settimeout(0.5)
            while True:
                try:
                    if not stream.s.recv(BUF_SIZE):
                        break
                except socket.timeout:
                    break
            child.flush()
            child.sendline("")
            child.expect(PROMPT)

            ok = action(stream, child, serial, position)
            if ok and then == "reset":
                child.sendline("reset")
                child.expect("resetting", timeout=5)
        finally:
            # fdspawn borrows the socket's fd; let the stream own teardown so we
            # don't close the fd out from under the socket object.
            stream.close()
    except (pexpect.TIMEOUT, pexpect.EOF) as e:
        log.error("[%s] netconsole conversation failed: %r\n  last output: %r",
                  serial, e, getattr(e, "value", None))
        ok = False
    except Exception:
        log.exception("[%s] worker crashed", serial)
        ok = False

    on_done(serial, ok)
    log.info("[%s] %s", serial, "complete" if ok else "fwdiscover operation FAILED")


@click.group(chain=True)
@click.option("--serial", "serials", type=str, multiple=True, required=True,
              help='Board serial to act on (repeatable, or "any").')
@click.option("--discovery-port", "-p", type=int, default=DEFAULT_DISCOVERY_PORT,
              help=f"UDP beacon port (default: {DEFAULT_DISCOVERY_PORT}).")
@click.option("--bind", "-b", type=str, default="0.0.0.0",
              help="Address to bind the beacon listener to.")
@click.option("--verbose", "-v", is_flag=True,
              help="Echo netconsole traffic to stdout.")
@click.option("--then", type=click.Choice(["reset", "prompt"]),
              default="reset", show_default=True,
              help="Board state after the final command: reboot, or sitting "
                   "at the U-Boot prompt. Chained commands always reboot in "
                   "between.")
@click.option("--crate", "crates", type=str, multiple=True, required=False,
              help='Crate serial number to restrict to (repeatable),')
@click.pass_context
def cli(ctx, serials, discovery_port, bind, verbose, then, crates):
    """Firmware maintenance for CRS boards over netconsole + xmodem.

    Commands may be chained in one invocation; each board reboots between
    commands and every command starts from a fresh beacon.

    \b
    Examples:
        rfmux firmware --serial 0110 repl
        rfmux firmware --serial 0110 reflash-spi boot.bin
        rfmux firmware --serial 0110 reflash-spi boot.bin --md5sum $(md5sum boot.bin | cut -d' ' -f1)
        rfmux firmware --serial 0110 reflash-spi boot.bin reflash-mmc t0-crs-image.wic.gz
        rfmux firmware --serial any  --crate 0123 reflash-spi boot.bin
        rfmux firmware --serial 0110 write-backplane-eeprom --backplane 4sbp --chassis-serial-number C0021 --slot 3
        rfmux firmware --serial any  write-backplane-eeprom --backplane 4sbp --chassis-serial-number C0021 --slot 0110=1 --slot 0111=2
        rfmux firmware --serial any  read-backplane-eeprom --backplane 4sbp --slot 0110=1 --slot 0111=2
    """
    # Optional dependencies, checked here (the single gateway to every
    # subcommand) so a missing package fails with a remedy now, not mid-flash
    # after the user has power-cycled a board to catch its boot window.
    if not _HAVE_IMPORTS:
        raise click.ClickException(
            f"The \"rfmux firmware\" command's Python dependencies are optional.\n"
            f"Install them with 'pip install rfmux[firmware]'.")

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[TqdmLogHandler()]
    )
    # xmodem logs each board-echoed byte as an ERROR while waiting for the
    # handshake; those are benign retries (a real failure returns False and is
    # reported by our own logging).
    logging.getLogger("xmodem.XMODEM").setLevel(logging.CRITICAL)

    ctx.obj = dict(serials=serials, discovery_port=discovery_port,
                   bind=bind, verbose=verbose, then=then, crate=crates)


@cli.result_callback()
@click.pass_context
def run_actions(ctx, actions, **_):
    """Run the actions collected from the chained subcommands."""
    run(ctx.obj, actions)


@cli.command(name="reflash-spi")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--md5sum", type=str, default=None, metavar="HASH",
              help="Verify the image's MD5 matches this before flashing.")
@click.pass_context
def reflash_spi_cmd(ctx, file, md5sum):
    """Write a boot.bin to QSPI flash."""
    path = pathlib.Path(file)
    data = path.read_bytes()
    validate_boot_bin(data, path)
    file_md5 = hashlib.md5(data).hexdigest()
    if md5sum is not None and file_md5 != md5sum.strip().lower():
        raise click.ClickException(
            f"File MD5 {file_md5} does not match expected {md5sum.strip().lower()}")
    log.info("QSPI reflash: %s (%d bytes, MD5 %s)", path, len(data), file_md5)
    return functools.partial(reflash_qspi, path=str(path), md5=file_md5)


@cli.command(name="reflash-mmc")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--md5sum", type=str, default=None, metavar="HASH",
              help="Verify the image's MD5 matches this before flashing.")
@click.pass_context
def reflash_mmc_cmd(ctx, image, md5sum):
    """Write a WIC image to MMC (whole card, including /home).

    IMAGE may be gzip-compressed (*.wic.gz) or raw; a raw image is compressed
    into a temporary file first, since the board-side write is U-Boot's
    gzwrite (and the transfer is far slower per byte than gzip is).
    """
    path = pathlib.Path(image)

    # Validate md5sum locally, always against the file the user named.
    if md5sum is not None:
        file_md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if file_md5 != md5sum.strip().lower():
            raise click.ClickException(
                f"File MD5 {file_md5} does not match expected {md5sum.strip().lower()}")

    with open(path, "rb") as fh:
        gzipped = fh.read(2) == b"\x1f\x8b"

    # Both branches read the whole image once, ending with a known-good gzip
    # plus the uncompressed size and CRC32 that the transcript checks against
    # gzwrite's completion report.
    size, crc = 0, 0
    tmp = None
    if gzipped:
        # Decompressing verifies the trailing CRC32/length record, which a
        # truncated download would fail.
        try:
            with gzip.open(path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    size += len(chunk)
                    crc = zlib.crc32(chunk, crc)
        except (OSError, EOFError, zlib.error):
            raise click.ClickException(
                f"{path}: corrupt or truncated gzip file.")
    else:
        # xmodem needs the compressed size up front, so compress into a
        # temporary file (honouring TMPDIR) rather than streaming.
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-", suffix=".wic.gz", delete=False)
        try:
            with open(path, "rb") as src, \
                    gzip.open(tmp, "wb", compresslevel=6) as dst, \
                    tqdm.tqdm(total=path.stat().st_size, unit="B",
                              unit_scale=True, desc=f"gzip {path.name}") as bar:
                while chunk := src.read(1 << 20):
                    size += len(chunk)
                    crc = zlib.crc32(chunk, crc)
                    dst.write(chunk)
                    bar.update(len(chunk))
            tmp.close()
        except BaseException:
            os.unlink(tmp.name)
            raise
        path = pathlib.Path(tmp.name)
        log.info("Compressed to %s (%d -> %d bytes)",
                 path, size, path.stat().st_size)

    log.info("MMC reflash: %s (%d bytes uncompressed, crc %#010x)",
             path, size, crc)
    if tmp is not None:
        # The action runs from the group's result callback, so tie the
        # temporary file's lifetime to the root context, not this callback.
        ctx.find_root().call_on_close(functools.partial(os.unlink, tmp.name))
    return functools.partial(reflash_mmc, path=str(path), size=size, crc=crc)


@cli.command(name="repl")
@click.argument("command", required=False, default=None)
@click.pass_context
def repl_cmd(ctx, command):
    """Drive U-Boot over netconsole.

    With COMMAND, run it once. Without one, drop into an interactive console
    (press Ctrl-] to exit); this requires exactly one specific --serial, and
    always ends at the U-Boot prompt regardless of --then - the session is
    setup for a human, who can reset the board themselves.
    """
    serials = ctx.obj["serials"]
    if command is None and (serials == ("any",) or len(set(serials)) != 1):
        raise click.UsageError(
            "interactive repl needs exactly one specific --serial")
    action = functools.partial(repl, cmd=command)
    if command is None:
        action.then = "prompt"
    return action


def parse_slot_map(slots, serials):
    """Parse --slot occurrences into a {serial: slot} mapping.

    Two forms: a single bare "N" (which needs exactly one specific --serial
    to attach to) or repeated "SERIAL=N" mappings for programming a whole
    crate in one run.
    """
    def slot_number(text):
        try:
            return int(text)
        except ValueError:
            raise click.UsageError(f"slot number {text!r} is not an integer")

    bare = [s for s in slots if "=" not in s]
    if bare and len(bare) != len(slots):
        raise click.UsageError(
            '--slot forms cannot be mixed: use a single bare "N" or '
            'repeated "SERIAL=N"')

    if bare:
        if len(bare) != 1:
            raise click.UsageError(
                "only one bare --slot N is meaningful; use --slot SERIAL=N "
                "to address several boards")
        if serials == ("any",) or len(set(serials)) != 1:
            raise click.UsageError(
                'a bare "--slot N" needs exactly one specific --serial; use '
                '"--slot SERIAL=N" to address a full crate')
        return {serials[0]: slot_number(bare[0])}

    mapping = {}
    for item in slots:
        serial, _, slot = item.partition("=")
        if not serial or serial == "any":
            raise click.UsageError(f"--slot {item}: missing board serial")
        if serial in mapping:
            raise click.UsageError(
                f"--slot {item}: serial {serial} is mapped more than once")
        mapping[serial] = slot_number(slot)

    claimants = {}
    for serial, slot in mapping.items():
        claimants.setdefault(slot, []).append(serial)
    duplicates = {slot: sers for slot, sers in claimants.items()
                  if len(sers) > 1}
    if duplicates:
        raise click.UsageError(
            "each slot may be mapped to only one serial: " + "; ".join(
                f"slot {slot} claimed by {', '.join(sers)}"
                for slot, sers in sorted(duplicates.items())))
    return mapping


def reconcile_slot_serials(opts, mapping):
    """Make a slot map the run's target list, or check it matches --serial.

    "--serial any" with a SERIAL=N map means "target exactly the mapped
    boards" (which also lets the run terminate once they're all done); an
    explicit --serial list must name exactly the mapped boards.
    """
    if opts["serials"] == ("any",):
        opts["serials"] = tuple(sorted(mapping))
    elif set(opts["serials"]) != set(mapping):
        raise click.UsageError(
            "--slot SERIAL=N mappings must cover exactly the boards named "
            "with --serial (or use --serial any to target the mapped boards)")


backplane_option = click.option(
    "--backplane", type=click.Choice(sorted(BACKPLANES)), required=True,
    help="Backplane design. Pins every FRU field and the EEPROM geometry "
         "that are properties of the design, so EEPROM contents stay "
         "consistent across crates.")


@cli.command(name="write-backplane-eeprom")
@backplane_option
@click.option("--slot", "slots", multiple=True, required=True,
              metavar="[SERIAL=]N",
              help='Crate slot number, stored as the "slot=N" chassis-area '
                   "custom field. Each slot's EEPROM is the sole record of "
                   "its position, so this is how boards learn their slot. "
                   "Give a bare N with a single --serial, or repeat "
                   "SERIAL=N to program a whole crate in one run.")
@click.option("--chassis-serial-number", required=True,
              help="Serial number of the crate being commissioned.")
@click.option("--board-serial-number", default="",
              help="Serial number of the backplane PCB, if tracked.")
@click.pass_context
def write_backplane_eeprom_cmd(ctx, backplane, slots, chassis_serial_number,
                               board_serial_number):
    """Commission crate backplane EEPROM(s) with IPMI FRU descriptors.

    The design-dependent FRU content (manufacturer, part numbers, EEPROM
    geometry) comes from the named --backplane; only per-crate data (slot
    mapping and serial numbers) is given here. Each written image is read
    back and verified before the board reports success.

    With a repeated "--slot SERIAL=N" mapping, every mapped board is
    programmed in a single run (one crate power-cycle), each receiving the
    shared fields plus its own slot.
    """
    mapping = parse_slot_map(slots, ctx.obj["serials"])
    reconcile_slot_serials(ctx.obj, mapping)

    design = BACKPLANES[backplane]
    eeprom_size = design["eeprom"]["size"]
    mfg_date_time = int(
        (datetime.datetime.now(datetime.timezone.utc) - FRU_EPOCH)
        .total_seconds() // 60)

    def build_image(slot):
        data = {
            "common": {"format_version": 1, "size": eeprom_size},
            "chassis": dict(design["chassis"],
                            serial_number=chassis_serial_number,
                            custom_fields=[f"slot={slot}"]),
            "board": dict(design["board"],
                          serial_number=board_serial_number,
                          mfg_date_time=mfg_date_time),
        }
        try:
            return fru.dump(data)
        except ValueError as e:
            raise click.ClickException(f"cannot encode FRU image: {e}")

    alen = 1 if eeprom_size <= 256 else 2
    # The action runs from the group's result callback, so tie the image
    # files' lifetime to the root context, not this callback.
    tmpdir = tempfile.TemporaryDirectory(prefix="backplane-fru-")
    ctx.find_root().call_on_close(tmpdir.cleanup)
    images = {}
    for serial, slot in sorted(mapping.items()):
        blob = build_image(slot)
        path = os.path.join(tmpdir.name, f"{serial}.bin")
        with open(path, "wb") as fh:
            fh.write(blob)
        images[serial] = (path, hashlib.md5(blob).hexdigest(), slot)
        log.info("Backplane EEPROM write: %s -> slot %d (md5 %s)",
                 serial, slot, images[serial][1])
    # The backplane hangs off the CRS board's I2C0 - board wiring, not a
    # design parameter.
    return functools.partial(write_backplane_eeprom, images=images,
                             size=eeprom_size, bus=0,
                             chip=design["eeprom"]["address"], alen=alen)


@cli.command(name="read-backplane-eeprom")
@backplane_option
@click.option("--slot", "slots", multiple=True, metavar="[SERIAL=]N",
              help="Expected slot number: the board fails unless its EEPROM "
                   "agrees. Give a bare N with a single --serial, or repeat "
                   "SERIAL=N to verify a whole crate in one run.")
@click.pass_context
def read_backplane_eeprom_cmd(ctx, backplane, slots):
    """Read and decode the backplane EEPROM's IPMI FRU descriptor.

    Reports the decoded FRU areas, including the crate slot number carried
    in the "slot=N" chassis-area custom field. With --slot, also verifies
    the stored slot number(s) against the expected value(s) - the readback
    counterpart to a whole-crate write-backplane-eeprom run.
    """
    expected = {}
    if slots:
        expected = parse_slot_map(slots, ctx.obj["serials"])
        reconcile_slot_serials(ctx.obj, expected)
    design = BACKPLANES[backplane]
    eeprom_size = design["eeprom"]["size"]
    alen = 1 if eeprom_size <= 256 else 2
    return functools.partial(read_backplane_eeprom, size=eeprom_size,
                             bus=0, chip=design["eeprom"]["address"],
                             alen=alen, expected=expected)


def run(opts, actions):
    """Listen for beacons and walk each matching board through "actions".

    Each board runs the actions in order, rebooting in between: a completed
    stage ends in a reset, the board beacons again on its next pass through
    U-Boot, and the following stage picks it up from a fresh environment.
    The final stage instead ends in the state named by --then. A failed
    stage abandons that board's remaining stages.
    """
    serials = opts["serials"]
    crates = opts["crate"]
    serve_any = "any" in serials
    target_serials = set() if serve_any else set(serials)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(1.0)
    sock.bind((opts["bind"], opts["discovery_port"]))

    if serve_any:
        log.info("Listening on port %d for any CRS board", opts["discovery_port"])
    else:
        log.info("Listening on port %d for boards: %s",
                 opts["discovery_port"], ", ".join(sorted(target_serials)))

    seen = set()        # serials we've already logged a beacon for
    threads = {}        # serial -> current/most recent worker Thread
    positions = {}      # serial -> stable tqdm progress-bar row
    progress = {}       # serial -> index of its next action
    failed = set()      # serials whose remaining actions were abandoned

    def on_done(serial, ok):
        if ok:
            progress[serial] += 1
        else:
            failed.add(serial)

    def finished(serial):
        return serial in failed or progress.get(serial, 0) >= len(actions)

    while True:
        # If specific serials were named, we know we're done once every one
        # of them has run out of actions (or failed) and its worker has
        # finished. This is the normal exit for a targeted run; the beacon
        # listener otherwise runs forever.
        if (not serve_any
                and target_serials <= progress.keys()
                and all(finished(s) for s in target_serials)
                and not any(t.is_alive() for t in threads.values())):
            log.info("All requested boards done, exiting")
            break

        try:
            data, (ip, port) = sock.recvfrom(BUF_SIZE)
        except socket.timeout:
            continue

        try:
            version, headers = parse_fwreq(data)
        except (ValueError, UnicodeDecodeError) as e:
            log.warning("Ignoring malformed beacon from %s:%d: %s", ip, port, e)
            continue

        serial = headers.get("Serial")
        crate_serial = headers.get("Crate Serial")
        if serial not in seen:
            seen.add(serial)
            log.info("Found CRS board %s at %s:%d", serial, ip, port)

        if crates and crate_serial not in crates:
            continue
        if not serve_any and serial not in target_serials:
            continue
        if serial in threads and threads[serial].is_alive():
            continue
        if finished(serial):
            continue

        index = progress.setdefault(serial, 0)
        # Chained stages are separated by a reboot so each starts from a
        # fresh U-Boot environment; only the final stage honours --then. An
        # action carrying its own "then" (interactive repl) overrides both.
        then = opts["then"] if index == len(actions) - 1 else "reset"
        then = getattr(actions[index], "then", then)
        positions.setdefault(serial, len(positions))
        if len(actions) > 1:
            log.info("[%s] starting stage %d/%d", serial, index + 1, len(actions))
        thread = threading.Thread(
            target=handle_board,
            args=(ip, port, version, serial, actions[index],
                  positions[serial], opts["verbose"], then, on_done),
            name=f"board-{serial}",
            daemon=True,
        )
        threads[serial] = thread
        thread.start()

    sock.close()


if __name__ == "__main__":
    cli()
