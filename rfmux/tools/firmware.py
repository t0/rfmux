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

The target board(s) are selected with --serial on the group (repeatable, or
"any"). Multiple boards are handled concurrently: each beaconing board that
matches gets its own worker thread. A targeted run exits once every named
serial is done; otherwise it listens until --timeout (default: never).
"""

import click
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
import textwrap
import threading
import time
import tty
import zlib

# optional dependencies - not mandated in pyproject.toml because they would
# bloat the Yocto build (where they are certainly unnecessary).
try:
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
    log.info("[%s] flash written, resetting", serial)

    child.sendline("reset")
    child.expect("resetting", timeout=5)
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

    child.sendline('reset')
    child.expect("resetting", timeout=5)
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
                 verbose=False):
    """Ack one board, run the requested action over netconsole, and log it.

    "action" is a callable taking (stream, child, serial) and returning bool.
    Runs in its own thread, so it reports its own outcome rather than raising.
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

    log.info("[%s] %s", serial, "complete" if ok else "fwdiscover operation FAILED")


@click.group()
@click.option("--serial", "serials", type=str, multiple=True, required=True,
              help='Board serial to act on (repeatable, or "any").')
@click.option("--discovery-port", "-p", type=int, default=DEFAULT_DISCOVERY_PORT,
              help=f"UDP beacon port (default: {DEFAULT_DISCOVERY_PORT}).")
@click.option("--bind", "-b", type=str, default="0.0.0.0",
              help="Address to bind the beacon listener to.")
@click.option("--verbose", "-v", is_flag=True,
              help="Echo netconsole traffic to stdout.")
@click.pass_context
def cli(ctx, serials, discovery_port, bind, verbose):
    """Firmware maintenance for CRS boards over netconsole + xmodem.

    \b
    Examples:
        rfmux firmware --serial any  repl reset
        rfmux firmware --serial 0110 repl
        rfmux firmware --serial 0110 reflash-spi boot.bin
        rfmux firmware --serial 0110 reflash-spi boot.bin --md5sum $(md5sum boot.bin | cut -d' ' -f1)
        rfmux firmware --serial 0110 reflash-mmc t0-crs-image.wic.gz
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
                   bind=bind, verbose=verbose)


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
    run(ctx.obj, functools.partial(reflash_qspi, path=str(path), md5=file_md5))


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
    try:
        run(ctx.obj, functools.partial(reflash_mmc, path=str(path),
                                       size=size, crc=crc))
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


@cli.command(name="repl")
@click.argument("command", required=False, default=None)
@click.pass_context
def repl_cmd(ctx, command):
    """Drive U-Boot over netconsole.

    With COMMAND, run it once. Without one, drop into an interactive console
    (press Ctrl-] to exit); this requires exactly one specific --serial.
    """
    serials = ctx.obj["serials"]
    if command is None and (serials == ("any",) or len(set(serials)) != 1):
        raise click.UsageError(
            "interactive repl needs exactly one specific --serial")
    run(ctx.obj, functools.partial(repl, cmd=command))


def run(opts, action):
    """Listen for beacons and run "action" against each matching board."""
    serials = opts["serials"]
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
    threads = {}        # serial -> worker Thread (one per handled board)

    while True:
        # If specific serials were named, we know we're done once every one of
        # them has been handled and its worker has finished. This is the normal
        # exit for a targeted run; the beacon listener otherwise runs forever.
        if (not serve_any
                and target_serials <= threads.keys()
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
        if serial not in seen:
            seen.add(serial)
            log.info("Found CRS board %s at %s:%d", serial, ip, port)

        if not serve_any and serial not in target_serials:
            continue
        if serial in threads:
            continue

        position = len(threads)
        thread = threading.Thread(
            target=handle_board,
            args=(ip, port, version, serial, action, position, opts["verbose"]),
            name=f"reflash-{serial}",
            daemon=True,
        )
        threads[serial] = thread
        thread.start()

    sock.close()


if __name__ == "__main__":
    cli()
