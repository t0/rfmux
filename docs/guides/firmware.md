# Firmware Management Guide

This guide covers fetching, managing, and flashing CRS firmware.

## Firmware Versions

Firmware is organized by release version:

```
firmware/
├── r1.5.4/
├── r1.5.5/
├── r1.5.6/
├── CHANGES          # Detailed firmware changelog
└── README.md
```

**Always check `firmware/CHANGES` before updating** to understand what's new:

```bash
cat firmware/CHANGES
```

## Prerequisites

Before fetching firmware, you need Git LFS installed.

### Install Git LFS

Follow the installation instructions at [git-lfs.github.com](https://git-lfs.github.com/) for your platform.

**Quick install:**
- **Linux:** `sudo apt-get install git-lfs`
- **macOS:** `brew install git-lfs`
- **Windows:** Download installer from [git-lfs.github.com](https://git-lfs.github.com/)

### Initialize Git LFS

Run once per repository after cloning:

```bash
git lfs install
```

## Fetching Firmware Files

Firmware files are distributed using Git LFS and are excluded by default to save bandwidth.

### Check Current State

When you first clone the repository, firmware files appear small because they're just LFS pointers:

```bash
rfmux/firmware$ ls -lh r1.5.6/rootfs.ext4.bz2
-rw-rw-r-- 1 user user 134 Jan 23 18:16 rootfs.ext4.bz2  # Too small!
```

View the pointer to see the actual file size:

```bash
rfmux/firmware$ cat r1.5.6/rootfs.ext4.bz2
version https://git-lfs.github.com/spec/v1
oid sha256:9fc580dd0fbb6b1b865d20762798b01dbb276dc3036c0e565aac02e1127119c1
size 156241632
```

### Download Firmware

**Pull specific version:**

```bash
cd rfmux/firmware
git lfs pull -X'' r1.5.6/*
```

The `-X''` (exclude nothing) flag ensures files are actually downloaded.

**Verify download:**

```bash
ls -lh r1.5.6/rootfs.ext4.bz2
-rw-rw-r-- 1 user user 149M Jan 23 18:21 rootfs.ext4.bz2  # Correct size!
```

**Pull all firmware versions** (warning: large download):

```bash
git lfs pull --exclude=
```

## Flashing MicroSD Cards

Firmware is distributed as compressed ext4 filesystem images that must be
written to MicroSD cards.  The upgrade flow is as follows:

1. Power down the CRS board
2. Remove the MicroSD card
3. Flash the card following instructions above
4. Reinsert the card into the CRS board
5. Power up and wait for boot (~30-60 seconds)
6. Verify board is accessible: `ping rfmux<serial>.local`

### Linux

#### Finding your Device

```bash
# Insert SD card, then check kernel messages:
sudo dmesg | tail -20

# Or list block devices:
lsblk
```

#### Unmounting Device Partitions

Some distributions (Mint, Ubuntu, but not Debian) will mount filesystems by default when you insert a flash card.
You must unmount these filesystems before proceeding, otherwise Linux's filesystem layer thinks it "owns" the devices and writes to the underlying image may be corrupted.
The `lsblk` command (see above) lists mounted partitions under the "MOUNTPOINTS" column.
You can unmount them using e.g. `sudo umount /dev/sdb1` (if your flash device is `/dev/sdb` and its partition `/dev/sdb1` is shown as a MOUNTPOINT).

#### Writing your Device

In the following commands, replace `DEVICE` with your SD card device (e.g., `sdb`). Use the device file (e.g., `/dev/sdb`), NOT partition files (e.g., `/dev/sdb1`).

**Warning:** Using the wrong device will destroy data on that drive - triple-check before running `dd`!

```bash
# Write firmware to SD card
bzcat r1.5.6/rootfs.ext4.bz2 | sudo dd of=/dev/DEVICE bs=1M status=progress

# Check and repair filesystem
sudo fsck -f /dev/DEVICE

# Resize to fill SD card
sudo resize2fs /dev/DEVICE

# Safely eject
sudo eject /dev/DEVICE
```

### macOS

```bash
# Find disk identifier
diskutil list

# Unmount (not eject)
diskutil unmountDisk /dev/diskN

# Write firmware
bzcat r1.5.6/rootfs.ext4.bz2 | sudo dd of=/dev/rdiskN bs=1m

# Eject
diskutil eject /dev/diskN
```

Use `/dev/rdiskN` (raw device) instead of `/dev/diskN` for faster writes.

### Windows

Windows cannot natively write ext4 filesystems. Options:

1. **Use WSL (Windows Subsystem for Linux):**
   - Access SD card via `/mnt/` path
   - Follow Linux instructions above

2. **Use third-party tools:**
   - [Rufus](https://rufus.ie/) (may work with raw images)
   - [Win32 Disk Imager](https://sourceforge.net/projects/win32diskimager/)

3. **Boot from Linux live USB** to flash cards

## Troubleshooting

### Git LFS files not downloading

**Symptoms:** Firmware files remain small (~130 bytes)

**Solution:**
```bash
git lfs install  # Ensure LFS is initialized
git lfs pull -X'' r1.5.6/*  # Force download
```

### SD card write fails

**Symptoms:** `dd` errors or incomplete writes

**Check:**
- SD card is not write-protected (physical switch)
- Sufficient disk space
- Card is not mounted (use `umount` first)
- Correct device path

### CRS won't boot after flash

**Check:**
- Firmware file downloaded completely (check file size)
- `fsck` and `resize2fs` completed without errors
- SD card is not corrupted (try a different card)
- Power supply is adequate (5V, ≥2A recommended)
