# CRS Firmware

## Fetching

Firmware files are distributed using git-lfs and are excluded by default.
So, at first, you should expect to see

    rfmux/firmware$ ls -ls r1.5.5/rootfs.ext4.bz2
    4 -rw-rw-r-- 1 gsmecher gsmecher 134 Jan 23 18:16 rootfs.ext4.bz2

This is too small for a firmware file - it's actually just a pointer to where
the file is actually stored in git-lfs.

    rfmux/firmware$ cat r1.5.5/rootfs.ext4.bz2
    version https://git-lfs.github.com/spec/v1
    oid sha256:9fc580dd0fbb6b1b865d20762798b01dbb276dc3036c0e565aac02e1127119c1
    size 156241632

In order to retrieve the file, you need to explicitly name it:

    rfmux/firmware$ git lfs pull -X'' r1.5.5/*

This results in a file large enough to use:

    rfmux/firmware$ ls -ls r1.5.5/rootfs.ext4.bz2
    152580 -rw-rw-r-- 1 gsmecher gsmecher 156241632 Jan 23 18:21 rootfs.ext4.bz2

## Updating the MicroSD Card

Firmware is written to a MicroSD card as follows:

    $ bzcat r1.5.5/rootfs.ext4.bz2 | sudo dd of=/dev/DEVICE bs=1M
    $ sudo fsck -f /dev/DEVICE
    $ sudo resize2fs /dev/DEVICE
    $ sudo eject /dev/DEVICE

...where DEVICE is replaced with the device name of your MicroSD card.  (You
can generally find this out by running "sudo dmesg | tail" just after inserting
the MicroSD card into the reader. Use the device file - e.g. /dev/sda
- not partition files like /dev/sda1.)

!!! BE CAREFUL to get the right device - don't clobber your hard drive by accident!
