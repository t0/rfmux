#!/usr/bin/env bash

cd ~/rfmux/firmware/r1.7.0
source /opt/xilinx/2025.2/Vitis/settings64.sh

read -p "Remove the MicroSD card from the CRS and turn it on. Press Enter to continue."

sudo /opt/xilinx/2025.2/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers > /dev/null

read -p "Connect this PC to the CRS via micro USB cable. Press Enter to continue."

if lsusb -d 0403:6011 > /dev/null; then 
	echo "USB device found."
else
	echo "WARNING: USB device not found. Check that the cable is plugged in and try running this script again." >&2
	exit 1
fi

if ! program_flash -f boot.bin -fsbl fsbl.elf -flash_type qspi-x8-dual_parallel; then
	echo "ERROR: flashing failed." >&2
	exit 1
fi

printf "If you see \"Flash Operation Successful\", power off and power on the CRS. \
The screen on the front panel should turn on after a few seconds and display the \
t0.technology logo momentarily. You have successfully upgraded the bootloader \
and may now proceed to upgrade the SD card image.\n\n"

read -p "If your CRS does not have a display, turn it off and turn it back on. \
Then start the serial port scan by pressing Enter:"

{
	stty 115200 raw -echo
	sleep 15
	printf "\x03" >&0
	sleep 0.2
	printf "env print bootloader_version\r" >&0
	timeout 1 cat
} <>/dev/ttyUSB1

printf "\n\nThe CRS has been probed for its bootloader version. A git hash \
should have been printed. If the hash matches\n\
\"bootloader_version=d951c3321e49f093cbbf90b40a742d531dd2171f\"\n\
then the upgrade was successful.\n"
