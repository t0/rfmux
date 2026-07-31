#!/usr/bin/env bash

cd ~/rfmux/firmware/r1.7.0rc5
source /opt/xilinx/2025.2/Vitis/settings64.sh

read -p "Remove the MicroSD card from the CRS and turn it on. Press Enter to continue."

sudo /opt/xilinx/2025.2/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers > /dev/null

read -p "Connect this PC to the CRS via micro USB cable. Press Enter to continue."

if true || lsusb -d 0403:6011 > /dev/null; then 
	echo "USB device found."
else
	echo "WARNING: USB device not found. Check that the cable is plugged in and try running this script again." >&2
	exit
fi

program_flash -f boot.bin -fsbl fsbl.elf -flash_type qspi-x8-dual_parallel

echo "If you see \"Flash Operation Successful\", power off and power on the CRS.\
The screen on the front panel should turn on after a few seconds with the\
t0.technology logo. You may now proceed"


