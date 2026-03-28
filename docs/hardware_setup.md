# Hardware Setup Guide

## Overview

The Quantum Quad-Brain compute array is built from commodity, off-the-shelf hardware:

| Component | Recommended Model | Qty |
|-----------|------------------|-----|
| Compute Module | Raspberry Pi CM4 (8 GB RAM, eMMC) | 4+ |
| NVMe SSD | Samsung 990 Pro 2 TB (PCIe Gen 4) | 2+ |
| Carrier Board | CM4IO or custom CM4-NVMe carrier | 4+ |
| DRAM | System RAM on compute host (64 GB+) | 1 |
| Network Switch | 1 GbE managed switch | 1 |
| Power Supply | 5 V / 5 A (USB-C PD) per CM4 node | 4+ |

---

## Physical Assembly

### 1. CM4 Carrier Board Preparation

1. Mount each CM4 module onto its carrier board, pressing firmly until the
   board-to-board connector fully seats.
2. Connect the MicroSD or eMMC boot image (see Software Integration guide).
3. Attach a USB-C power cable rated ≥ 5 V / 5 A to each carrier board.

### 2. NVMe SSD Installation

1. Insert the NVMe SSD(s) into the M.2 Key-M slot on the carrier board or a
   dedicated PCIe adapter connected to the cluster's host node.
2. Secure the drive with the M.2 retaining screw.
3. Verify the device appears as `/dev/nvme0n1` (and `/dev/nvme1n1` etc.) after
   boot using:

   ```bash
   lsblk -d -o NAME,MODEL,SIZE,TRAN | grep nvme
   ```

4. Confirm sequential read bandwidth:

   ```bash
   fio --name=seq_read --filename=/dev/nvme0n1 --rw=read \
       --bs=128k --direct=1 --size=4G --numjobs=1 --runtime=30s \
       --group_reporting
   ```

   Expect ≥ 6 GB/s for PCIe Gen 4 NVMe devices.

### 3. Network Setup

1. Connect all CM4 nodes and the host machine to the same managed Ethernet switch.
2. Assign static IP addresses from the `10.0.0.100–10.0.0.131` range (configurable
   in `array_config.yaml`).
3. Verify connectivity:

   ```bash
   for i in $(seq 0 3); do ping -c1 10.0.0.$((100+i)); done
   ```

### 4. Power Sequencing

Power on NVMe drives before compute nodes to avoid device enumeration races.

---

## System Requirements (Host / Controller Node)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| Python | 3.9 | 3.12 |
| RAM | 16 GB | 64 GB |
| Storage (OS) | 32 GB | 128 GB SSD |

---

## Flashing CM4 eMMC

```bash
# On the host, install rpiboot
sudo apt install rpiboot

# Set the CM4 to USB boot mode (bridge JP1/nRPIBOOT on carrier board)
sudo rpiboot

# Flash Raspberry Pi OS Lite (64-bit)
rpi-imager  # or: sudo dd if=raspios-lite.img of=/dev/sdX bs=4M status=progress
```

---

## BIOS / Firmware Considerations

- Enable PCIe Gen 4 in the carrier board firmware for full NVMe bandwidth.
- Set `BOOT_ORDER=0xf416` in `config.txt` for NVMe-first boot on CM4IO boards.
