#!/usr/bin/env bash
# One-time: partition + format the Kingston NV3 as a model store at /mnt/nvme.
# Run as root:  sudo bash setup-nvme.sh --yes
set -euo pipefail

# The disk this was written for. Set all three for another machine. The serial
# is checked against what the kernel reports. That check stops the script from
# erasing the wrong disk.
BYID=${BYID:-/dev/disk/by-id/nvme-KINGSTON_SNV3S500G_50026B7785D45B79}
WANT_SERIAL=${WANT_SERIAL:-50026B7785D45B79}
WANT_DEV=${WANT_DEV:-/dev/nvme0n1}
MNT=${MNT:-/mnt/nvme}
OWNER=${OWNER:-${SUDO_USER:-root}:${SUDO_USER:-root}}
PATH=/sbin:/usr/sbin:$PATH

[[ ${1:-} == --yes ]] || { echo "refusing without --yes"; exit 1; }
[[ $EUID -eq 0 ]]     || { echo "must run as root"; exit 1; }

# readlink -e, not -f: -f succeeds on a dangling link and prints the
# unresolved path.
DEV=$(readlink -e "$BYID") || { echo "by-id path missing: $BYID"; exit 1; }
echo "target: $BYID -> $DEV"

# --- refuse unless this is exactly the disk we mean, and it is idle ---
[[ $DEV == "$WANT_DEV" ]] || { echo "unexpected device $DEV (wanted $WANT_DEV; set WANT_DEV)"; exit 1; }
SERIAL=$(tr -d ' ' < "/sys/block/${DEV##*/}/device/serial")
[[ $SERIAL == "$WANT_SERIAL" ]] || { echo "serial mismatch: $SERIAL"; exit 1; }

if grep -q "^$DEV" /proc/mounts; then
  echo "REFUSING: $DEV is mounted"; grep "^$DEV" /proc/mounts; exit 1
fi
if grep -q "^$DEV" /proc/swaps; then
  echo "REFUSING: $DEV is in use as swap"; exit 1
fi
if [[ -n $(ls -A "/sys/block/${DEV##*/}/holders/" 2>/dev/null) ]]; then
  echo "REFUSING: $DEV has holders (lvm/md/dm/luks)"; exit 1
fi
# one line == the disk itself and no partition of any number
if [[ $(lsblk -n -o NAME "$DEV" | wc -l) -ne 1 ]]; then
  echo "REFUSING: $DEV already has partitions -- inspect them first"
  lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT "$DEV"; exit 1
fi
echo "checks passed: not mounted, not swap, no holders, no partitions"
echo

lsblk -o NAME,SIZE,TYPE,FSTYPE,PTTYPE,MOUNTPOINT "$DEV"
echo
read -r -p "ERASE $DEV (serial $SERIAL) and format ext4? type ERASE: " ans
[[ $ans == ERASE ]] || { echo "aborted"; exit 1; }

# The partition node, decided before anything is destroyed. nvme and mmc name
# it ${DEV}p1, a sata disk /dev/sda1.
case $DEV in
  *[0-9]) PART=${DEV}p1 ;;
  *)      PART=${DEV}1  ;;
esac
echo "partition will be $PART"

wipefs -a "$DEV"
parted -s "$DEV" mklabel gpt
parted -s -a optimal "$DEV" mkpart models ext4 1MiB 100%
partprobe "$DEV"; udevadm settle; sleep 2
[[ -b $PART ]] || { echo "partition $PART did not appear"; exit 1; }

# -T largefile: a few ~50 GB shards, so very few inodes needed.
# lazy_*_init=0: initialise now, not in the background during benchmarks.
mkfs.ext4 -F -m 0 -L qwen-models -T largefile \
  -E lazy_itable_init=0,lazy_journal_init=0 "$PART"

UUID=$(blkid -s UUID -o value "$PART")
mkdir -p "$MNT"
# nofail: a secondary data disk must never drop the box into emergency mode
grep -q "$UUID" /etc/fstab || \
  echo "UUID=$UUID  $MNT  ext4  defaults,noatime,nofail  0  2" >> /etc/fstab
systemctl daemon-reload
mount "$MNT"
chown "$OWNER" "$MNT"

echo; echo "done:"; df -hT "$MNT"; echo "fstab: $(grep "$UUID" /etc/fstab)"
