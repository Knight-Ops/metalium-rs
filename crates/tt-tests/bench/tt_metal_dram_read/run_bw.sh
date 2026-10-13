#!/bin/bash
# usage: run_bw.sh <log> [dram_read_bw args...]   -- tt-metal v077, PCIe card 1 only, fast dispatch
LOG=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
cd /home/carl/projects/tt-metal-v077
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) TT_VISIBLE_DEVICES=1 READER_KERNEL=$HERE/reader.cpp
timeout ${BW_TIMEOUT:-1500} $HERE/dram_read_bw "$@" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
