#!/usr/bin/env bash
# End-to-end milestone-1 pipeline. Assumes `pip install -e .` and that audio
# lives under data/raw (see README for the MTG-Jamendo download steps).
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-1500}"   # subset size
K="${2:-10}"     # neighbors / eval cutoff

audio-sim download-metadata --name autotagging
audio-sim subset --n "$N" --category genre
audio-sim embed
audio-sim evaluate --k "$K"

echo
echo "Done. Try: audio-sim query --key <TRACK_ID>  or  audio-sim query --text 'mellow jazz guitar'"
