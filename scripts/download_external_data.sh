#!/usr/bin/env bash
# Download PUBLIC datasets only (never the challenge data).
set -euo pipefail
DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "[1/2] MUSAN (music/noise/speech) ..."
# https://www.openslr.org/17/
wget -c https://www.openslr.org/resources/17/musan.tar.gz -P "$DATA_DIR"
tar -xzf "$DATA_DIR/musan.tar.gz" -C "$DATA_DIR" && mv "$DATA_DIR/musan" "$DATA_DIR/musan" || true

echo "[2/2] RIRS_NOISES (room impulse responses) ..."
# https://www.openslr.org/28/
wget -c https://www.openslr.org/resources/28/rirs_noises.zip -P "$DATA_DIR"
unzip -o "$DATA_DIR/rirs_noises.zip" -d "$DATA_DIR" && mv "$DATA_DIR/RIRS_NOISES" "$DATA_DIR/rirs" || true

echo "Done. (LibriSpeech optional: https://www.openslr.org/12/ )"
