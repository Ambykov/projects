#!/bin/bash
# convert-audio.sh — конвертация аудио в формат для тестов (WAV, PCM16, mono, 16 kHz)
#
# Использование:
#   ./convert-audio.sh <input_file> [output_dir]
#
# Вывод: <output_dir>/<basename>.wav
# По умолчанию output_dir = текущая директория

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_dir]"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-.}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: File not found: $INPUT_FILE"
    exit 1
fi

BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.wav"

echo "Converting: $INPUT_FILE -> $OUTPUT_FILE"
ffmpeg -y -i "$INPUT_FILE" -ar 16000 -ac 1 -acodec pcm_s16le "$OUTPUT_FILE"

echo "Done. Output: $OUTPUT_FILE"
