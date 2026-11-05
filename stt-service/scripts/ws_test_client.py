#!/usr/bin/env python3
"""
WebSocket клиент для проверки стриминга GigaSTT.

Подключается к ws://127.0.0.1:9876/v1/ws (или через шлюз).
Отправляет WAV чанками по 100 мс (1600 сэмплов @ 16 кГц).
Замеряет time-to-first-partial и time-to-final.
Сохраняет результаты в results/docker-ws-check/ или results/docker-ws-via-gateway/.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import websockets


async def transcribe_file(
    audio_path: Path,
    ws_url: str,
) -> dict:
    """Транскрибирует один файл, открывая новое WS соединение. TTFP от начала подключения."""
    import wave

    connect_start = time.perf_counter()

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))
        ready = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ready_data = json.loads(ready)
        if ready_data.get("type") != "ready":
            return {"error": f"Expected ready, got {ready_data}"}

        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            audio_data = wav.readframes(wav.getnframes())

        if sample_rate != 16000 or n_channels != 1 or sampwidth != 2:
            return {"error": f"Expected 16kHz mono PCM16, got {sample_rate}Hz {n_channels}ch {sampwidth*8}bit"}

        audio_duration = len(audio_data) / (16000 * 2)
        chunk_size = 1600
        chunk_bytes = chunk_size * 2

        first_partial_time: Optional[float] = None
        final_text: str = ""
        final_time: Optional[float] = None
        partial_count = 0

        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_bytes]
            await ws.send(chunk)
            offset += len(chunk)
            await asyncio.sleep(0.001)

        await ws.send(json.dumps({"type": "end_of_utterance"}))

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)

                if data.get("type") == "partial":
                    if first_partial_time is None:
                        first_partial_time = time.perf_counter() - connect_start
                    partial_count += 1
                elif data.get("type") == "final":
                    final_text = data.get("text", "")
                    final_time = time.perf_counter() - connect_start
                    break
            except asyncio.TimeoutError:
                break

        total_time = time.perf_counter() - connect_start

        return {
            "path": str(audio_path),
            "audio_duration_sec": audio_duration,
            "time_to_first_partial_sec": first_partial_time,
            "time_to_final_sec": final_time,
            "total_time_sec": total_time,
            "final_text": final_text,
            "partial_count": partial_count,
        }


async def run_run(audio_files: list[tuple[Path, str]], ws_url: str) -> list[dict]:
    """Прогоняет все файлы через отдельные WS соединения."""
    results = []

    for audio_path, reference in audio_files:
        result = await transcribe_file(audio_path, ws_url)
        if "error" not in result:
            result["reference"] = reference
        results.append(result)

    return results


async def run_run(audio_files: list[tuple[Path, str]], ws_url: str) -> list[dict]:
    """Прогоняет все файлы через отдельные WS соединения."""
    results = []

    for audio_path, reference in audio_files:
        result = await transcribe_file(audio_path, ws_url)
        if "error" not in result:
            result["reference"] = reference
        results.append(result)

    return results


def load_manifest_subset(manifest_path: Path, n_files: int = 10) -> list[tuple[Path, str]]:
    """Загружает подмножество manifest для WebSocket теста."""
    samples = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rel_path, reference_text = parts
            audio_path = manifest_path.parent / rel_path

            if audio_path.exists() and len(samples) < n_files:
                samples.append((audio_path, reference_text.strip()))

    return samples


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Вычисляет WER между reference и hypothesis."""
    import jiwer
    import re

    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[.,!?;:]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Guard against empty strings
    if not ref_norm or not hyp_norm:
        return 1.0

    return jiwer.wer(ref_norm, hyp_norm)


async def main():
    import sys
    import os
    is_gateway = "--gateway" in sys.argv
    api_key = os.getenv("API_KEY", "change-me-in-production")
    ws_url = f"ws://127.0.0.1:8100/ws/stt?token={api_key}" if is_gateway else "ws://127.0.0.1:9876/v1/ws"

    mode = "через шлюз" if is_gateway else "(напрямую)"
    print(f"=== WebSocket Client для GigaSTT {mode} ===")
    print(f"URL: {ws_url}")
    print(f"Методика: отдельное WS соединение на файл, 10 файлов, прогрев 1, замер 5")
    print()

    manifest_path = Path("audio-samples/manifest.tsv")
    samples = load_manifest_subset(manifest_path, n_files=10)
    print(f"Файлы: {len(samples)}")
    print()

    # Прогрев: 1 прогон
    print("=== Прогрев (не учитывается в статистике) ===")
    warmup_results = await run_run(samples, ws_url)
    for r in warmup_results:
        if "error" not in r and r.get("time_to_first_partial_sec"):
            ttfp = r["time_to_first_partial_sec"] * 1000
            print(f"  {Path(r['path']).name}: TTFP={ttfp:.0f}мс")
    print()

    # Замер: 5 прогонов
    print("=== Замер (5 прогонов) ===")
    all_ttfp = []
    all_ttft = []

    for run in range(1, 6):
        print(f"\nПрогон {run}/5:")
        run_results = await run_run(samples, ws_url)

        run_ttfp = []
        for r in run_results:
            if "error" not in r and r.get("time_to_first_partial_sec"):
                ttfp = r["time_to_first_partial_sec"] * 1000
                run_ttfp.append(ttfp)
                print(f"  {Path(r['path']).name}: TTFP={ttfp:.0f}мс")

        if run_ttfp:
            run_median = sorted(run_ttfp)[len(run_ttfp) // 2]
            print(f"  → median: {run_median:.0f}мс")
            all_ttfp.append(run_median)
            all_ttft.append(sum(run_ttfp) / len(run_ttfp))

    print()
    print("=== Итоговая статистика ===")
    if all_ttfp:
        ttfp_median = sorted(all_ttfp)[len(all_ttfp) // 2]
        ttfp_std = (sum((x - ttfp_median) ** 2 for x in all_ttfp) / len(all_ttfp)) ** 0.5
        print(f"TTFP median: {ttfp_median:.0f}мс, std: {ttfp_std:.1f}мс")
        print(f"Все 5 значений TTFP: {all_ttfp}")


if __name__ == "__main__":
    asyncio.run(main())
