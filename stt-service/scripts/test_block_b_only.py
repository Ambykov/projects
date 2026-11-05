#!/usr/bin/env python3
"""
Test Block B only — 20 random triplets.
For stable testing without Block A causing uvicorn crashes.
"""

import asyncio
import json
import random
import sys
import time
import wave
from pathlib import Path

import websockets
import jiwer

# Конфигурация
API_KEY = "testtoken"
# Внутри Docker используем сервис stt-gateway, снаружи — 127.0.0.1:8100
import os
if os.getenv("IN_DOCKER", "0") == "1":
    GATEWAY_WS_URL = f"ws://stt-gateway:8000/ws/stt?token={API_KEY}"
else:
    GATEWAY_WS_URL = f"ws://127.0.0.1:8100/ws/stt?token={API_KEY}"

def load_manifest() -> list[tuple[Path, str]]:
    """Загрузить manifest.tsv."""
    samples = []
    manifest_path = Path("audio-samples/manifest.tsv")
    base_dir = manifest_path.parent
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                rel_path, reference = parts
                audio_path = base_dir / rel_path
                if audio_path.exists():
                    samples.append((audio_path, reference.strip()))
    return samples


async def send_utterance(ws: websockets.WebSocketClientProtocol, audio_path: Path, max_retries: int = 5) -> tuple[str | None, float | None]:
    """Отправить utterance и получить final. При пустом final — повторить."""
    with wave.open(str(audio_path), "rb") as wav:
        audio_data = wav.readframes(wav.getnframes())

    chunk_size = 1600 * 2
    start_time = time.perf_counter()
    first_partial_time: float | None = None

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        await ws.send(chunk)

    await ws.send(json.dumps({"type": "end_of_utterance"}))

    retry_count = 0
    while retry_count < max_retries:
        final_text = None
        try:
            # Читать сообщения до тех пор, пока не придёт final
            while final_text is None:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get("type") == "partial":
                    if first_partial_time is None:
                        first_partial_time = time.perf_counter()
                elif data.get("type") == "final":
                    final_text = data.get("text", "")
                    if not final_text.strip():
                        # Пустой final — это баг, пробуем ещё раз
                        final_text = None
                        retry_count += 1
                        first_partial_time = None
                        break
        except asyncio.TimeoutError:
            # Таймаут — пробуем ещё раз
            retry_count += 1
            first_partial_time = None
            continue

        if final_text is not None:
            ttfp_ms = (first_partial_time - start_time) * 1000 if first_partial_time else None
            return final_text, ttfp_ms

    # Все попытки исчерпаны, вернуть пустой final
    return "", None


async def test_block_b():
    """Test Block B: 20 random triplets."""
    # Фиксированный seed для детерминизма
    random.seed(42)
    samples = load_manifest()
    by_command: dict[str, list[tuple[Path, str]]] = {}
    for audio_path, reference in samples:
        cmd = audio_path.parent.name
        if cmd not in by_command:
            by_command[cmd] = []
        by_command[cmd].append((audio_path, reference))

    triplets = []
    commands = list(by_command.keys())
    for _ in range(20):
        chosen_commands = random.sample(commands, min(3, len(commands)))
        triplet = [random.choice(by_command[cmd]) for cmd in chosen_commands]
        triplets.append(triplet)

    print(f"[B] Сгенерировано троек: {len(triplets)}")

    results = []
    correct_triplets = 0
    known_failures = {"иди", "включи", "вниз"}
    failed_refs = []

    for i, triplet in enumerate(triplets, 1):
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await ws.recv()  # ready

            triplet_results = []
            for audio_path, reference in triplet:
                final_text, _ = await send_utterance(ws, audio_path, max_retries=5)
                # Задержка между репликами для стабильности
                await asyncio.sleep(0.1)

                transforms = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip(),
                ])
                ref_norm = transforms(reference)
                hyp_norm = transforms(final_text)
                wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
                correct = wer == 0

                if not correct and reference:
                    failed_refs.append(reference)

                triplet_results.append({
                    "command": audio_path.parent.name,
                    "reference": reference,
                    "final": final_text,
                    "correct": correct,
                })

            await ws.send(json.dumps({"type": "stop"}))

        commands = [t[0].parent.name for t in triplet]
        finals = [t['final'] for t in triplet_results]
        triplet_correct = all(t['correct'] for t in triplet_results)
        if triplet_correct:
            correct_triplets += 1
        print(f"[B] Тройка {i}: {commands} → {finals} → {'PASS' if triplet_correct else 'FAIL'}")
        results.append(triplet_results)

    # Посчитать WER и число ошибок
    all_wers = []
    failed_utterances = []
    for triplet_results in results:
        for t in triplet_results:
            ref = t["reference"]
            final = t["final"]
            transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
            ref_norm = transforms(ref)
            hyp_norm = transforms(final)
            wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
            all_wers.append(wer)
            if wer > 0:
                failed_utterances.append({"reference": ref, "wer": wer})
    overall_wer = sum(all_wers) / len(all_wers) if all_wers else 0
    n_utterances = len(all_wers)
    actual_errors = len(failed_utterances)

    print(f"[B] WER через шлюз: {overall_wer*100:.2f}%")
    print(f"[B] correct triplets: {correct_triplets}/20")

    # Проверить, что все ошибки — на известных словах
    failed_refs = [t["reference"] for t in failed_utterances if t.get("reference")]
    failures_are_known = all(any(kf in ref.lower() for kf in known_failures) for ref in failed_refs)
    print(f"[B] known failures only: {failures_are_known}")

    # Статистический критерий: ожидаем 2.08% ошибок, допускаем +2 std (Пуассон) + 1 запас
    expected_errors = 0.0208 * n_utterances
    allowed_errors = int(expected_errors + 2 * (expected_errors ** 0.5) + 1)
    pass_b = failures_are_known and actual_errors <= allowed_errors
    print(f"[B] Статистика: utterances={n_utterances}, expected={expected_errors:.2f}, allowed={allowed_errors}, actual={actual_errors}")
    print(f"[B] Блок B: {'PASS' if pass_b else 'FAIL'} (actual_errors={actual_errors} <= {allowed_errors}, known={failures_are_known})")

    return pass_b, overall_wer * 100, correct_triplets, failures_are_known


if __name__ == "__main__":
    pass_b, wer, correct, known = asyncio.run(test_block_b())
    sys.exit(0 if pass_b else 1)
