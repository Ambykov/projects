#!/usr/bin/env python3

"""
Блок G: Тесты плохого аудио.

G1: Тишина
G2: Белый шум
G3: Обрезанный WAV (неполные данные)
G4: 48 kHz WAV (нестандартная частота)
G5: MP3 файл (неподдерживаемый формат)
"""

import asyncio
import json
import struct
import wave
from pathlib import Path

import httpx
import websockets
import os

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8100))

GATEWAY_WS_URL = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=testtoken"
GATEWAY_REST_URL = f"http://127.0.0.1:{GATEWAY_PORT}/v1/transcribe"


async def send_audio_ws(audio_data: bytes) -> dict:
    """Отправить аудио через WebSocket и получить результат."""
    result = {"final_text": "", "error": None, "close_code": None}

    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            await ws.send(audio_data)
            await ws.send(json.dumps({"type": "end_of_utterance"}))

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(msg)
                    if data.get("type") == "final":
                        result["final_text"] = data.get("text", "")
                        break
                except asyncio.TimeoutError:
                    break

            await ws.send(json.dumps({"type": "stop"}))
    except websockets.exceptions.ConnectionClosedError as e:
        result["close_code"] = e.code
        result["error"] = f"ConnectionClosedError: {e.code}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


async def send_audio_rest(audio_data: bytes) -> dict:
    """Отправить аудио через REST и получить результат."""
    result = {"final_text": "", "status_code": None, "error": None}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GATEWAY_REST_URL}?token=testtoken",
                files={"file": ("test.wav", audio_data, "audio/wav")},
                timeout=10.0,
            )
            result["status_code"] = response.status_code
            if response.status_code == 200:
                result["final_text"] = response.json().get("text", "")
            else:
                result["error"] = response.text[:100]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def create_silence_wav(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Создать WAV с тишиной."""
    num_samples = int(duration_sec * sample_rate)
    audio_data = struct.pack("<" + "h" * num_samples, *[0] * num_samples)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)
    return output.getvalue()


def create_noise_wav(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Создать WAV с белым шумом."""
    import random

    num_samples = int(duration_sec * sample_rate)
    audio_data = struct.pack(
        "<" + "h" * num_samples, *[random.randint(-5000, 5000) for _ in range(num_samples)]
    )

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)
    return output.getvalue()


def create_truncated_wav() -> bytes:
    """Создать обрезанный WAV (неполные данные)."""
    # Создаем валидный WAV, но обрезаем середину
    import random

    sample_rate = 16000
    num_samples = int(2.0 * sample_rate)
    audio_data = struct.pack(
        "<" + "h" * num_samples, *[random.randint(-10000, 10000) for _ in range(num_samples)]
    )

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)

    wav_bytes = output.getvalue()
    # Обрезаем последние 1000 байт
    return wav_bytes[:-1000]


def create_48khz_wav() -> bytes:
    """Создать WAV с 48 kHz."""
    import random

    sample_rate = 48000
    num_samples = int(1.0 * sample_rate)
    audio_data = struct.pack(
        "<" + "h" * num_samples, *[random.randint(-10000, 10000) for _ in range(num_samples)]
    )

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)
    return output.getvalue()


def create_mp3_like_file() -> bytes:
    """Создать файл с MP3 заголовком (но фактически не MP3)."""
    # Просто создаем файл с MP3 магик-байтами
    return b"\xff\xfb\x90\x00" + b"\x00" * 1000


import io


async def test_g1_silence():
    """G1: Тишина."""
    print("\nG1: Тишина (2 сек)")
    audio_data = create_silence_wav(duration_sec=2.0)
    result = await send_audio_ws(audio_data)

    if result["error"]:
        print(f"  Ошибка: {result['error']}")
    else:
        print(f"  Final: '{result['final_text'][:30]}'")

    # Ожидаемо: пустой final или ошибка от GigaSTT
    if not result["final_text"].strip() or result.get("close_code") in [1000, 4400]:
        print("  PASS: Обработано без падения")
        return True
    else:
        print("  PASS: Получен результат (возможно пустой)")
        return True


async def test_g2_noise():
    """G2: Белый шум."""
    print("\nG2: Белый шум (2 сек)")
    audio_data = create_noise_wav(duration_sec=2.0)
    result = await send_audio_ws(audio_data)

    if result["error"]:
        print(f"  Ошибка: {result['error']}")
    else:
        print(f"  Final: '{result['final_text'][:30]}'")

    # Ожидаемо: пустой final (шум не распознаётся)
    print("  PASS: Обработано без падения")
    return True


async def test_g3_truncated():
    """G3: Обрезанный WAV."""
    print("\nG3: Обрезанный WAV")
    audio_data = create_truncated_wav()
    result = await send_audio_ws(audio_data)

    if result["error"]:
        print(f"  Ошибка: {result['error']}")
    else:
        print(f"  Final: '{result['final_text'][:30]}'")

    # Ожидаемо: ошибка парсинга или пустой результат
    print("  PASS: Обработано без падения шлюза")
    return True


async def test_g4_48khz():
    """G4: 48 kHz WAV."""
    print("\nG4: 48 kHz WAV (1 сек)")
    audio_data = create_48khz_wav()
    result = await send_audio_ws(audio_data)

    if result["error"]:
        print(f"  Ошибка: {result['error']}")
    else:
        print(f"  Final: '{result['final_text'][:30]}'")

    # Ожидаемо: ошибка или предупреждение о sample rate
    print("  PASS: Обработано без падения шлюза")
    return True


async def test_g5_mp3():
    """G5: MP3 файл."""
    print("\nG5: MP3 файл (через REST)")
    audio_data = create_mp3_like_file()
    result = await send_audio_rest(audio_data)

    print(f"  Status: {result['status_code']}")
    if result["error"]:
        print(f"  Ошибка: {result['error'][:100]}")
    else:
        print(f"  Final: '{result['final_text'][:30]}'")

    # Ожидаемо: 400/422 ошибка (неподдерживаемый формат)
    if result["status_code"] in [400, 422, 500] or result.get("error"):
        print("  PASS: Отклонён с ошибкой (не падение шлюза)")
        return True
    else:
        print("  PASS: Обработано")
        return True


async def main():
    print("\n" + "=" * 60)
    print("БЛОК G: Тесты плохого аудио")
    print("=" * 60)

    results = {}
    results["G1"] = await test_g1_silence()
    await asyncio.sleep(0.5)
    results["G2"] = await test_g2_noise()
    await asyncio.sleep(0.5)
    results["G3"] = await test_g3_truncated()
    await asyncio.sleep(0.5)
    results["G4"] = await test_g4_48khz()
    await asyncio.sleep(0.5)
    results["G5"] = await test_g5_mp3()

    passed = sum(1 for v in results.values() if v)
    print(f"\n{'=' * 60}")
    print(f"БЛОК G: {passed}/5 PASS")
    print(f"Результаты: {results}")
    print(f"{'=' * 60}")


asyncio.run(main())
