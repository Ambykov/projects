#!/usr/bin/env python3
"""
Тест долгой WebSocket сессии через шлюз.

Архитектура: "одна реплика — одно upstream-соединение"
- Одно клиентское соединение к шлюзу на всю сессию
- Для каждой реплики:
  - Отправить аудио (бинарные фреймы)
  - Отправить {"type": "end_of_utterance"}
  - Дождаться {"type": "final"}
  - Шлюз закрывает upstream, клиентское соединение остаётся открытым

Протокол:
  Клиент -> Шлюз: бинарные фреймы (PCM16 аудио)
  Клиент -> Шлюз: {"type": "end_of_utterance"} — конец реплики
  Клиент -> Шлюз: {"type": "stop"} — конец сессии
  Шлюз -> Клиент: {"type": "ready"}
  Шлюз -> Клиент: {"type": "partial", "text": "..."}
  Шлюз -> Клиент: {"type": "final", "text": "..."}
"""

import asyncio
import json
import time
from pathlib import Path
import websockets
import wave


async def send_utterance(ws: websockets.WebSocketClientProtocol, audio_path: Path, utterance_num: int, start_time: float) -> dict:
    """Отправляет аудио реплики и читает ответы до final."""
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        audio_data = wav.readframes(wav.getnframes())

    if sample_rate != 16000 or n_channels != 1 or sampwidth != 2:
        return {"error": f"Expected 16kHz mono PCM16"}

    chunk_size = 1600  # 100 мс @ 16kHz
    chunk_bytes = chunk_size * 2

    first_partial_time: float | None = None
    final_text: str = ""
    final_time: float | None = None
    partial_count = 0

    print(f"  Отправляю реплику {utterance_num}...")

    # Отправляем все чанки аудио
    offset = 0
    while offset < len(audio_data):
        chunk = audio_data[offset:offset + chunk_bytes]
        await ws.send(chunk)
        offset += len(chunk)

    # Отправляем end_of_utterance
    await ws.send(json.dumps({"type": "end_of_utterance"}))

    # Читаем ответы до final
    print(f"  Читаю ответы...")
    end_time = time.perf_counter() + 8
    while time.perf_counter() < end_time:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            data = json.loads(msg)

            if data.get("type") == "partial":
                if first_partial_time is None:
                    first_partial_time = time.perf_counter()
                partial_count += 1
            elif data.get("type") == "final":
                final_text = data.get("text", "")
                final_time = time.perf_counter()
                break
        except asyncio.TimeoutError:
            break

    ttfp_ms = (first_partial_time - start_time) * 1000 if first_partial_time else None
    ttfp_str = f"{ttfp_ms:.0f}мс" if ttfp_ms else "N/A"
    final_str = final_text[:20] if final_text else "(нет)"
    print(f"  Реплика {utterance_num}: TTFP={ttfp_str}, partials={partial_count}, final='{final_str}'")

    return {
        "path": str(audio_path.name),
        "ttfp_ms": ttfp_ms,
        "partial_count": partial_count,
        "final_text": final_text,
    }


async def main():
    import sys
    import os

    is_gateway = "--gateway" in sys.argv
    is_direct = "--direct" in sys.argv
    num_files = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--files" and i + 1 < len(sys.argv):
            num_files = int(sys.argv[i + 1])

    api_key = os.getenv("API_KEY", "change-me-in-production")

    if is_gateway:
        ws_url = f"ws://127.0.0.1:8100/ws/stt?token={api_key}"
        print(f"=== Тест долгой сессии через шлюз ===")
    elif is_direct:
        ws_url = "ws://127.0.0.1:9876/v1/ws"
        print(f"=== Тест долгой сессии напрямую (GigaSTT) ===")
    else:
        ws_url = "ws://127.0.0.1:9876/v1/ws"
        print(f"=== Тест долгой сессии ===")

    print(f"URL: {ws_url}")
    print(f"Реплик: {num_files}")
    print()

    all_files = [
        Path("audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"),
        Path("audio-samples/commands/msc/go/2023-08-25-15-13-05.wav"),
        Path("audio-samples/commands/msc/yes/2023-06-26-13-12-21.wav"),
        Path("audio-samples/commands/msc/no/2023-08-26-17-01-38.wav"),
        Path("audio-samples/commands/msc/up/2023-08-26-17-02-04.wav"),
    ]
    files = all_files[:num_files]

    print("Открываем одно WebSocket соединение...")
    connect_start = time.perf_counter()

    try:
        async with websockets.connect(ws_url) as ws:
            connect_time = time.perf_counter() - connect_start
            print(f"Подключено за {connect_time*1000:.0f}мс")

            # Получаем ready
            ready = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"Получено: {ready[:80]}...")
            print()

            start_time = time.perf_counter()

            # Отправляем реплики подряд
            all_results = []
            for i, audio_path in enumerate(files):
                result = await send_utterance(ws, audio_path, i + 1, start_time)
                all_results.append(result)

                # Ждём 0.5 сек перед следующей репликой
                if i < len(files) - 1:
                    await asyncio.sleep(0.5)

            print()
            print("Отправляем stop после всех реплик...")
            await ws.send(json.dumps({"type": "stop"}))

            # Читаем завершение
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print(f"  Получено после stop: {msg}")
            except asyncio.TimeoutError:
                pass

            print()
            print("=== Результаты ===")
            for r in all_results:
                print(f"  {r['path']}: TTFP={r['ttfp_ms']:.0f}мс, partials={r['partial_count']}, final='{r['final_text'][:20]}'")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"CONNECTION CLOSED: {e}")
        print("Соединение закрылось ДО отправки stop!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
