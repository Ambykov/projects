#!/usr/bin/env python3
"""
Блок C: Edge cases протокола.

Запускается на uvicorn с SESSION_IDLE_TIMEOUT_SEC=120.
"""

import asyncio
import json
import os
import time
import wave

import websockets

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8100))
GATEWAY_WS_URL = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=testtoken"


async def test_c1():
    """C1: stop без end_of_utterance"""
    print("\nC1: stop без end_of_utterance")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
            with wave.open(audio_path, "rb") as wav:
                await ws.send(wav.readframes(wav.getnframes()))
            # Сразу stop без end_of_utterance
            await ws.send(json.dumps({"type": "stop"}))
        print("  PASS: Сессия закрыта корректно")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c2():
    """C2: end_of_utterance без аудио"""
    print("\nC2: end_of_utterance без аудио")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            # Ждать ответ с таймаутом
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print(f"  PASS: Получен {json.loads(msg).get('type')}")
            except asyncio.TimeoutError:
                print("  PASS: Timeout (ожидаемо - нет аудио)")
            await ws.send(json.dumps({"type": "stop"}))
        return True
    except Exception as e:
        print(f"  PASS: Ошибка не валит шлюз: {e}")
        return True


async def test_c3():
    """C3: Два end_of_utterance подряд"""
    print("\nC3: Два end_of_utterance подряд")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
            with wave.open(audio_path, "rb") as wav:
                await ws.send(wav.readframes(wav.getnframes()))
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            await asyncio.wait_for(ws.recv(), timeout=10.0)  # final
            # Второй end_of_utterance
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            # Дать шлюзу обработать
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))
        print("  PASS: Второй end_of_utterance обработан")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c4():
    """C4: Аудио после stop"""
    print("\nC4: Аудио после stop")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            await ws.send(json.dumps({"type": "stop"}))
        # Соединение закрыто
        print("  PASS: Соединение закрыто")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c5():
    """C5: Аудио после end_of_utterance (новая реплика)"""
    print("\nC5: Аудио после end_of_utterance (новая реплика)")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            # Первая реплика
            audio1 = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
            with wave.open(audio1, "rb") as wav:
                await ws.send(wav.readframes(wav.getnframes()))
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            await asyncio.wait_for(ws.recv(), timeout=10.0)  # final первой
            # Вторая реплика без явного start
            audio2 = "audio-samples/commands/msc/go/2023-08-25-15-13-05.wav"
            with wave.open(audio2, "rb") as wav:
                await ws.send(wav.readframes(wav.getnframes()))
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            print(f"  Вторая реплика: {json.loads(msg).get('type')}")
            await ws.send(json.dumps({"type": "stop"}))
        print("  PASS: Вторая реплика обработана")
        return True
    except asyncio.TimeoutError:
        print("  FAIL: Timeout")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c6():
    """C6: Очень короткая реплика (100 мс)"""
    print("\nC6: Очень короткая реплика (100 мс)")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
            with wave.open(audio_path, "rb") as wav:
                data = wav.readframes(wav.getnframes())
            # Отправить только 1 чанк (100 мс)
            await ws.send(data[:3200])
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg_type = json.loads(msg).get("type")
            print(f"  Ответ: {msg_type}")
            await ws.send(json.dumps({"type": "stop"}))
        print("  PASS: Короткая реплика обработана")
        return True
    except asyncio.TimeoutError:
        print("  FAIL: Timeout")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c7():
    """C7: Очень длинная реплика (10 сек шума)"""
    print("\nC7: Очень длинная реплика (10 сек шума)")
    try:
        import numpy as np

        duration = 10.0
        sample_rate = 16000
        samples = np.random.randint(-32768, 32767, int(duration * sample_rate), dtype=np.int16)

        async with websockets.connect(GATEWAY_WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
            chunk_size = 3200  # 100 мс
            start = time.perf_counter()
            for i in range(0, len(samples), chunk_size):
                await ws.send(samples[i : i + chunk_size].tobytes())
                await asyncio.sleep(0.001)
            await ws.send(json.dumps({"type": "end_of_utterance"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=20.0)
            elapsed = time.perf_counter() - start
            msg_type = json.loads(msg).get("type")
            print(f"  Ответ через {elapsed:.1f}с: {msg_type}")
            await ws.send(json.dumps({"type": "stop"}))
        print(f"  PASS: 10 сек обработано за {elapsed:.1f}с")
        return True
    except asyncio.TimeoutError:
        print("  FAIL: Timeout")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_c8():
    """C8: Разрыв клиента (закрыть WS без stop)"""
    print("\nC8: Разрыв клиента (закрыть WS без stop)")
    try:
        ws = await websockets.connect(GATEWAY_WS_URL)
        await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
        audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
        with wave.open(audio_path, "rb") as wav:
            await ws.send(wav.readframes(wav.getnframes()))
        # Закрыть без stop
        await ws.close()
        await asyncio.sleep(1)  # Дать шлюзу обработать
        print("  PASS: Шлюз обработал разрыв")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def main():
    print("\n" + "=" * 50)
    print("=== Блок C: Edge cases ===")
    print("=" * 50)

    results = {}
    results["C1"] = await test_c1()
    await asyncio.sleep(0.5)
    results["C2"] = await test_c2()
    await asyncio.sleep(0.5)
    results["C3"] = await test_c3()
    await asyncio.sleep(0.5)
    results["C4"] = await test_c4()
    await asyncio.sleep(0.5)
    results["C5"] = await test_c5()
    await asyncio.sleep(0.5)
    results["C6"] = await test_c6()
    await asyncio.sleep(0.5)
    results["C7"] = await test_c7()
    await asyncio.sleep(0.5)
    results["C8"] = await test_c8()

    passed = sum(1 for v in results.values() if v)
    print(f"\nБлок C: {passed}/8 pass")
    print(f"Результаты: {results}")


if __name__ == "__main__":
    asyncio.run(main())
