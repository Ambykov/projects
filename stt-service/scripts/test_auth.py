"""Block D: Auth tests (D1-D7)."""
import asyncio
import json
import os
import wave

import httpx
import websockets

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8100))
GATEWAY_WS_URL = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=testtoken"
GATEWAY_WS_URL_NO_TOKEN = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt"
GATEWAY_REST_URL = f"http://127.0.0.1:{GATEWAY_PORT}/v1/transcribe"
AUTH_NONE_PORT = 8101  # Second uvicorn with AUTH_MODE=none


async def test_d1():
    """D1: WS без токена — должен получить 4401"""
    print("\nD1: WS без токена")
    try:
        async with websockets.connect(GATEWAY_WS_URL_NO_TOKEN) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"  FAIL: Получен ответ вместо 4401: {msg}")
            return False
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 4401:
            print("  PASS: Код 4401 (Unauthorized)")
            return True
        print(f"  FAIL: Код {e.code} вместо 4401")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d2():
    """D2: WS с неверным токеном — должен получить 4401"""
    print("\nD2: WS с неверным токеном")
    try:
        url = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=wrongtoken"
        async with websockets.connect(url) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"  FAIL: Получен ответ вместо 4401: {msg}")
            return False
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 4401:
            print("  PASS: Код 4401 (Unauthorized)")
            return True
        print(f"  FAIL: Код {e.code} вместо 4401")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d3():
    """D3: WS с токеном в заголовке Authorization — должен получить ready"""
    print("\nD3: WS с токеном в заголовке Authorization")
    try:
        # Подключиться без токена в URL, но с Authorization header
        # websockets 13.x использует extra_headers вместо additional_headers
        async with websockets.connect(
            GATEWAY_WS_URL_NO_TOKEN,
            extra_headers={"Authorization": "Bearer testtoken"}
        ) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            if data.get("type") == "ready":
                print("  PASS: Получен ready с Authorization header")
                await ws.send(json.dumps({"type": "stop"}))
                return True
            print(f"  FAIL: Ожидался ready, получен: {data}")
            return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d4():
    """D4: REST без токена — должен получить 401"""
    print("\nD4: REST без токена")
    try:
        audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
        with wave.open(audio_path, "rb") as wav:
            files = {"file": ("test.wav", wav.readframes(wav.getnframes()), "audio/wav")}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(GATEWAY_REST_URL, files=files)
        if response.status_code == 401:
            print("  PASS: Код 401")
            return True
        print(f"  FAIL: Код {response.status_code} вместо 401")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d5():
    """D5: REST с неверным токеном — должен получить 401"""
    print("\nD5: REST с неверным токеном")
    try:
        audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
        with wave.open(audio_path, "rb") as wav:
            files = {"file": ("test.wav", wav.readframes(wav.getnframes()), "audio/wav")}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    GATEWAY_REST_URL,
                    files=files,
                    headers={"Authorization": "Bearer wrongtoken"}
                )
        if response.status_code == 401:
            print("  PASS: Код 401")
            return True
        print(f"  FAIL: Код {response.status_code} вместо 401")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d6():
    """D6: REST с верным токеном — должен получить 200"""
    print("\nD6: REST с верным токеном")
    try:
        audio_path = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"
        # Read entire file including WAV header
        with open(audio_path, "rb") as f:
            content = f.read()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GATEWAY_REST_URL,
                files={"file": ("test.wav", content, "audio/wav")},
                headers={"Authorization": "Bearer testtoken"}
            )
        data = response.json()
        if response.status_code == 200 and "text" in data:
            print(f"  PASS: Код 200, text='{data['text'][:30]}...'")
            return True
        print(f"  FAIL: Код {response.status_code}, body={data}")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def test_d7():
    """D7: WS без токена на порту 8101 (AUTH_MODE=none) — должен получить ready"""
    print("\nD7: WS без токена на порту 8101 (AUTH_MODE=none)")
    try:
        url = f"ws://127.0.0.1:{AUTH_NONE_PORT}/ws/stt"
        async with websockets.connect(url) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            if data.get("type") == "ready":
                print("  PASS: Получен ready без токена на порту 8101")
                await ws.send(json.dumps({"type": "stop"}))
                return True
            print(f"  FAIL: Ожидался ready, получен: {data}")
            return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


async def main():
    print("\n" + "=" * 50)
    print("=== Блок D: Auth tests ===")
    print("=" * 50)

    results = {}
    results["D1"] = await test_d1()
    await asyncio.sleep(0.3)
    results["D2"] = await test_d2()
    await asyncio.sleep(0.3)
    results["D3"] = await test_d3()
    await asyncio.sleep(0.3)
    results["D4"] = await test_d4()
    await asyncio.sleep(0.3)
    results["D5"] = await test_d5()
    await asyncio.sleep(0.3)
    results["D6"] = await test_d6()
    await asyncio.sleep(0.3)
    results["D7"] = await test_d7()

    passed = sum(1 for v in results.values() if v)
    print(f"\nБлок D: {passed}/7 pass")
    print(f"Результаты: {results}")


if __name__ == "__main__":
    asyncio.run(main())
