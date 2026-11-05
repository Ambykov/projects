#!/usr/bin/env python3

"""
Блок E4: Backpressure тест.

1. Открыть 20 WS-соединений и держать их открытыми
2. Открыть 21-е соединение — должно получить 4429
3. Проверить метрики: ws_active_sessions=20, ws_sessions_rejected_total{reason="limit"}=1
4. Закрыть 10 сессий
5. Открыть 5 новых — все 5 должны пройти
"""

import asyncio
import json

import httpx
import websockets
import os

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8100))

GATEWAY_WS_URL = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=testtoken"
METRICS_URL = f"http://127.0.0.1:{GATEWAY_PORT}/metrics"


async def keep_alive_session(session_id: int, event: asyncio.Event):
    """Поддержать сессию открытой до события."""
    try:
        ws = await websockets.connect(GATEWAY_WS_URL)
        await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
        print(f"  Session {session_id}: connected")
        await event.wait()  # Ждать сигнала закрыть
        await ws.close()
        print(f"  Session {session_id}: closed")
    except Exception as e:
        print(f"  Session {session_id}: error - {e}")


async def try_connect() -> tuple[bool, int | None, str]:
    """Попытаться подключиться и вернуть результат."""
    try:
        ws = await websockets.connect(GATEWAY_WS_URL, open_timeout=5.0)
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        await ws.close()
        return True, None, "ok"
    except websockets.exceptions.ConnectionClosedError as e:
        return False, e.code, f"closed:{e.code}"
    except asyncio.TimeoutError:
        return False, None, "timeout"
    except Exception as e:
        return False, None, f"error:{e}"


async def get_metric_value(name: str) -> float:
    """Получить значение метрики."""
    async with httpx.AsyncClient() as client:
        response = await client.get(METRICS_URL, timeout=5.0)
    for line in response.text.split("\n"):
        if line.startswith(f"stt_gateway_{name}"):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return 0.0


async def main():
    print("\n" + "=" * 60)
    print("БЛОК E4: Backpressure тест")
    print("=" * 60)

    # Шаг 1: Открыть 20 сессий
    print("\n--- Шаг 1: Открытие 20 сессий ---")
    close_event = asyncio.Event()
    sessions = []

    for i in range(20):
        ws = await websockets.connect(GATEWAY_WS_URL)
        await asyncio.wait_for(ws.recv(), timeout=5.0)  # ready
        sessions.append(ws)
        print(f"  Session {i+1}: connected")

    await asyncio.sleep(0.5)

    # Проверить метрики
    active = await get_metric_value("ws_active_sessions")
    print(f"\n  ws_active_sessions: {active}")

    # Шаг 2: Попытаться открыть 21-ю сессию
    print("\n--- Шаг 2: Попытка открыть 21-ю сессию ---")
    success, code, msg = await try_connect()
    print(f"  21-я сессия: {msg}")

    if code == 4429:
        print("  PASS: Получен код 4429 (Too Many Sessions)")
    else:
        print(f"  FAIL: Ожидался код 4429, получен {code}")

    # Проверить метрики после отказа
    rejected_limit = await get_metric_value("ws_sessions_rejected_total{reason=\"limit\"}")
    print(f"\n  ws_sessions_rejected_total{{reason=\"limit\"}}: {rejected_limit}")

    # Шаг 3: Закрыть 10 сессий
    print("\n--- Шаг 3: Закрытие 10 сессий ---")
    for i in range(10):
        await sessions[i].close()
        print(f"  Session {i+1}: closed")
    sessions = sessions[10:]

    await asyncio.sleep(1)

    active_after_close = await get_metric_value("ws_active_sessions")
    print(f"\n  ws_active_sessions после закрытия: {active_after_close}")

    # Шаг 4: Открыть 5 новых сессий
    print("\n--- Шаг 4: Открытие 5 новых сессий ---")
    new_sessions = []
    for i in range(5):
        try:
            ws = await websockets.connect(GATEWAY_WS_URL)
            await asyncio.wait_for(ws.recv(), timeout=5.0)
            new_sessions.append(ws)
            print(f"  New Session {i+1}: connected")
        except Exception as e:
            print(f"  New Session {i+1}: FAIL - {e}")

    await asyncio.sleep(0.5)

    active_final = await get_metric_value("ws_active_sessions")
    print(f"\n  ws_active_sessions в конце: {active_final}")

    # Закрыть все оставшиеся сессии
    for ws in sessions + new_sessions:
        await ws.close()

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ E4:")
    print(f"  21-я сессия отклонена с кодом 4429: {'PASS' if code == 4429 else 'FAIL'}")
    print(f"  ws_active_sessions = 20: {'PASS' if active == 20 else 'FAIL'}")
    print(f"  ws_sessions_rejected_total{{reason=\"limit\"}} >= 1: {'PASS' if rejected_limit >= 1 else 'FAIL'}")
    print(f"  5 новых сессий прошли: {'PASS' if len(new_sessions) == 5 else 'FAIL'}")
    print("=" * 60)


asyncio.run(main())
