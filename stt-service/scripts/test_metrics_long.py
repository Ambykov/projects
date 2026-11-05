"""Block F: Metrics test with long utterance (4 sec) for partials."""
import asyncio
import json
import os
import wave

import httpx
import websockets

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8100))
GATEWAY_WS_URL = f"ws://127.0.0.1:{GATEWAY_PORT}/ws/stt?token=testtoken"
METRICS_URL = f"http://127.0.0.1:{GATEWAY_PORT}/metrics"

# Files for 4-second combined audio
LONG_AUDIO_FILES = [
    "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav",
    "audio-samples/commands/msc/go/2023-08-25-15-13-05.wav",
    "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav",
    "audio-samples/commands/msc/go/2023-08-25-15-13-05.wav",
]
SHORT_AUDIO_PATH = "audio-samples/commands/msc/stop/2023-08-26-17-01-25.wav"


def load_long_audio():
    """Load 4-second combined audio."""
    import numpy as np
    samples = []
    for path in LONG_AUDIO_FILES:
        with wave.open(path, "rb") as wav:
            samples.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16))
    return np.concatenate(samples).tobytes()


def load_short_audio():
    """Load short audio (1 sec)."""
    with wave.open(SHORT_AUDIO_PATH, "rb") as wav:
        return wav.readframes(wav.getnframes())


async def get_metrics_raw():
    async with httpx.AsyncClient() as client:
        response = await client.get(METRICS_URL, timeout=5.0)
    return response.text


async def get_metric_value(metrics_text: str, name: str, labels: dict = None) -> float:
    for line in metrics_text.split("\n"):
        if line.startswith(f"stt_gateway_{name}"):
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str in line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            pass
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
    return 0.0


async def send_long_utterance(ws):
    """Send long utterance (4 sec) and collect responses."""
    audio_data = load_long_audio()
    chunk_size = 1600  # 100ms chunks
    partials = []
    
    for i in range(0, len(audio_data), chunk_size):
        await ws.send(audio_data[i:i+chunk_size])
    
    await ws.send(json.dumps({"type": "end_of_utterance"}))
    
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(msg)
        if data.get("type") == "partial":
            partials.append(data.get("text", ""))
        elif data.get("type") == "final":
            return len(partials), data.get("text", "")
    return 0, ""


async def send_short_utterance(ws):
    """Send short utterance (1 sec) and get final."""
    audio_data = load_short_audio()
    await ws.send(audio_data)
    await ws.send(json.dumps({"type": "end_of_utterance"}))
    
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(msg)
        if data.get("type") == "final":
            return data.get("text", "")
    return ""


async def main():
    print("\n" + "=" * 60)
    print("БЛОК F: Метрики (длинная + короткая реплики)")
    print("=" * 60)

    # Метрики ДО
    print("\n--- Метрики ДО ---")
    metrics_before = await get_metrics_raw()
    
    partials_before = await get_metric_value(metrics_before, "partials_received_total", {})
    ttfp_before = await get_metric_value(metrics_before, "ws_time_to_first_partial_seconds_count", {})
    final_before = await get_metric_value(metrics_before, "ws_time_to_final_seconds_count", {})
    
    print(f"  partials_received_total: {partials_before}")
    print(f"  ws_time_to_first_partial_seconds_count: {ttfp_before}")
    print(f"  ws_time_to_final_seconds_count: {final_before}")

    # Длинная реплика (4 сек)
    print("\n--- Длинная реплика (4 сек) ---")
    async with websockets.connect(GATEWAY_WS_URL) as ws:
        await ws.recv()  # ready
        partials, final_text = await send_long_utterance(ws)
        print(f"  Partials: {partials}")
        print(f"  Final: '{final_text[:30]}...'")
        await ws.send(json.dumps({"type": "stop"}))

    await asyncio.sleep(0.5)

    # Метрики ПОСЛЕ длинной
    print("\n--- Метрики ПОСЛЕ длинной ---")
    metrics_after_long = await get_metrics_raw()
    
    partials_after_long = await get_metric_value(metrics_after_long, "partials_received_total", {})
    ttfp_after_long = await get_metric_value(metrics_after_long, "ws_time_to_first_partial_seconds_count", {})
    final_after_long = await get_metric_value(metrics_after_long, "ws_time_to_final_seconds_count", {})
    
    print(f"  partials_received_total: {partials_after_long} (дельта: {partials_after_long - partials_before:+.0f})")
    print(f"  ws_time_to_first_partial_seconds_count: {ttfp_after_long} (дельта: {ttfp_after_long - ttfp_before:+.0f})")
    print(f"  ws_time_to_final_seconds_count: {final_after_long} (дельта: {final_after_long - final_before:+.0f})")

    # Короткие реплики (3 x 1 сек)
    print("\n--- 3 коротких реплики (1 сек каждая) ---")
    async with websockets.connect(GATEWAY_WS_URL) as ws:
        await ws.recv()  # ready
        for i in range(3):
            final_text = await send_short_utterance(ws)
            print(f"  Реплика {i+1}: '{final_text[:20]}...'")
        await ws.send(json.dumps({"type": "stop"}))

    await asyncio.sleep(0.5)

    # Метрики ПОСЛЕ коротких
    print("\n--- Метрики ПОСЛЕ коротких ---")
    metrics_after_short = await get_metrics_raw()
    
    partials_after_short = await get_metric_value(metrics_after_short, "partials_received_total", {})
    ttfp_after_short = await get_metric_value(metrics_after_short, "ws_time_to_first_partial_seconds_count", {})
    final_after_short = await get_metric_value(metrics_after_short, "ws_time_to_final_seconds_count", {})
    
    print(f"  partials_received_total: {partials_after_short} (дельта: {partials_after_short - partials_after_long:+.0f})")
    print(f"  ws_time_to_first_partial_seconds_count: {ttfp_after_short} (дельта: {ttfp_after_short - ttfp_after_long:+.0f})")
    print(f"  ws_time_to_final_seconds_count: {final_after_short} (дельта: {final_after_short - final_after_long:+.0f})")

    # Проверка
    print("\n--- Проверка ---")
    # Длинная: partials > 0, TTFP вырос
    long_ok = (partials_after_long - partials_before) > 0 and (ttfp_after_long - ttfp_before) > 0
    # Короткие: partials не вырос, final вырос на 3
    short_partials_ok = (partials_after_short - partials_after_long) == 0
    short_final_ok = (final_after_short - final_after_long) == 3
    
    print(f"  Длинная: partials>0 и TTFP растёт: {'PASS' if long_ok else 'FAIL'}")
    print(f"  Короткие: partials не растёт: {'PASS' if short_partials_ok else 'FAIL'}")
    print(f"  Короткие: final растёт на 3: {'PASS' if short_final_ok else 'FAIL'}")

    block_f_pass = long_ok and short_partials_ok and short_final_ok

    print("\n" + "=" * 60)
    print(f"БЛОК F: {'PASS' if block_f_pass else 'FAIL'}")
    print("=" * 60)

    return block_f_pass


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
