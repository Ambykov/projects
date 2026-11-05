"""Load test for STT Gateway.

Usage:
    python scripts/load_test.py --clients 5 --utterances 10 --out results/load-test/5x10 --watch-pool
    python scripts/load_test.py --clients 20 --utterances 10 --out results/load-test/20x10 --watch-pool
"""
import argparse
import asyncio
import os
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

WS_URL = os.environ.get("STT_WS_URL", "ws://127.0.0.1:8100/ws/stt")
TOKEN = os.environ.get("STT_API_KEY", "testtoken")
GIGASTT_READY_URL = os.environ.get("GIGASTT_READY_URL", "http://127.0.0.1:9876/ready")

CHUNK_MS = 100
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * BYTES_PER_SAMPLE


def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16000, f"expected 16000 Hz, got {wav.getframerate()}"
        assert wav.getnchannels() == 1, f"expected mono, got {wav.getnchannels()}"
        assert wav.getsampwidth() == 2, f"expected 16-bit, got {wav.getsampwidth()*8}-bit"
        return wav.readframes(wav.getnframes())


async def pool_watcher(stop_event: asyncio.Event, pool_log: list, interval_sec: float = 0.5):
    async with httpx.AsyncClient(timeout=2.0) as client:
        while not stop_event.is_set():
            try:
                r = await client.get(GIGASTT_READY_URL)
                data = r.json()
                pool_log.append({
                    "t": round(time.time(), 3),
                    "available": data.get("pool_available"),
                    "total": data.get("pool_total"),
                })
            except Exception as e:
                pool_log.append({"t": round(time.time(), 3), "error": str(e)})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                pass


async def run_client(client_id: int, n_utterances: int, audio_bytes: bytes, timeout_sec: float) -> list[dict]:
    results = []
    url = f"{WS_URL}?token={TOKEN}"
    try:
        async with websockets.connect(url, ping_interval=30, ping_timeout=10, open_timeout=10) as ws:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ready = json.loads(raw)
                if ready.get("type") != "ready":
                    return [{"client_id": client_id, "utterance_id": -1,
                             "error": f"unexpected first message: {ready}"}]
            except asyncio.TimeoutError:
                return [{"client_id": client_id, "utterance_id": -1, "error": "ready_timeout"}]
            except ConnectionClosed as e:
                code = e.rcvd.code if e.rcvd else None
                return [{"client_id": client_id, "utterance_id": -1, "error": "closed_before_ready", "close_code": code}]

            for u in range(n_utterances):
                t0 = time.perf_counter()
                ttfp_ms = None
                ttft_ms = None
                final_text = None
                error = None
                close_code = None

                try:
                    for off in range(0, len(audio_bytes), CHUNK_BYTES):
                        await ws.send(audio_bytes[off:off+CHUNK_BYTES])
                    await ws.send(json.dumps({"type": "end_of_utterance"}))
                except ConnectionClosed as e:
                    close_code = e.rcvd.code if e.rcvd else None
                    results.append({"client_id": client_id, "utterance_id": u,
                                    "error": "send_failed", "close_code": close_code})
                    break

                try:
                    deadline = time.perf_counter() + timeout_sec
                    while True:
                        remain = deadline - time.perf_counter()
                        if remain <= 0:
                            raise asyncio.TimeoutError()
                        raw = await asyncio.wait_for(ws.recv(), timeout=remain)
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        if mtype == "partial":
                            if ttfp_ms is None:
                                ttfp_ms = round((time.perf_counter() - t0) * 1000, 1)
                        elif mtype == "final":
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 1)
                            final_text = msg.get("text", "")
                            break
                        elif mtype == "error":
                            error = msg.get("code")
                            break
                        elif mtype == "ready":
                            continue
                except asyncio.TimeoutError:
                    error = "client_timeout"
                except ConnectionClosed as e:
                    close_code = e.rcvd.code if e.rcvd else None
                    error = "connection_closed"

                results.append({
                    "client_id": client_id, "utterance_id": u,
                    "ttfp_ms": ttfp_ms, "ttft_ms": ttft_ms, "final_text": final_text,
                    "error": error, "close_code": close_code,
                })
                if error:
                    break

            try:
                await ws.send(json.dumps({"type": "stop"}))
                await ws.close()
            except Exception:
                pass
    except Exception as e:
        results.append({"client_id": client_id, "utterance_id": -1, "error": f"connect_failed: {e}"})
    return results


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--utterances", type=int, required=True)
    parser.add_argument("--wav", type=Path,
                        default=Path("audio-samples/commands/msc/stop/2023-06-26-21-14-35.wav"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--watch-pool", action="store_true")
    args = parser.parse_args()

    audio_bytes = load_wav(args.wav)
    args.out.mkdir(parents=True, exist_ok=True)

    pool_log = []
    stop_event = asyncio.Event()
    watcher = None
    if args.watch_pool:
        watcher = asyncio.create_task(pool_watcher(stop_event, pool_log))

    t_start = time.perf_counter()
    tasks = [run_client(i, args.utterances, audio_bytes, args.timeout) for i in range(args.clients)]
    all_results_nested = await asyncio.gather(*tasks)
    t_elapsed = time.perf_counter() - t_start

    stop_event.set()
    if watcher:
        await watcher

    results = [r for lst in all_results_nested for r in lst]
    ttfp_vals = [r["ttfp_ms"] for r in results if r.get("ttfp_ms")]
    ttft_vals = [r["ttft_ms"] for r in results if r.get("ttft_ms")]

    errors = {}
    close_codes = {}
    for r in results:
        if r.get("error"):
            errors[r["error"]] = errors.get(r["error"], 0) + 1
        if r.get("close_code") is not None:
            close_codes[str(r["close_code"])] = close_codes.get(str(r["close_code"]), 0) + 1

    pool_avail = [p["available"] for p in pool_log if p.get("available") is not None]

    total_theoretical = args.clients * args.utterances
    total_sent = len(results)
    total_completed = len([r for r in results if r.get("ttft_ms")])
    total_errors = len([r for r in results if r.get("error")])

    clients_full = 0
    for lst in all_results_nested:
        if len(lst) == args.utterances and all(r.get("ttft_ms") for r in lst):
            clients_full += 1
    clients_aborted = args.clients - clients_full

    summary = {
        "scenario": f"{args.clients}x{args.utterances}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clients": args.clients,
        "utterances_per_client": args.utterances,
        "total_theoretical": total_theoretical,
        "total_sent": total_sent,
        "total_completed": total_completed,
        "total_errors": total_errors,
        "clients_full": clients_full,
        "clients_aborted": clients_aborted,
        "elapsed_sec": round(t_elapsed, 2),
        "ttfp_ms": {
            "count": len(ttfp_vals),
            "p50": pct(ttfp_vals, 50), "p95": pct(ttfp_vals, 95), "p99": pct(ttfp_vals, 99),
            "min": min(ttfp_vals) if ttfp_vals else None,
            "max": max(ttfp_vals) if ttfp_vals else None,
        },
        "ttft_ms": {
            "count": len(ttft_vals),
            "p50": pct(ttft_vals, 50), "p95": pct(ttft_vals, 95), "p99": pct(ttft_vals, 99),
            "min": min(ttft_vals) if ttft_vals else None,
            "max": max(ttft_vals) if ttft_vals else None,
        },
        "errors_by_code": errors,
        "close_codes": close_codes,
        "pool_min_available": min(pool_avail) if pool_avail else None,
        "pool_samples": len(pool_log),
    }

    (args.out / "metrics.json").write_text(
        json.dumps({"summary": summary, "results": results, "pool_log": pool_log}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
