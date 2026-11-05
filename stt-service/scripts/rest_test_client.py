#!/usr/bin/env python3
"""
REST клиент для проверки Docker контейнера GigaSTT.

Читает audio-samples/manifest.tsv (144 файла).
Для каждого WAV делает POST http://127.0.0.1:9876/v1/transcribe (multipart file).
Собирает: path, reference, hypothesis, latency_sec, cpu_sec.
Считает WER через jiwer, агрегирует latency (median, std, min, max).
Сохраняет results/docker-rest-check/metrics.json и summary.md.
"""

import json
import time
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import jiwer


def median(values: list[float]) -> float:
    """Вычисляет медиану списка."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def load_manifest(manifest_path: Path) -> list[tuple[Path, str]]:
    """Загружает manifest.tsv."""
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
            if audio_path.exists():
                samples.append((audio_path, reference_text.strip()))
    return samples


def transcribe_file(audio_path: Path, reference: str, server_url: str) -> dict:
    """Транскрибирует один файл через REST API."""
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f)}
        response = requests.post(f"{server_url}/v1/transcribe", files=files, timeout=60)

    wall_latency = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu

    if response.status_code != 200:
        return {
            "path": str(audio_path),
            "reference": reference,
            "hypothesis": "",
            "latency_sec": wall_latency,
            "cpu_sec": cpu_time,
            "wer": 1.0,
            "error": f"HTTP {response.status_code}",
        }

    result = response.json()
    hypothesis = result.get("text", "")

    # Вычисляем WER с нормализацией
    def normalize_text(text: str) -> str:
        """Нормализует текст для сравнения."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[.,!?;:]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        wer_value = 0.0 if not hyp_norm else 1.0
    else:
        wer_value = jiwer.wer(ref_norm, hyp_norm)

    return {
        "path": str(audio_path),
        "reference": reference,
        "hypothesis": hypothesis,
        "latency_sec": wall_latency,
        "cpu_sec": cpu_time,
        "wer": wer_value,
    }


def main():
    server_url = "http://127.0.0.1:9876"
    manifest_path = Path("audio-samples/manifest.tsv")
    outdir = Path("results/docker-rest-check")

    print(f"=== REST Client для Docker контейнера GigaSTT ===")
    print(f"Сервер: {server_url}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {outdir}")
    print()

    # Проверка доступности сервера
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        print(f"Health: {health.json()}")
    except Exception as e:
        print(f"ERROR: Server not available: {e}")
        return

    # Загрузка manifest
    samples = load_manifest(manifest_path)
    print(f"Найдено файлов: {len(samples)}")
    print()

    # Прогон
    results = []
    all_latencies = []
    all_cpus = []
    total_wer = 0.0

    for i, (audio_path, reference) in enumerate(samples, 1):
        result = transcribe_file(audio_path, reference, server_url)
        results.append(result)

        all_latencies.append(result["latency_sec"])
        all_cpus.append(result["cpu_sec"])
        total_wer += result["wer"]

        if result.get("error"):
            print(f"[{i}/{len(samples)}] {audio_path.name}: ERROR - {result['error']}")
        else:
            status = "OK" if result["wer"] == 0 else f"WER={result['wer']:.2%}"
            print(f"[{i}/{len(samples)}] {audio_path.name}: {status} ({result['latency_sec']*1000:.0f}ms)")

    # Агрегация
    overall_wer = total_wer / len(samples) if samples else 0
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    latency_median = median(all_latencies)
    latency_std = (sum((x - avg_latency) ** 2 for x in all_latencies) / len(all_latencies)) ** 0.5 if len(all_latencies) > 1 else 0
    latency_min = min(all_latencies) if all_latencies else 0
    latency_max = max(all_latencies) if all_latencies else 0

    avg_cpu = sum(all_cpus) / len(all_cpus) if all_cpus else 0
    cpu_median = median(all_cpus)

    # Подсчёт ошибок
    total_subs = sum(1 for r in results if r.get("wer", 0) > 0 and r.get("hypothesis"))
    total_errors = sum(1 for r in results if r.get("wer", 0) > 0)

    # Сохранение metrics.json
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "docker-rest-check",
        "server_url": server_url,
        "total_files": len(samples),
        "valid_samples": len([r for r in results if not r.get("error")]),
        "samples": results,
        "summary": {
            "overall_wer": round(overall_wer, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "latency_median_sec": round(latency_median, 4),
            "latency_std_sec": round(latency_std, 4),
            "latency_min_sec": round(latency_min, 4),
            "latency_max_sec": round(latency_max, 4),
            "avg_cpu_sec": round(avg_cpu, 4),
            "cpu_median_sec": round(cpu_median, 4),
            "total_errors": total_errors,
        },
    }

    metrics_file = outdir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Генерация summary.md
    md_lines = [
        "# Проверка Docker контейнера GigaSTT через REST API",
        "",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Сервер:** {server_url}",
        "",
        "## Общие метрики",
        "",
        "| Метрика | Значение |",
        "|---------|----------|",
        f"| Всего файлов | {len(samples)} |",
        f"| WER | {overall_wer:.2%} |",
        f"| Latency (median) | {latency_median*1000:.0f} мс |",
        f"| Latency (std) | {latency_std*1000:.0f} мс |",
        f"| Latency (min/max) | {latency_min*1000:.0f} / {latency_max*1000:.0f} мс |",
        f"| CPU time (median) | {cpu_median*1000:.0f} мс |",
        f"| Ошибки | {total_errors} из {len(samples)} |",
        "",
        "## Сравнение с бенчмарком",
        "",
        "| Метрика | Docker REST | Бенчмарк (локальный) | Разница |",
        "|---------|-------------|---------------------|---------|",
        f"| WER | {overall_wer:.2%} | 2.08% | {'✅ Совпадает' if abs(overall_wer - 0.0208) < 0.02 else '⚠️ Отличается'} |",
        f"| Latency median | {latency_median*1000:.0f} мс | 226 мс | {'✅ В норме' if latency_median < 0.226 * 1.2 else '⚠️ Выше'} |",
        "",
        "## Выводы",
        "",
    ]

    if abs(overall_wer - 0.0208) < 0.02:
        md_lines.append("✅ WER совпадает с бенчмарком (2.08%). Контейнер работает корректно.")
    else:
        md_lines.append(f"⚠️ WER {overall_wer:.2%} отличается от бенчмарка (2.08%). Возможно, модель не загрузилась корректно.")

    if latency_median < 0.226 * 1.2:
        md_lines.append(f"✅ Latency в норме ({latency_median*1000:.0f} мс vs 226 мс). HTTP overhead приемлем.")
    else:
        md_lines.append(f"⚠️ Latency выше ожидаемого ({latency_median*1000:.0f} мс vs 226 мс).")

    with open(outdir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print()
    print(f"=== Результаты сохранены ===")
    print(f"metrics.json: {metrics_file}")
    print(f"summary.md: {outdir / 'summary.md'}")
    print()
    print(f"WER: {overall_wer:.2%}")
    print(f"Latency median: {latency_median*1000:.0f} мс")


if __name__ == "__main__":
    main()
