#!/usr/bin/env python3
"""
download-gigastt-model.py — скачивание модели GigaSTT (если нужно вручную).

Модель обычно скачивается автоматически при первом запуске gigastt.Transcriber().
Этот скрипт — для явного скачивания.
"""

import gigastt

print("Downloading GigaSTT model...")
gigastt.download()
print("Done. Model stored in ~/.gigastt/models/")
