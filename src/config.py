from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

# Убираем информационный шум TensorFlow в консоли.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

SEED = 42
VARIANT_COLUMNS = ["x1", "x2", "x3", "y"]
EXPECTED_SCHEMA = {
    "id": "integer",
    "timestamp": "datetime",
    "category": "category",
    "x1": "numeric",
    "x2": "numeric",
    "x3": "numeric",
    "y": "numeric",
}

# Один timestamp на весь запуск — все файлы относятся к одной сессии.
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prepare_output_dir() -> Path:
    """Создаёт output/ и удаляет результаты предыдущего запуска."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in OUTPUT_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)
    return OUTPUT_DIR


def output_path(base_name: str, suffix: str) -> Path:
    return OUTPUT_DIR / f"{base_name}_{RUN_TIMESTAMP}.{suffix.lstrip('.')}"


def save_figure(fig, base_name: str) -> Path:
    """Сохраняет PNG с общей меткой времени текущего запуска."""
    path = output_path(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def set_seeds(seed: int = SEED) -> dict[str, str]:
    random.seed(seed)
    np.random.seed(seed)
    statuses = {"python": "ok", "numpy": "ok"}

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        statuses["torch"] = "ok"
    except Exception as exc:
        statuses["torch"] = f"unavailable: {exc}"

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        statuses["tensorflow"] = "ok"
    except Exception as exc:
        statuses["tensorflow"] = f"unavailable: {exc}"

    return statuses
