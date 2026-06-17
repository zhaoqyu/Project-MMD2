from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from transformers import set_seed
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)

# def set_global_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     try:
#         import torch

#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     except Exception:
#         pass


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def dataset_slug(dataset_name: str) -> str:
    return slugify(dataset_name.replace("/", "_"))


def latest_checkpoint_dir(run_dir: str | Path) -> Path | None:
    checkpoints = sorted(Path(run_dir).glob("checkpoints/checkpoint-*"))
    return checkpoints[-1] if checkpoints else None

