"""Detect available compute (CPU, RAM, GPUs) and choose a model size preset."""
from __future__ import annotations

import os

from .config import PRESETS


def detect() -> dict:
    cpus = os.cpu_count() or 1
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = float("nan")
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 1e9, 1)})
    except ImportError:
        pass
    return {"cpus": cpus, "ram_gb": round(ram_gb, 1), "gpus": gpus, "n_gpus": len(gpus)}


def choose_preset(hw: dict, requested: str = "auto") -> str:
    """Pick the largest preset the hardware can comfortably hold.

    Predictor memory scales with per-GPU VRAM; the Market Analyser runs on CPU so it
    also scales with RAM/cores.
    """
    if requested != "auto":
        if requested not in PRESETS:
            raise ValueError(f"unknown size {requested!r}; choose from {list(PRESETS)} or auto")
        return requested
    min_vram = min((g["vram_gb"] for g in hw["gpus"]), default=0.0)
    if min_vram >= 80 and hw["ram_gb"] >= 64 and hw["cpus"] >= 8:
        return "xlarge"
    if min_vram >= 40 and hw["ram_gb"] >= 32:
        return "large"
    if min_vram >= 16 or (hw["n_gpus"] == 0 and hw["ram_gb"] >= 16 and hw["cpus"] >= 4):
        return "medium"
    return "small"


if __name__ == "__main__":
    import json

    hw = detect()
    hw["preset"] = choose_preset(hw)
    print(json.dumps(hw, indent=2))
