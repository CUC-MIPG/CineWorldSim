"""Runtime compatibility shim for ScriptAgent dependencies.

Loaded automatically by Python when this directory is included in PYTHONPATH.
"""

from __future__ import annotations


def _patch_qwen_omni_utils() -> None:
    try:
        from qwen_omni_utils import vision_process as vp  # type: ignore
    except Exception:
        return

    defaults = {
        "IMAGE_FACTOR": getattr(vp, "FRAME_FACTOR", 2),
        "MIN_PIXELS": 4 * 28 * 28,
        "MAX_PIXELS": 16384 * 28 * 28,
        "VIDEO_MIN_PIXELS": 4 * 28 * 28,
        "VIDEO_MAX_PIXELS": 16384 * 28 * 28,
        "VIDEO_TOTAL_PIXELS": int(128000 * 28 * 28 * 0.9),
    }

    for key, value in defaults.items():
        if not hasattr(vp, key):
            setattr(vp, key, value)


_patch_qwen_omni_utils()
