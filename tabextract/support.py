"""Shared I/O, validation and cancellation helpers."""

import math
import os
import subprocess
import sys
from pathlib import Path
import cv2


class CancelledError(RuntimeError):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise CancelledError("已取消提取")


def write_image(path, image):
    """Handle Chinese Windows paths without cv2.imwrite's path limitations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def open_file(path):
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(
            ["open" if sys.platform == "darwin" else "xdg-open", str(path)]
        )


def parse_bars_per_row(value):
    """Accept automatic layout or an explicit integer shared by CLI and GUI."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto", "自动")):
        return None
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ValueError("每行小节数请选择自动，或填写 1 到 8 之间的整数") from None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
        raise ValueError("每行小节数请选择自动，或填写 1 到 8 之间的整数")
    return value


def validate_options(sample_fps=6, dpi=300, bars_per_row=None):
    if not math.isfinite(sample_fps) or not 0.5 <= sample_fps <= 30:
        raise ValueError("采样频率必须在 0.5 到 30 之间")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("DPI 必须是 72 到 600 之间的整数")
    if bars_per_row is not None and (
        isinstance(bars_per_row, bool)
        or not isinstance(bars_per_row, int)
        or not 1 <= bars_per_row <= 8
    ):
        raise ValueError("每行小节数必须是 1 到 8 之间的整数")
