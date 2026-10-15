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


def parse_time(value):
    """Seconds or hh:mm:ss; empty means an open end."""
    if value is None or str(value).strip() == '':
        return None
    try:
        parts = [float(p) for p in str(value).strip().split(':')]
        if not 1 <= len(parts) <= 3 or any(not math.isfinite(p) or p < 0 for p in parts):
            raise ValueError()
        if len(parts) > 1 and any(p >= 60 for p in parts[1:]):
            raise ValueError()
        return sum(p * 60**i for i, p in enumerate(reversed(parts)))
    except ValueError:
        raise ValueError('时间请填写非负秒数或 时:分:秒') from None


def time_bounds(time_range, duration=None):
    start, end = (0., None) if time_range is None else time_range
    start, end = parse_time(start) or 0., parse_time(end)
    if end is not None and end <= start:
        raise ValueError('结束时间必须晚于开始时间')
    if duration is not None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法获取视频时长')
        if start >= duration or (end is not None and end > duration + .05):
            raise ValueError(f'处理时间超出视频时长（{duration:.2f} 秒）')
        end = min(end, duration) if end is not None else duration
    return start, end


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
