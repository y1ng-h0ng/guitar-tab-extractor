"""自动检测视频中吉他谱所在的矩形区域。

原理：吉他谱区域含有大量长水平线（六线谱的谱线），并且这些
线在多帧之间位置稳定。对采样帧做二值化 + 水平形态学开运算提取
长水平线，统计每行的线像素密度及其跨帧“持续率”，密度高且持续
率高的连续行带即为谱面区域。动画背景（渐变等值线、移动色块）
虽然也可能产生水平边缘，但位置随时间漂移，持续率低，会被过滤。
"""

import cv2
import numpy as np
from .geometry import staff_from_strength


def sample_frames(video_path, count=24):
    """从视频中均匀采样若干帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        seen = 0
        rng = np.random.default_rng(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seen += 1
            if len(frames) < count:
                frames.append(frame)
            else:
                slot = int(rng.integers(seen))
                if slot < count:
                    frames[slot] = frame
        cap.release()
        return frames
    idxs = np.linspace(0, total - 1, min(count, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _thin_horizontal_lines(binary, max_line_height=10, min_span_ratio=0.0):
    """从二值图中提取细长的水平线组件。"""
    h, w = binary.shape
    k_open = max(w // 160, 8)
    k_close = max(w // 10, 48)
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, 1)))
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    if n <= 1:
        return lines
    valid = stats[1:, cv2.CC_STAT_HEIGHT] <= max_line_height
    if min_span_ratio > 0:
        valid &= stats[1:, cv2.CC_STAT_WIDTH] >= int(w * min_span_ratio)
    lut = np.zeros(n, dtype=np.uint8)
    lut[1:][valid] = 255
    return lut[labels]


def _horizontal_line_mask(gray, min_span_ratio=0.0, ridge_diff=30):
    g = gray.astype(np.int16)
    above = np.empty_like(g); above[4:] = g[:-4]; above[:4] = g[:1]
    below = np.empty_like(g); below[:-4] = g[4:]; below[-4:] = g[-1:]
    bright = (((g - above) > ridge_diff) & ((g - below) > ridge_diff)).astype(np.uint8) * 255
    dark = (((above - g) > ridge_diff) & ((below - g) > ridge_diff)).astype(np.uint8) * 255
    return np.maximum(_thin_horizontal_lines(bright, min_span_ratio=min_span_ratio), _thin_horizontal_lines(dark, min_span_ratio=min_span_ratio))


def _detect_from_masks(masks, H, W, search_top_ratio, pad):
    row_counts = np.stack([(m > 0).sum(axis=1) for m in masks]).astype(np.float64)
    row_mean = row_counts.mean(axis=0)
    persist = (row_counts > max(W * 0.10, 40)).mean(axis=0)
    score = row_mean * persist
    top = int(H * search_top_ratio)
    band = score[top:]
    fallback = (0, int(H * 0.45), W, H - int(H * 0.45), 0.0)
    if band.size == 0 or band.max() <= 0:
        return fallback
    normalized = band / band.max()
    groups = staff_from_strength(normalized, coverage=0.18, min_spacing=4, peak_gap=1)
    groups = [g for g in groups if np.median(np.diff(g)) <= 40]
    if not groups:
        return fallback
    lines = max(groups, key=lambda g: sum(normalized[int(round(y))] for y in g))
    run_start = int(lines[0]); run_end = int(np.ceil(lines[-1])) + 1
    confidence = float(np.mean([normalized[int(round(y))] for y in lines]))
    spacing = float(np.median(np.diff(lines)))
    pad_top = int(spacing * 4) + 8; pad_bot = int(spacing * 3) + 8
    y0 = top + max(run_start - pad_top, 0); y1 = top + min(run_end + pad_bot, band.size)
    col_acc = np.zeros(W, dtype=np.float64)
    for m in masks:
        col_acc += (m[y0:y1, :] > 0).sum(axis=0)
    col_acc /= len(masks)
    if col_acc.max() <= 0:
        x0, x1 = 0, W
    else:
        cols = np.where(col_acc > col_acc.max() * 0.15)[0]
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
    x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0); x1 = min(x1 + pad, W); y1 = min(y1 + pad, H)
    return x0, y0, x1 - x0, y1 - y0, confidence


def detect_tab_region(video_path, search_top_ratio=0.35, sample_count=24, pad=8):
    frames = sample_frames(video_path, sample_count)
    if not frames:
        raise RuntimeError("视频中没有可读取的帧")
    H, W = frames[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    for span in (0.45, 0.0):
        masks = [_horizontal_line_mask(g, span) for g in grays]
        x, y, w, h, conf = _detect_from_masks(masks, H, W, search_top_ratio, pad)
        if conf > 0:
            return x, y, w, h, conf
    return 0, int(H * 0.45), W, H - int(H * 0.45), 0.0
