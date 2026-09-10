"""Separate static notation from footage using within-page consensus.
No OCR: letters, dots, ties and stems stay image strokes. No area-based deletion
of small components; no vertical clipping to the six staff lines.
"""

import cv2
import numpy as np
from .geometry import staff_groups
from .support import check_cancel


def stroke_response(bgr, polarity="bright", kernel=9):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    neutral = bgr.min(axis=2) if polarity == "bright" else 255 - bgr.max(axis=2)
    if polarity == "dark":
        gray = 255 - gray
    # A pair of 1-D openings erases L/T junctions: each direction mistakes
    # the crossing stroke for background. A 2-D opening preserves those
    # evidenced junctions together with the thin stems and beams.
    response = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, np.ones((kernel, kernel), np.uint8)
    )
    response[neutral < 50] = 0
    return response


def infer_polarity(bgr):
    """Use long-line evidence rather than total dark/bright background area."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    above, below = np.roll(gray, 2, axis=0), np.roll(gray, -2, axis=0)
    scores = []
    for sign in (1, -1):
        ridge = np.minimum(sign * (gray - above), sign * (gray - below)) > 25
        scores.append(float(np.sort(ridge.mean(axis=1))[-12:].sum()))
    return "bright" if scores[0] >= scores[1] else "dark"


def clean_frames(frames, polarity="auto", upscale=3, cancel_event=None):
    if not frames:
        raise ValueError("没有用于清洗的帧")
    if polarity == "auto":
        votes = [
            infer_polarity(frames[i]) for i in (0, len(frames) // 2, len(frames) - 1)
        ]
        polarity = max(set(votes), key=votes.count)
    features = []
    for frame in frames:
        check_cancel(cancel_event)
        image = cv2.resize(
            frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
        features.append(stroke_response(image, polarity, kernel=3 * upscale))
    stack = np.stack(features)
    low, high = np.percentile(stack, [25, 75], axis=0)
    del stack, features
    alpha = np.clip((high - 8) * (255 / 35), 0, 255)
    alpha[low < 10] = 0
    page = 255 - alpha.astype(np.uint8)
    # Accept faint pixels only within evidenced horizontal line bands.
    groups = staff_groups(page)
    for lines in groups:
        for line in lines:
            y = int(round(line))
            a, b = max(0, y - upscale), min(page.shape[0], y + upscale + 1)
            valid = low[a:b] >= 5
            line_alpha = np.clip((high[a:b] - 4) * (255 / 30), 0, 255)
            page[a:b] = np.minimum(
                page[a:b], np.where(valid, 255 - line_alpha, 255).astype(np.uint8)
            )
    if groups:
        # Remove detached scenery far from all staff systems. Components touching
        # the notation envelope are kept WHOLE, even when stems extend below it.
        zone = np.zeros(page.shape[0], bool)
        for lines in groups:
            spacing = float(np.median(np.diff(lines)))
            a = max(0, round(lines[0] - spacing * 4))
            b = min(page.shape[0], round(lines[-1] + spacing * 4) + 1)
            zone[a:b] = True
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (page < 245).astype(np.uint8), 8
        )
        keep = np.zeros(count, bool)
        for i in range(1, count):
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            keep[i] = zone[y : y + h].any()
        page[~keep[labels]] = 255
    return page


def clean_page(bgr_img, upscale=3, **_compat):
    """Single-image compatibility entry point; videos use clean_frames."""
    return clean_frames([bgr_img], upscale=upscale)
