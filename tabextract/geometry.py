"""Observed staff and bar lines; no reconstruction of musical symbols."""

import cv2
import numpy as np


def spans(indices, gap=1):
    indices = np.asarray(indices)
    if not len(indices):
        return []
    return [
        (int(g[0]), int(g[-1]))
        for g in np.split(indices, np.where(np.diff(indices) > gap)[0] + 1)
    ]


def staff_groups(page, coverage=0.24):
    """Find six equally spaced lines, accounting for line thickness."""
    strength = (page < 180).mean(axis=1)
    return staff_from_strength(strength, coverage)


def staff_from_strength(strength, coverage=0.24, min_spacing=2, peak_gap=2):
    """Six observed line peaks, also usable for temporal region evidence."""
    peaks = spans(np.where(strength > coverage)[0], gap=peak_gap)
    centers = [
        float(np.average(np.arange(a, b + 1), weights=strength[a : b + 1]))
        for a, b in peaks
    ]
    candidates = []
    centers = np.array(centers)
    combinations = set()
    for i in range(len(centers) - 5):
        for j in range(i + 5, len(centers)):
            step = (centers[j] - centers[i]) / 5
            predicted = centers[i] + np.arange(6) * step
            indices = np.abs(centers[:, None] - predicted).argmin(axis=0)
            if len(set(indices)) == 6:
                combinations.add(tuple(indices))
    for indices in combinations:
        lines = centers[list(indices)]
        gaps = np.diff(lines)
        space = float(np.median(gaps))
        if space < min_spacing or np.max(abs(gaps - space)) > max(1, space * 0.24):
            continue
        score = sum(float(strength[peaks[k][0] : peaks[k][1] + 1].max()) for k in indices)
        score *= 1 - float(np.mean(abs(gaps - space))) / space
        candidates.append((score, lines))
    chosen = []
    for _, lines in sorted(candidates, key=lambda p: p[0], reverse=True):
        if not any(lines[0] <= old[-1] and lines[-1] >= old[0] for old in chosen):
            chosen.append(lines)
    return sorted(chosen, key=lambda g: g[0])


def bar_lines(page, lines):
    """Strokes through all six lines, excluding down stems and playheads."""
    space = float(np.median(np.diff(lines)))
    top, bottom = int(round(lines[0])), int(round(lines[-1]))
    ink = (page < 170).astype(np.uint8)
    radius = max(1, int(round(space / 12)))
    near = cv2.dilate(ink, np.ones((1, radius * 2 + 1), np.uint8))
    candidate = near[top : bottom + 1].mean(axis=0) > 0.88
    runs = spans(np.where(candidate)[0], gap=max(2, int(space * 0.5)))
    bars = []
    for a, b in runs:
        if b - a > space * 0.9:
            continue
        x = a + int(np.argmax(ink[top : bottom + 1, a : b + 1].sum(axis=0)))
        below_a = min(page.shape[0], bottom + max(3, round(space * 0.25)))
        below_b = min(page.shape[0], bottom + round(space * 2))
        side = max(2, round(space * 0.4))
        strip = ink[
            below_a:below_b, max(0, a - side) : min(page.shape[1], b + side + 1)
        ]
        occupied = np.where(strip.any(axis=1))[0] if strip.size else []
        if any(v - u + 1 >= space * 0.65 for u, v in spans(occupied)):
            continue
        if not bars or x - bars[-1] > space * 2.2:
            bars.append(x)
    return bars


def trim_vertical(image, margin=10):
    """Include all retained symbols, not only the staff height."""
    rows = np.where((image < 245).any(axis=1))[0]
    if not rows.size:
        return image
    return np.pad(
        image[rows[0] : rows[-1] + 1], ((margin, margin), (0, 0)), constant_values=255
    )
