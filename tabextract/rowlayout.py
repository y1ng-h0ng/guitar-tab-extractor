"""Keep crossing notation together and justify rows without stretching glyphs."""

import cv2
import numpy as np
from .geometry import spans


def crossing_marks(image, lines, boundaries):
    """Find observed ink bridges across bars (ties/slurs), without OCR."""
    space = float(np.median(np.diff(lines)))
    top, bottom = round(lines[0]), round(lines[-1])
    ink = (image < 200).astype(np.uint8)
    band = max(1, round(space * 0.19))
    for y in lines:
        yy = round(y)
        ink[max(0, yy - band) : yy + band + 1] = 0
    radius = round(space * 3)
    guard = max(1, round(space * 0.19))
    bridge = max(3, round(space * 0.6)) | 1
    a, b = max(0, top - round(space * 2)), min(image.shape[0], round(bottom + space * 3))
    found = []
    for boundary in boundaries:
        if boundary is None or boundary < radius or boundary + radius >= image.shape[1]:
            found.append(False)
            continue
        patch = ink[a:b, boundary - radius : boundary + radius + 1].copy()
        patch[max(0, top - a - guard) : bottom - a + guard + 1, radius - guard : radius + guard + 1] = 0
        joined = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        crossing = False
        for x, y, w, h, area in stats[1:count]:
            if (x < radius - space and x + w > radius + space and max(2, space * 0.1) < h < space * 4 and area > space * 0.5):
                crossing = True
                break
        found.append(crossing)
    return found


def choose_auto_breaks(units, crossings, staff_spacing):
    """Balance source widths at a readable A4 scale, keeping whole measures.

    About 78 staff spaces fit the PDF content width at a 6.5pt string gap;
    reserve four spaces for row margins. Dense measures get fewer neighbors,
    sparse measures more. Source resolution cannot change this decision.
    """
    total = len(units)
    if not total:
        return []
    desired = max(1.0, staff_spacing * 74)
    costs, previous = [float("inf")] * (total + 1), [None] * (total + 1)
    costs[0] = 0.0
    for end in range(1, total + 1):
        for count in range(1, min(8, end) + 1):
            start = end - count
            width = units[end - 1][3] - units[start][2]
            ratio = width / desired
            penalty = 1 + 3 * (1 - ratio) ** 2 + 12 * max(0, ratio - 1) ** 2
            if end < total and crossings[end - 1]:
                penalty += 100
            cost = costs[start] + penalty
            if cost < costs[end] - 1e-9:
                costs[end], previous[end] = cost, start
    groups, end = [], total
    while end:
        start = previous[end]
        groups.append((start, end))
        end = start
    return groups[::-1]


def choose_breaks(units, crossings, target):
    total = len(units)
    if not total:
        return []
    usual_width = max(units[min(i + target, total) - 1][3] - units[i][2] for i in range(0, total, target))
    costs = [float("inf")] * (total + 1)
    previous = [None] * (total + 1)
    costs[0] = 0
    for end in range(1, total + 1):
        for count in range(1, min(end, target + 2) + 1):
            start = end - count
            if count < max(1, target - 2) and end != total:
                continue
            if end == total and count <= target:
                deviation = 0.12 * (target - count) ** 2
                if count == 1 and total > target:
                    deviation += 2
            else:
                deviation = float((count - target) ** 2)
            phase = min(end % target, (-end) % target)
            penalty = 0 if end == total else 0.8 * phase ** 2
            if end < total and crossings[end - 1]:
                penalty += 100
            width = units[end - 1][3] - units[start][2]
            penalty += 40 * max(0, width / max(1, usual_width) - 1) ** 2
            cost = costs[start] + deviation + penalty + 0.05
            if cost < costs[end] - 1e-9:
                costs[end], previous[end] = cost, start
    groups = []
    end = total
    while end:
        start = previous[end]
        groups.append((start, end))
        end = start
    return groups[::-1]


def justify_row(row, lines, target_width):
    """Insert space only in blank channels; copy glyphs at their original size."""
    extra = target_width - row.shape[1]
    if extra <= 0:
        return row, []
    space = float(np.median(np.diff(lines)))
    band = max(1, round(space * 0.25))
    nonstaff = row < 245
    horizontal = cv2.morphologyEx(
        nonstaff.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((1, max(3, round(space * .3))), np.uint8)
    ) > 0
    forbidden = np.zeros(row.shape[1], bool)
    for y in lines:
        yy = round(y)
        region = np.s_[max(0, yy - band):yy + band + 1]
        # Ignore only evidenced long strings, not an entire horizontal band:
        # a tiny dot centered on a string must not become an expansion column.
        forbidden |= (nonstaff[region] & ~horizontal[region]).any(axis=0)
        nonstaff[region] = False
    content = np.where(nonstaff.any(axis=0))[0]
    if not content.size:
        return row, []
    # Dense rhythms and dashed technique lines can leave only narrow safe
    # channels. Reduce the comfort margin before giving up; every insertion
    # still passes through a column with no non-staff ink.
    gaps = []
    for fraction in (.6, .3, .1, 0):
        guard = round(space * fraction)
        occupied = cv2.dilate(nonstaff.any(axis=0).astype(np.uint8)[None, :],
                              np.ones((1, 2 * guard + 1), np.uint8))[0]
        occupied[:content[0] + guard + 1] = 1
        occupied[max(0, content[-1] - guard):] = 1
        gaps = []
        for a, b in spans(np.where(occupied == 0)[0]):
            candidates = np.where(~forbidden[a:b + 1])[0] + a
            if candidates.size:
                x = int(candidates[np.argmin(abs(candidates - (a + b) / 2))])
                gaps.append((a, b, x))
        if gaps:
            break
    if not gaps:
        return row, []
    weights = np.sqrt([b - a + 1 for a, b, _ in gaps])
    cumulative = np.rint(np.cumsum(weights / weights.sum()) * extra).astype(int)
    additions = np.diff(np.r_[0, cumulative])
    pieces, inserts, last = [], [], 0
    for (a, b, x), count in zip(gaps, additions):
        if count <= 0:
            continue
        pieces.append(row[:, last:x])
        column = np.full((row.shape[0], 1), 255, np.uint8)
        for y in lines:
            yy = round(y)
            lo, hi = max(0, yy - band), min(row.shape[0], yy + band + 1)
            column[lo:hi] = row[lo:hi, x:x + 1]
        pieces.append(np.repeat(column, int(count), axis=1))
        inserts.append({"source_x": int(x), "pixels": int(count)})
        last = x
    pieces.append(row[:, last:])
    return np.hstack(pieces), inserts
