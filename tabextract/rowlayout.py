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
            # A little compression is preferable to stretching an underfull
            # row. Strongly discourage packing beyond the readable allowance.
            penalty = (1 + 16 * max(0, 1 - ratio) ** 2
                       + 3 * max(0, ratio - 1) ** 2
                       + 120 * max(0, ratio - 1.18) ** 2)
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


def borrow_measure(units, crossings, groups, target_width, target_count):
    """Fill a short manual row with one more whole measure when it fits.

    Keep connected markings together and leave at least three measures (or
    the requested count for 1/2) in the following row, avoiding a sparse tail.
    """
    groups = list(groups)
    for i in range(len(groups) - 1):
        start, end = groups[i]
        next_start, next_end = groups[i + 1]
        if end != next_start or end - start >= target_count + 1:
            continue
        if next_end - end <= min(3, target_count):
            continue
        width = units[end - 1][3] - units[start][2]
        wider = units[end][3] - units[start][2]
        if (width < target_width * .9 and target_width <= wider <= target_width * 1.18
                and not crossings[end]):
            groups[i], groups[i + 1] = (start, end + 1), (end + 1, next_end)
    return groups


def compact_row(row, lines, target_width):
    """Remove only safe whitespace, preserving every non-string ink column.

    Keep generous space beside glyphs and retain at least 70% of each gap.
    Continuous annotations and small marks on strings prevent removal.
    """
    needed = row.shape[1] - target_width
    if needed <= 0:
        return row, []
    space = float(np.median(np.diff(lines)))
    ink = row < 245
    nonstaff = ink.copy()
    # Only ignore strings which are straight and have constant thickness
    # locally. A dot, angled stroke or stem on a string stays occupied.
    horizontal = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((1, max(5, round(space))), np.uint8)) > 0
    band = max(1, round(space * .25))
    for y in lines:
        yy = round(y)
        region = np.s_[max(0, yy - band):yy + band + 1]
        nonstaff[region] &= ~horizontal[region]
    occupied = nonstaff.any(axis=0).astype(np.uint8)
    guard = max(2, round(space * .6))
    occupied = cv2.dilate(occupied[None, :], np.ones((1, 2 * guard + 1), np.uint8))[0]
    ink_x = np.where(ink.any(axis=0))[0]
    if not ink_x.size:
        return row, []
    occupied[:ink_x[0] + guard + 1] = 1
    occupied[max(0, ink_x[-1] - guard):] = 1
    gaps = [(a, b) for a, b in spans(np.where(occupied == 0)[0])]
    capacity = np.array([int((b - a + 1) * .3) for a, b in gaps], dtype=int)
    available = int(capacity.sum())
    if not available:
        return row, []
    remove = min(needed, available)
    counts = np.diff(np.r_[0, np.rint(np.cumsum(capacity) * remove / available).astype(int)])
    keep = np.ones(row.shape[1], bool)
    removed = []
    for (a, b), count in zip(gaps, counts):
        if count:
            x = (a + b - int(count) + 1) // 2
            keep[x:x + count] = False
            removed.append({'source_x': int(x), 'pixels': int(count)})
    return row[:, keep], removed


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
