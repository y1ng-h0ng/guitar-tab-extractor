"""Ordered overlap matching and complete-measure reflow.
Only adjacent source views are compared. A musical repetition elsewhere in the
song is never a reason to delete a measure. No low-ink fallback cuts a measure.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np
from .geometry import staff_groups, bar_lines, trim_vertical
from .support import check_cancel
from .rowlayout import crossing_marks, choose_breaks, justify_row


@dataclass
class Assembly:
    rows: list = field(default_factory=list)
    joins: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    row_info: list = field(default_factory=list)
    measures: int = 0
    complete_measures: int = 0
    staff_spacing: float = 21.0
    runs: list = field(default_factory=list)
    crossing_marks: list = field(default_factory=list)


def _feature(page, lines):
    space = float(np.median(np.diff(lines)))
    feat = (255 - page).astype(np.float32) / 255
    top, bottom = round(lines[0]), round(lines[-1])
    feat[: max(0, round(top - space * 2))] = 0
    feat[min(page.shape[0], round(bottom + space * 3.5)) :] = 0
    for line in lines:
        y = round(line)
        r = max(1, round(space * 0.13))
        feat[max(0, y - r) : y + r + 1] = 0
    a, b = max(0, round(top - space)), max(0, round(top - space * 0.2))
    feat[a:b] *= 2
    return cv2.GaussianBlur(feat, (3, 3), 0.7)


def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


def find_horizontal_shift(a, b, lines):
    if a.shape != b.shape:
        return None
    fa, fb = _feature(a, lines), _feature(b, lines)
    same = _cosine(fa, fb)
    if same > 0.998:
        return {"shift": 0, "score": same, "overlap_score": same, "duplicate": True}
    width = a.shape[1]
    factor = max(1, round(width / 1280))
    sa = cv2.resize(fa, (width // factor, a.shape[0] // factor), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(fb, sa.shape[::-1], interpolation=cv2.INTER_AREA)
    tw = max(50, round(sa.shape[1] * 0.23))
    response = cv2.matchTemplate(sa, sb[:, :tw], cv2.TM_CCOEFF_NORMED)[0]
    work = response.copy()
    work[: max(3, round(len(work) * 0.03))] = -1
    candidates = []
    for _ in range(6):
        pos = int(np.argmax(work))
        score = float(work[pos])
        if score < 0.55:
            break
        shift = pos * factor
        best = None
        for dx in range(max(1, shift - factor), min(width - tw * factor, shift + factor) + 1):
            ov = min(width - dx, b.shape[1])
            match = _cosine(fa[:, dx : dx + ov], fb[:, :ov])
            if best is None or match > best[1]:
                best = (dx, match)
        if best and best[1] > 0.72:
            candidates.append({"shift": best[0], "score": score, "overlap_score": best[1], "duplicate": False})
        win = max(12, round(sa.shape[1] * 0.07))
        work[max(0, pos - win) : pos + win + 1] = -1
    if not candidates:
        return None
    candidates.sort(key=lambda d: d["overlap_score"], reverse=True)
    best = candidates[0]
    if best["score"] < 0.70 or best["overlap_score"] < 0.78:
        return None
    if len(candidates) > 1 and best["overlap_score"] - candidates[1]["overlap_score"] < 0.025:
        return None
    return best


def _horizontal_runs(pages, result, log, cancel_event):
    canvas = pages[0].copy()
    previous = pages[0]
    offset = 0
    lines = staff_groups(previous)[0]
    for i, current in enumerate(pages[1:], 1):
        check_cancel(cancel_event)
        match = find_horizontal_shift(previous, current, lines)
        if match is None:
            result.warnings.append(f"源页面 {i} → {i + 1} 的重叠未能可靠确认，已分别保留；请核对接缝。")
            result.joins.append({"from_page": i, "to_page": i + 1, "status": "unresolved"})
            result.runs.append(canvas)
            canvas = current.copy()
            offset = 0
        elif match["duplicate"]:
            result.joins.append({"from_page": i, "to_page": i + 1, "status": "same_view", **match})
            log(f"页面 {i} → {i + 1}：同一画面，无新增谱面")
            previous = current
            continue
        else:
            dx = match["shift"]
            space = float(np.median(np.diff(lines)))
            aa = bar_lines(previous, lines)
            bb = bar_lines(current, lines)
            pairs = [(x, y) for x in aa for y in bb if abs(x - (y + dx)) <= max(3, space * 0.2) and dx + space < x < previous.shape[1] - space]
            if not pairs:
                result.warnings.append(f"源页面 {i} → {i + 1} 虽有像素重叠，但未确认共同小节线，已分别保留。")
                result.joins.append({"from_page": i, "to_page": i + 1, "status": "unresolved_bar", **match})
                result.runs.append(canvas)
                canvas = current.copy()
                offset = 0
            else:
                center = (dx + previous.shape[1]) / 2
                x, y = min(pairs, key=lambda pair: abs(pair[0] - center))
                guard = max(2, round(space * 0.16))
                ca = max(0, x - guard)
                cb = max(0, y - guard)
                end = offset + ca
                canvas = np.hstack([canvas[:, :end], current[:, cb:]])
                offset = end - cb
                result.joins.append({"from_page": i, "to_page": i + 1, "status": "joined", "seam_previous": ca, "seam_current": cb, **match})
                log(f"页面 {i} → {i + 1}：重叠 {previous.shape[1] - dx} 像素，整小节接缝，匹配 {match['overlap_score']:.3f}")
        previous = current
        groups = staff_groups(previous)
        if groups:
            lines = groups[0]
    result.runs.append(canvas)


def _system_crops(page):
    groups = staff_groups(page)
    if len(groups) <= 1:
        return [page]
    cuts = [0]
    for a, b in zip(groups, groups[1:]):
        lo, hi = round(a[-1] + np.median(np.diff(a))), round(b[0] - np.median(np.diff(b)))
        blank = np.where((page[lo:hi] < 240).sum(axis=1) == 0)[0]
        if not blank.size:
            raise ValueError("相邻谱表的符号区没有安全空隙，请分别框选谱表后处理")
        cuts.append(lo + int(blank[len(blank) // 2]))
    cuts.append(page.shape[0])
    return [page[a:b] for a, b in zip(cuts, cuts[1:])]


def _system_signature(system):
    lines = staff_groups(system)[0]
    s = float(np.median(np.diff(lines)))
    feat = _feature(system, lines)
    a, b = max(0, round(lines[0] - s * 2)), min(system.shape[0], round(lines[-1] + s * 3))
    return cv2.resize(feat[a:b], (640, 100), interpolation=cv2.INTER_AREA)


def _vertical_runs(pages, result, log):
    previous = []
    for i, page in enumerate(pages):
        systems = _system_crops(page)
        sigs = [_system_signature(s) for s in systems]
        matched = 0
        if previous:
            for count in range(min(len(previous), len(sigs)), 0, -1):
                if all(_cosine(a, b) > 0.93 for a, b in zip(previous[-count:], sigs[:count])):
                    matched = count
                    break
            result.joins.append({"from_page": i, "to_page": i + 1, "status": "systems", "overlap_systems": matched})
        result.runs.extend(systems[matched:])
        previous = sigs
        log(f"页面 {i + 1}：{len(systems)} 个谱表，页首重叠 {matched} 个")


def _measure_units(image, result, run_index):
    groups = staff_groups(image)
    if len(groups) != 1:
        result.warnings.append(f"片段 {run_index + 1} 无法可靠分小节，作为整行保留。")
        return [(image, False, 0, image.shape[1], None, None)]
    lines = groups[0]
    space = float(np.median(np.diff(lines)))
    bars = bar_lines(image, lines)
    result.staff_spacing = space
    if len(bars) < 2:
        result.warnings.append(f"片段 {run_index + 1} 的小节线不足，作为整行保留。")
        return [(image, False, 0, image.shape[1], None, None)]
    guard = max(2, round(space * 0.16))
    units = []
    for i, (a, b) in enumerate(zip(bars, bars[1:])):
        left = 0 if i == 0 else max(0, a - guard)
        units.append((image[:, left : min(image.shape[1], b + guard + 1)], True, left, min(image.shape[1], b + guard + 1), a, b))
    tail = image[:, max(0, bars[-1] - guard) :]
    far = min(tail.shape[1], guard + round(space * 2))
    row_support = []
    for y in lines:
        yy = round(y)
        row_support.append(int((tail[max(0, yy - 1) : yy + 2, far:] < 180).any(axis=0).sum()))
    if row_support and min(row_support) > space * 2:
        units.append((tail, False, max(0, bars[-1] - guard), image.shape[1], bars[-1], None))
        result.warnings.append(f"片段 {run_index + 1} 的末尾小节右侧边界未在画面内确认，保留可见部分，请对照原视频。")
    return units


def assemble_score(pages, bars_per_row=4, log=print, cancel_event=None):
    if not pages:
        raise ValueError("没有可拼接的谱面")
    result = Assembly()
    if all(len(staff_groups(p)) == 1 for p in pages):
        _horizontal_runs(pages, result, log, cancel_event)
    else:
        _vertical_runs(pages, result, log)
    for run_index, run in enumerate(result.runs):
        check_cancel(cancel_event)
        units = _measure_units(run, result, run_index)
        lines = staff_groups(run)
        crossings = crossing_marks(run, lines[0], [u[5] for u in units]) if lines else [False] * len(units)
        breaks = choose_breaks(units, crossings, bars_per_row)
        line_ends = {end for _, end in breaks}
        first_unit = result.measures
        for n, crossing in enumerate(crossings):
            if crossing:
                kept = n + 1 not in line_ends
                result.crossing_marks.append({"run": run_index + 1, "after_unit": first_unit + n + 1, "kept_on_same_row": kept})
                if not kept:
                    result.warnings.append(f"第 {first_unit + n + 1} 小节后的跨小节连线无法在当前行宽内同行，建议调整每行目标小节数并核对。")
        for start, end in breaks:
            group = units[start:end]
            left, right = group[0][2], group[-1][3]
            side = round(result.staff_spacing * 1.2)
            row = np.pad(run[:, left:right], ((0, 0), (side, side)), constant_values=255)
            if lines:
                top = lines[0][0]
                space = result.staff_spacing
                label_top = max(0, round(top - space * 1.15))
                label_bottom = max(label_top + 1, round(top - space * 0.16))
                for boundary, is_left in [(group[0][4], True), (group[-1][5], False)]:
                    if boundary is None:
                        continue
                    a = max(0, boundary - side)
                    b = min(run.shape[1], boundary + round(space * 1.8))
                    patch = run[:label_bottom, a:b]
                    n, labels, stats, _ = cv2.connectedComponentsWithStats((patch < 245).astype(np.uint8), 8)
                    for k in range(1, n):
                        x, y, w, h, area = stats[k]
                        if w > space * 2 or h > space * 2:
                            continue
                        center = a + x + w / 2
                        is_number = y >= label_top - space * 0.15 and h >= space * 0.2 and boundary - space * 0.45 <= center <= boundary + space * 1.4
                        belongs_right = is_number or center >= boundary
                        keep = belongs_right if is_left else not belongs_right
                        yy, xx = np.where(labels == k)
                        gx = a + xx - left + side
                        valid = (gx >= 0) & (gx < row.shape[1])
                        if keep:
                            row[yy[valid], gx[valid]] = patch[yy[valid], xx[valid]]
                        else:
                            row[yy[valid], gx[valid]] = 255
            row = trim_vertical(row, margin=12)
            count = len(group)
            complete = sum(int(unit[1]) for unit in group)
            result.row_info.append({"run": run_index + 1, "first_unit": result.measures + 1, "units": count, "complete": complete, "width": row.shape[1], "height": row.shape[0], "target_units": bars_per_row, "end_of_run": end == len(units), "break_after_connected_mark": bool(crossings[end - 1])})
            result.rows.append(row)
            result.measures += count
            result.complete_measures += complete
    target_width = max(row.shape[1] for row in result.rows)
    for index, (row, info) in enumerate(zip(result.rows, result.row_info)):
        info["natural_width"] = row.shape[1]
        info["space_insertions"] = []
        if info["end_of_run"] and info["units"] < bars_per_row:
            continue
        groups = staff_groups(row)
        if groups:
            expanded, inserts = justify_row(row, groups[0], target_width)
            result.rows[index] = expanded
            info["width"] = expanded.shape[1]
            info["space_insertions"] = inserts
            if expanded.shape[1] < target_width * 0.95:
                result.warnings.append(f"第 {index + 1} 行没有足够的安全空隙用于行宽对齐，保留原始符号间距。")
    return result


def stack_rows(rows, gap=36):
    width = max(r.shape[1] for r in rows)
    parts = []
    for i, row in enumerate(rows):
        if i:
            parts.append(np.full((gap, width), 255, np.uint8))
        parts.append(np.pad(row, ((0, 0), (0, width - row.shape[1])), constant_values=255))
    return np.vstack(parts)


def stitch_pages(pages, log=print):
    return stack_rows(assemble_score(pages, log=log).rows)
