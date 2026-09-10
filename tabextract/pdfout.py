"""Fixed A4 pages with margins. A score row is an indivisible layout unit."""

import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas
from .geometry import staff_groups, bar_lines
from .support import check_cancel


def _add_final_barline(row):
    """Turn the last confirmed measure boundary into a thin+thick final barline.

    If the six staff lines continue substantially past the last detected bar,
    the source likely ends with a partially visible measure; in that case the
    image is left unchanged rather than inventing a musical ending.
    """
    page = row.copy()
    groups = staff_groups(page)
    if len(groups) != 1:
        return page

    lines = groups[0]
    bars = bar_lines(page, lines)
    if not bars:
        return page

    space = float(np.median(np.diff(lines)))
    x = int(bars[-1])

    # A complete final measure should stop at its right barline. If all six
    # strings visibly continue well beyond it, this is probably the left edge
    # of an incomplete tail and must not be decorated as a final boundary.
    probe = min(page.shape[1], x + max(2, round(space * 0.35)))
    support = []
    for y in lines:
        yy = int(round(y))
        strip = page[max(0, yy - 1) : min(page.shape[0], yy + 2), probe:]
        support.append(int((strip < 180).any(axis=0).sum()) if strip.size else 0)
    if support and min(support) > space * 1.5:
        return page

    thin = max(1, round(space * 0.10))
    gap = max(2, round(space * 0.20))
    thick = max(2, round(space * 0.30))
    line_band = max(1, round(space * 0.06))

    thin_left = max(0, x - thin // 2)
    thin_right = thin_left + thin
    thick_left = thin_right + gap
    thick_right = thick_left + thick
    padding = max(2, round(space * 0.35))
    required_width = thick_right + padding
    if required_width > page.shape[1]:
        page = np.pad(
            page,
            ((0, 0), (0, required_width - page.shape[1])),
            constant_values=255,
        )

    top = max(0, int(round(lines[0])))
    bottom = min(page.shape[0] - 1, int(round(lines[-1])))

    # Redraw the existing final boundary as a clear thin line.
    page[top : bottom + 1, thin_left:thin_right] = 0

    # Continue the six strings through the small gap up to the heavy bar.
    for y in lines:
        yy = int(round(y))
        a = max(0, yy - line_band)
        b = min(page.shape[0], yy + line_band + 1)
        page[a:b, x:thick_right] = 0

    # Heavy line on the right: conventional final-barline appearance.
    page[top : bottom + 1, thick_left:thick_right] = 0
    return page


def layout_rows(rows, staff_spacing=21.0):
    page_w, page_h = A4
    margin = 42.52
    content_w = page_w - 2 * margin
    scale = min(6.5 / staff_spacing, content_w / max(row.shape[1] for row in rows))
    heights = [row.shape[0] * scale for row in rows]
    if any(height > page_h - 155 for height in heights):
        raise ValueError("谱表过高，无法在一页内清晰放置；请重新框选谱面")
    page_count, used, count = 1, 0, 0
    for height in heights:
        capacity = page_h - (150 if page_count == 1 else 130)
        addition = height + (24 if count else 0)
        if count and used + addition > capacity:
            page_count += 1
            used, count, addition = 0, 0, height
        used += addition
        count += 1
    total = len(rows)
    prefix = np.r_[0, np.cumsum(heights)]
    cost = {(0, 0): 0.0}
    previous = {}
    for p in range(1, page_count + 1):
        capacity = page_h - (150 if p == 1 else 130)
        for end in range(p, total + 1):
            for start in range(p - 1, end):
                before = cost.get((p - 1, start))
                if before is None:
                    continue
                n = end - start
                height = prefix[end] - prefix[start] + 24 * (n - 1)
                if height > capacity:
                    continue
                value = before + (n - total / page_count) ** 2 + 0.1 * (height / capacity) ** 2
                if value < cost.get((p, end), float("inf")):
                    cost[p, end] = value
                    previous[p, end] = start
    groups, end = [], total
    for p in range(page_count, 0, -1):
        start = previous[p, end]
        groups.append((start, end))
        end = start
    pages = []
    for p, (start, end) in enumerate(reversed(groups)):
        top = page_h - (85 if p == 0 else 65)
        spare = top - 65 - (prefix[end] - prefix[start])
        gap = max(24, min(40, spare / max(1, end - start - 1)))
        placed = []
        for index in range(start, end):
            height = heights[index]
            placed.append({"row": index, "x": margin, "y": top - height, "width": rows[index].shape[1] * scale, "height": height})
            top -= height + gap
        pages.append(placed)
    return pages


def export_rows(rows, out_path, dpi=300, staff_spacing=21.0, title="吉他谱", warnings=None, cancel_event=None):
    if not rows:
        raise ValueError("没有可导出的谱表")
    if not 72 <= dpi <= 600:
        raise ValueError("DPI 必须在 72 到 600 之间")
    rows = list(rows)
    rows[-1] = _add_final_barline(rows[-1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layouts = layout_rows(rows, staff_spacing)
    font_path = Path(__file__).resolve().parents[1] / "assets" / "NotoSansSC-Regular.ttf"
    font_name = "TabCJK"
    if font_path.is_file():
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    else:
        font_name = "STSong-Light"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    descriptor, tmp = tempfile.mkstemp(prefix=".tabextract-", suffix=".pdf", dir=str(out_path.parent))
    os.close(descriptor)
    try:
        canvas = Canvas(tmp, pagesize=A4, pageCompression=1)
        canvas.setTitle(title)
        canvas.setAuthor("Guitar Tab Extractor")
        page_w, page_h = A4
        for number, placements in enumerate(layouts, 1):
            check_cancel(cancel_event)
            canvas.setFillColorRGB(0.12, 0.15, 0.18)
            canvas.setFont(font_name, 17 if number == 1 else 10)
            canvas.drawString(42.52, page_h - (43 if number == 1 else 33), title)
            if number == 1:
                canvas.setFont(font_name, 9)
                canvas.setFillColorRGB(0.4, 0.43, 0.45)
                canvas.drawString(42.52, page_h - 61, "吉他六线谱")
            for item in placements:
                check_cancel(cancel_event)
                raster = Image.fromarray(rows[item["row"]]).convert("L")
                target = (max(1, round(item["width"] / 72 * dpi)), max(1, round(item["height"] / 72 * dpi)))
                if raster.size != target:
                    raster = raster.resize(target, Image.Resampling.LANCZOS)
                canvas.drawImage(ImageReader(raster), item["x"], item["y"], item["width"], item["height"], mask=None)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColorRGB(0.45, 0.45, 0.45)
            canvas.drawRightString(page_w - 42.52, 27, f"{number} / {len(layouts)}")
            if warnings:
                canvas.setFont(font_name, 8)
                note = "有待核对内容，详见提取报告。"
                if any("末尾" in w for w in warnings):
                    note = "末尾小节右侧边界未完整入镜，请对照原视频。"
                canvas.drawString(42.52, 27, note)
            canvas.showPage()
        canvas.save()
        check_cancel(cancel_event)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return layouts


def export_pdf(image, out_path, dpi=300, **_compat):
    layouts = export_rows([image], out_path, dpi=dpi)
    return len(layouts), [image.shape[:2]]
