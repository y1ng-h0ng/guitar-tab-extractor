"""Video -> stable views -> temporal cleanup -> complete-measure rows -> PDF."""

import json
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from .region import detect_tab_region
from .pages import scan_segments, load_segment_frames
from .clean import clean_frames
from .geometry import staff_groups
from .stitch import assemble_score, stack_rows
from .pdfout import export_rows
from .support import check_cancel, validate_options, write_image


def inspect_video(video_path, region=None, cancel_event=None):
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"找不到视频文件: {video_path}")
    check_cancel(cancel_event)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError("无法打开视频")
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 5))
        ok, preview = cap.read()
        if not ok:
            raise OSError("无法读取视频预览")
    finally:
        cap.release()
    if region is None:
        x, y, w, h, confidence = detect_tab_region(str(video_path))
        if confidence <= 0:
            raise ValueError("无法自动定位六线谱，请使用“预览 / 框选谱面”手动框选")
        region = (x, y, w, h)
    else:
        confidence = 1.0
    if len(region) != 4 or any(not isinstance(v, (int, np.integer)) for v in region):
        raise ValueError("谱面区域需要四个整数 x,y,w,h")
    x, y, w, h = map(int, region)
    if x < 0 or y < 0 or w < 80 or h < 25 or x + w > width or y + h > height:
        raise ValueError(f"谱面框超出画面，或太小；视频尺寸 {width} × {height}")
    check_cancel(cancel_event)
    return (x, y, w, h), float(confidence), preview


def _save_json(path, value):
    path = Path(path)
    descriptor, tmp = tempfile.mkstemp(
        prefix=".tabextract-", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_pipeline(
    video_path,
    out_pdf,
    sample_fps=6.0,
    dpi=300,
    debug_dir=None,
    log=print,
    progress_cb=None,
    region=None,
    bars_per_row=4,
    polarity="auto",
    cancel_event=None,
):
    validate_options(sample_fps, dpi, bars_per_row)
    if polarity not in ("auto", "bright", "dark"):
        raise ValueError("谱面颜色模式无效")
    source, out = Path(video_path).resolve(), Path(out_pdf).resolve()
    if source == out:
        raise ValueError("输出文件不能覆盖输入视频")
    if out.suffix.lower() != ".pdf":
        raise ValueError("输出文件需要使用 .pdf 后缀")
    out.parent.mkdir(parents=True, exist_ok=True)
    debug = Path(debug_dir).resolve() if debug_dir else None
    if debug:
        debug.mkdir(parents=True, exist_ok=True)

    def progress(pct, msg, announce=True):
        check_cancel(cancel_event)
        if announce:
            log(msg)
        if progress_cb:
            progress_cb(float(pct), msg)

    progress(2, "[1/5] 定位谱面，保留上方标记与下方符干 ...")
    manual = region is not None
    region, confidence, preview = inspect_video(source, region, cancel_event)
    x, y, w, h = region
    progress(8, f"谱面区域: {region}；定位评分 {confidence:.2f}")
    if debug:
        annotated = preview.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 80, 255), 2)
        write_image(debug / "region_preview.png", annotated)
    progress(10, "[2/5] 按音符特征检测翻页 ...")
    segments, actual_polarity = scan_segments(
        source,
        region,
        sample_fps,
        polarity,
        log,
        progress=lambda p: progress(10 + 25 * p, "正在扫描视频 ...", False),
        cancel_event=cancel_event,
    )
    pages = []
    view_info = []
    skipped = []
    progress(36, "[3/5] 每页多帧合成，去除背景和播放指示线 ...")
    for i, segment in enumerate(segments):
        frames = load_segment_frames(source, region, segment, cancel_event=cancel_event)
        page = clean_frames(frames, actual_polarity, cancel_event=cancel_event)
        if not staff_groups(page):
            skipped.append({"start": segment.start, "end": segment.end})
            if debug:
                write_image(debug / f"skipped_{i + 1:02}_raw.png", frames[len(frames) // 2])
                write_image(debug / f"skipped_{i + 1:02}_clean.png", page)
            log(f"跳过 {segment.start:.2f}–{segment.end:.2f} 秒：未见有效六线谱")
            continue
        pages.append(page)
        info = {
            "page": len(pages),
            "start_seconds": segment.start,
            "end_seconds": segment.end,
            "samples": len(frames),
        }
        view_info.append(info)
        if debug:
            write_image(
                debug / f"page_{len(pages):02}_raw.png", frames[len(frames) // 2]
            )
            write_image(debug / f"page_{len(pages):02}_clean.png", page)
        progress(
            36 + 39 * (i + 1) / len(segments),
            f"谱面 {len(pages)}: {segment.start:.2f}–{segment.end:.2f} 秒",
        )
    if not pages:
        raise RuntimeError("没有提取到有效谱面，请检查框选区域与颜色模式")
    progress(77, "[4/5] 相邻页去重，保护连音换行并对齐行宽 ...")
    result = assemble_score(pages, bars_per_row, log, cancel_event)
    for item in result.joins:
        if "to_page" in item:
            item["time_seconds"] = view_info[item["to_page"] - 1]["start_seconds"]
    if skipped and any(
        s["start"] > view_info[0]["start_seconds"]
        and s["end"] < view_info[-1]["end_seconds"]
        for s in skipped
    ):
        result.warnings.append(
            "视频中间存在未提取的片段，详见报告的 skipped_segments，请对照原片检查。"
        )
    for warning in result.warnings:
        log("待核对: " + warning)
    full = stack_rows(result.rows)
    if debug:
        write_image(debug / "stitched_full.png", full)
        for i, row in enumerate(result.rows):
            write_image(debug / f"row_{i + 1:02}.png", row)
        for i, run in enumerate(result.runs):
            write_image(debug / f"run_{i + 1:02}.png", run)
    progress(90, "[5/5] 按 A4 排版，整行分页 ...")
    title = source.stem
    layouts = export_rows(
        result.rows,
        out,
        dpi,
        result.staff_spacing,
        title,
        result.warnings,
        cancel_event,
    )
    report_path = out.with_suffix(".report.json")
    summary = {
        "source_video": source.name,
        "region": list(region),
        "region_confidence": confidence,
        "manual_region": manual,
        "polarity": actual_polarity,
        "sample_fps": sample_fps,
        "dpi": dpi,
        "bars_per_row": bars_per_row,
        "video_pages": len(pages),
        "pdf_pages": len(layouts),
        "measures": result.measures,
        "complete_measure_units": result.complete_measures,
        "score_rows": len(result.rows),
        "stitched_size": list(full.shape),
        "output_pdf": str(out),
        "report_path": str(report_path),
        "warnings": result.warnings,
        "source_views": view_info,
        "skipped_segments": skipped,
        "joins": result.joins,
        "rows": result.row_info,
        "crossing_marks": result.crossing_marks,
        "layout_policy": "以目标小节数为主，跨小节连线处允许局部调整；在安全空白处扩展行宽，不拉伸音符。",
        "pdf_layout": layouts,
        "note": "小节数由图像边界推定，未进行音高或节奏的语义识别；源视频未显示的内容不会补写。",
    }
    _save_json(report_path, summary)
    progress(
        100,
        f"完成: {len(layouts)} 页 PDF，{len(result.rows)} 行谱表；待核对 {len(result.warnings)} 项",
    )
    return summary
