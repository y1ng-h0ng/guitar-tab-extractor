"""Video -> stable views -> temporal cleanup -> complete-measure rows -> PDF."""

import json
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from .region import detect_tab_region
from .pages import scan_segments, load_segment_frames
from .clean import clean_frames, infer_polarity
from .scroll import make_scroll_bridge
from .geometry import staff_groups
from .stitch import assemble_score, stack_rows
from .pdfout import export_rows
from .support import check_cancel, parse_bars_per_row, validate_options, write_image, time_bounds
from .review import build_issues
from .cache import fingerprint, digest


def inspect_video(video_path, region=None, cancel_event=None, time_range=None):
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
        position = max(0, total // 5)
        if time_range is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            start, end = time_bounds(time_range, total / fps if fps > 0 else 0)
            position = min(total-1, round((start+(end-start)*.2)*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, preview = cap.read()
        if not ok:
            raise OSError("无法读取视频预览")
    finally:
        cap.release()
    if region is None:
        x, y, w, h, confidence = detect_tab_region(str(video_path), time_range=time_range)
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
    video_path, out_pdf, sample_fps=6.0, dpi=300, debug_dir=None, log=print,
    progress_cb=None, region=None, bars_per_row=None, polarity='auto',
    generate_report=False, cancel_event=None, source_url=None, review_session=None,
    _edit=None, time_range=None, cache_store=None, source_options=None,
):
    options = dict(locals())
    options.pop('cache_store')
    time_range = time_bounds(time_range) if time_range is not None else None
    options['time_range'] = time_range
    if cache_store is None or _edit is not None:
        return _run_pipeline(**options)
    validate_options(sample_fps, dpi, parse_bars_per_row(bars_per_row))
    log('正在检查可恢复的缓存 ...')
    identity = dict(kind='extraction', version=1, source=fingerprint(video_path, cancel_event),
                    region=region, sample_fps=sample_fps, polarity=polarity, time_range=time_range)
    with cache_store.job(identity) as checkpoint:
        checkpoint.save('request', dict(source=source_url or str(Path(video_path).resolve()),
            out_pdf=str(Path(out_pdf).resolve()), sample_fps=sample_fps, dpi=dpi,
            region=region, bars_per_row=bars_per_row, polarity=polarity, time_range=time_range,
            source_options=source_options))
        summary = _run_pipeline(**options, _checkpoint=checkpoint)
        if review_session is not None:
            review_session.cache_store = cache_store
            review_session.cache_identity = identity
        return summary


def _run_pipeline(
    video_path,
    out_pdf,
    sample_fps=6.0,
    dpi=300,
    debug_dir=None,
    log=print,
    progress_cb=None,
    region=None,
    bars_per_row=None,
    polarity="auto",
    generate_report=False,
    cancel_event=None,
    source_url=None,
    review_session=None,
    _edit=None,
    time_range=None,
    _checkpoint=None,
    source_options=None,
):
    bars_per_row = parse_bars_per_row(bars_per_row)
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
    if _edit is None:
        located = _checkpoint.load('location') if _checkpoint else None
        if located is None:
            located = inspect_video(source, region, cancel_event, time_range=time_range)
            if _checkpoint:
                _checkpoint.save('location', located)
        else:
            log('复用已保存的谱面定位。')
        region, confidence, preview = located
    else:
        region = tuple(review_session.summary["region"])
        confidence = review_session.summary["region_confidence"]
        manual = review_session.summary["manual_region"]
        debug = None
    x, y, w, h = region
    progress(8, f"谱面区域: {region}；定位评分 {confidence:.2f}")
    if debug:
        annotated = preview.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 80, 255), 2)
        write_image(debug / "region_preview.png", annotated)
    progress(10, "[2/5] 按音符特征检测翻页 ...")
    if _edit is None:
        scanned = _checkpoint.load('scan') if _checkpoint else None
        if scanned is None:
            scanned = scan_segments(source, region, sample_fps, polarity, log,
                progress=lambda p: progress(10 + 25 * p, '正在扫描视频 ...', False),
                cancel_event=cancel_event, time_range=time_range, checkpoint=_checkpoint)
            if _checkpoint:
                _checkpoint.save('scan', scanned)
        else:
            log('复用已完成的翻页扫描。')
        segments, actual_polarity = scanned
    else:
        segments = [r["original"] for r in review_session.records]
        actual_polarity = review_session.summary["polarity"]
        log("局部重处理：复用其他片段，仅更新所选片段与相邻接缝。")
    pages = []
    records = []
    bridge_cache = dict(review_session.bridges) if _edit else {}
    view_info = []
    skipped = []
    overrides = _checkpoint.load('overrides', {}) if _checkpoint else {}
    progress(36, "[3/5] 每页多帧合成，去除背景和播放指示线 ...")
    for i, segment in enumerate(segments):
        check_cancel(cancel_event)
        cached_record = (overrides.get(i+1) or _checkpoint.load(f'record-{i+1}')) if _checkpoint else None
        if _edit is not None and i+1 != _edit["id"]:
            record = review_session.records[i]
        elif cached_record is not None:
            record = cached_record
            log(f'复用片段 {i+1} 的提取结果。')
        else:
            original = segment
            selected_polarity = actual_polarity if _edit is None else _edit["polarity"]
            gain = 1.0 if _edit is None else _edit["detail_gain"]
            segment = segment if _edit is None else _edit["segment"]
            frames = load_segment_frames(source, region, segment, cancel_event=cancel_event)
            effective_polarity = selected_polarity
            if effective_polarity == "auto":
                votes = [infer_polarity(frames[j]) for j in (0, len(frames)//2, len(frames)-1)]
                effective_polarity = max(set(votes), key=votes.count)
            diagnostics = []
            page = clean_frames(frames, effective_polarity, cancel_event=cancel_event,
                                detail_gain=gain, diagnostics=diagnostics)
            record = {"id": i+1, "original": original, "segment": segment,
                      "page": page, "raw": frames[len(frames)//2], "samples": len(frames),
                      "polarity": selected_polarity, "detail_gain": gain,
                      "effective_polarity": effective_polarity,
                      "revision": 0 if _edit is None else _edit["revision"],
                      "diagnostics": diagnostics, "valid": bool(staff_groups(page))}
            if _edit is not None and not record["valid"]:
                raise RuntimeError("此局部设置未提取到有效六线谱，原片段和 PDF 已保留；请调整采样时间或颜色模式")
            if _checkpoint:
                _checkpoint.save(f'record-{i+1}', record)
        records.append(record)
        segment, page = record["segment"], record["page"]
        if not record["valid"]:
            skipped.append({"start": segment.start, "end": segment.end, "segment_id": i+1})
            if debug:
                write_image(debug / f"skipped_{i + 1:02}_raw.png", record["raw"])
                write_image(debug / f"skipped_{i + 1:02}_clean.png", page)
            log(f"跳过 {segment.start:.2f}–{segment.end:.2f} 秒：未见有效六线谱")
            continue
        pages.append(page)
        info = {
            "page": len(pages),
            "start_seconds": segment.start,
            "end_seconds": segment.end,
            "samples": record["samples"],
            "segment_id": record["id"],
            "detail_gain": record["detail_gain"],
            "polarity": record["polarity"],
        }
        view_info.append(info)
        if debug:
            write_image(
                debug / f"page_{len(pages):02}_raw.png", record["raw"]
            )
            write_image(debug / f"page_{len(pages):02}_clean.png", page)
        progress(
            36 + 39 * (i + 1) / len(segments),
            f"谱面 {len(pages)}: {segment.start:.2f}–{segment.end:.2f} 秒",
        )
    if not pages:
        raise RuntimeError("没有提取到有效谱面，请检查框选区域与颜色模式")
    progress(77, "[4/5] 相邻页去重，保护连音换行并对齐行宽 ...")

    def bridge_provider(index, previous, current):
        kept = [r for r in records if r["valid"]]
        a, b = kept[index-1], kept[index]
        key = (a["id"], a["revision"], b["id"], b["revision"])
        cache_name = 'bridge-' + digest(key)
        saved = _checkpoint.load(cache_name) if _checkpoint else None
        if saved is not None:
            bridge_cache[key] = saved['value']
        if key in bridge_cache:
            return bridge_cache[key]
        progress(77 + 11 * index / max(1, len(pages) - 1),
                 f"补采页面 {index} → {index + 1} 的翻页过程 ...")
        # Keep the original transition interval. Reclean only this bridge with
        # the adjoining views' mean gain, then verify BOTH seams as usual.
        # Opposite foreground polarities cannot share a single aligned bridge.
        pa = a.get("effective_polarity", a["polarity"])
        pb = b.get("effective_polarity", b["polarity"])
        bridge = None if pa != pb else make_scroll_bridge(
            source, region, a["original"], b["original"],
            pa, sample_fps, cancel_event,
            detail_gain=(a["detail_gain"]+b["detail_gain"])/2,
        )
        bridge_cache[key] = bridge
        if _checkpoint:
            _checkpoint.save(cache_name, {'value': bridge})
        if debug and bridge is not None:
            write_image(debug / f"scroll_{index:02}_clean.png", bridge["page"])
        return bridge

    log("自动分行：按小节宽度与连线安排每行数量，并统一行宽。" if bars_per_row is None
        else f"自定义分行：每行目标 {bars_per_row} 小节，并统一行宽。")
    assembly_key = 'assembly-' + digest([(r['id'], r['revision']) for r in records])
    result = assemble_score(pages, bars_per_row, log, cancel_event, bridge_provider,
        preassembled=_checkpoint.load(assembly_key) if _checkpoint else None,
        save_assembly=(lambda value: _checkpoint.save(assembly_key, value)) if _checkpoint else None)
    issues = build_issues(records, result.joins, result.row_info)
    for item in result.joins:
        if "to_page" in item:
            item["time_seconds"] = view_info[item["to_page"] - 1]["start_seconds"]
    if skipped and any(
        s["start"] > view_info[0]["start_seconds"]
        and s["end"] < view_info[-1]["end_seconds"]
        for s in skipped
    ):
        result.warnings.append("视频中间存在未提取的片段，请对照原片检查。")
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
    report_path = out.with_suffix(".report.json") if generate_report else None
    summary = {
        "source_video": source.name,
        "source_url": source_url,
        "time_range": time_range,
        "region": list(region),
        "region_confidence": confidence,
        "manual_region": manual,
        "polarity": actual_polarity,
        "sample_fps": sample_fps,
        "dpi": dpi,
        "bars_per_row": bars_per_row,
        "row_layout_mode": "auto" if bars_per_row is None else "manual",
        "video_pages": len(pages),
        "pdf_pages": len(layouts),
        "measures": result.measures,
        "complete_measure_units": result.complete_measures,
        "score_rows": len(result.rows),
        "stitched_size": list(full.shape),
        "output_pdf": str(out),
        "report_path": str(report_path) if report_path else None,
        "warnings": result.warnings,
        "issues": issues,
        "segment_settings": [{"segment_id": r["id"], "start_seconds": r["segment"].start,
                              "end_seconds": r["segment"].end, "polarity": r["polarity"],
                              "detail_gain": r["detail_gain"], "included": r["valid"]} for r in records],
        "source_views": view_info,
        "skipped_segments": skipped,
        "joins": result.joins,
        "scroll_joins": sum(j["status"] == "joined_scroll" for j in result.joins),
        "rows": result.row_info,
        "crossing_marks": result.crossing_marks,
        "layout_policy": "按小节宽度和连线分行；短行优先尝试多放完整小节后适度压缩，自定义时仍以目标数量为主。各行含末行统一总宽；过宽行先收紧安全空白，必要时整行横向缩放，总压缩不超过15%；较短行补充空白对齐。",
        "pdf_layout": layouts,
        "note": "小节数由图像边界推定，未进行音高或节奏的语义识别；源视频未显示的内容不会补写。",
    }
    if report_path:
        _save_json(report_path, summary)
    if review_session is not None:
        if _checkpoint:
            kept = [r for r in records if r['valid']]
            for a, b in zip(kept, kept[1:]):
                key = (a['id'], a['revision'], b['id'], b['revision'])
                saved = _checkpoint.load('bridge-' + digest(key))
                if saved is not None:
                    bridge_cache[key] = saved['value']
        review_session.records = records
        review_session.bridges = bridge_cache
        review_session.rows = result.rows
        review_session.assembly = result.stitch_state
        review_session.summary = summary
        if _edit is None:
            review_session.config = dict(video_path=str(source), out_pdf=str(out),
                sample_fps=sample_fps, dpi=dpi, region=list(region), bars_per_row=bars_per_row,
                polarity=actual_polarity, generate_report=generate_report, source_url=source_url,
                time_range=time_range)
    progress(
        100,
        f"完成: {len(layouts)} 页 PDF，{len(result.rows)} 行谱表；待核对 {len(result.warnings)} 项",
    )
    return summary
