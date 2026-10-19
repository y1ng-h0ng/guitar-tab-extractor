"""Session-local review and transactional reprocessing of one stable segment."""

import copy
import math
import os
import shutil
import tempfile
from pathlib import Path

from .pages import Segment
from .support import check_cancel
from .cache import digest, assembly_name
from .regions import validate_region, apply_regions
from .geometry import staff_groups


class ReviewSession:
    """Active review, with optional persistent checkpoints after successful edits."""

    def __init__(self):
        self.records = []
        self.bridges = {}
        self.config = {}
        self.summary = None
        self.rows = []
        self.cache_store = None
        self.cache_identity = None
        self.assembly = {}

    def close(self):
        self.records.clear()
        self.bridges.clear()
        self.config.clear()
        self.rows.clear()
        self.summary = None
        self.cache_store = self.cache_identity = None
        self.assembly = {}

    def reprocess(self, segment_id, start, end, polarity, detail_gain,
                  log=print, progress_cb=None, cancel_event=None):
        if not self.summary:
            raise ValueError("请先生成谱面")
        record = next((r for r in self.records if r["id"] == segment_id), None)
        if record is None:
            raise ValueError("片段不存在")
        original = record["original"]
        if (not all(math.isfinite(v) for v in (start, end, detail_gain))
                or start < original.start - .001 or end > original.end + .001
                or end - start < min(.2, original.end-original.start)):
            raise ValueError("采样时间必须位于此片段原始范围内，且至少保留 0.2 秒（短片段请保留完整范围）")
        if polarity not in ("auto", "bright", "dark") or not .5 <= detail_gain <= 2:
            raise ValueError("颜色模式或细节系数无效；系数范围为 0.5–2.0")
        segment = Segment(max(original.first, round(start*original.fps)),
                          min(original.last, round(end*original.fps)), original.fps)
        if segment.last <= segment.first:
            raise ValueError("采样范围过短")
        edit = {"id": segment_id, "segment": segment, "polarity": polarity,
                "detail_gain": float(detail_gain), "revision": record["revision"]+1}
        return self._publish(edit, log, progress_cb, cancel_event)

    def edit_regions(self, segment_id, kind=None, bbox=None, undo=False,
                     log=print, progress_cb=None, cancel_event=None):
        if not self.summary:
            raise ValueError('请先生成谱面')
        original = next((r for r in self.records if r['id'] == segment_id), None)
        if original is None:
            raise ValueError('片段不存在')
        operations = list(original.get('regions', []))
        if undo:
            if not operations: raise ValueError('此片段没有可撤销的框选')
            operations.pop()
        else:
            if len(operations) >= 100:
                raise ValueError('每个片段最多保存 100 个框选，请先撤销不需要的框选')
            operations.append(validate_region(kind, bbox, original['page'].shape))
        record = dict(original, regions=operations, revision=original['revision']+1)
        record['base_page'] = original.get('base_page', original['page'])
        record['page'] = apply_regions(record['base_page'], [(record, 0, 0)])
        if not staff_groups(record['page']):
            raise ValueError('此框选会破坏六线谱结构，请缩小范围；原结果已保留')
        edit = {'id': segment_id, 'revision': record['revision'], 'replacement': record}
        return self._publish(edit, log, progress_cb, cancel_event)

    def _publish(self, edit, log, progress_cb, cancel_event):
        from .pipeline import run_pipeline, _save_json
        segment_id = edit['id']
        candidate = copy.copy(self)
        candidate.records = list(self.records)
        if 'replacement' in edit:
            # Drawing changes no source evidence or geometry; reuse transitions.
            candidate.bridges = {(a, edit['revision'] if a == segment_id else ar,
                                  b, edit['revision'] if b == segment_id else br): value
                                 for (a, ar, b, br), value in self.bridges.items()}
        else:
            candidate.bridges = {key: value for key, value in self.bridges.items()
                                 if segment_id not in (key[0], key[2])}
        config = dict(self.config)
        out = Path(config.pop("out_pdf"))
        report = bool(config.get("generate_report"))
        # Render into a sibling staging directory. Neither the current PDF nor
        # cached pages change on cancellation, extraction or export failure.
        with tempfile.TemporaryDirectory(prefix=".tabextract-review-", dir=out.parent) as tmp:
            staged = Path(tmp)/out.name
            summary = run_pipeline(out_pdf=staged, **config, review_session=candidate,
                                   _edit=edit, log=log, progress_cb=progress_cb,
                                   cancel_event=cancel_event)
            summary["output_pdf"] = str(out)
            summary["report_path"] = str(out.with_suffix(".report.json")) if report else None
            targets = [(staged, out)]
            if report:
                _save_json(staged.with_suffix(".report.json"), summary)
                targets.append((staged.with_suffix(".report.json"), out.with_suffix(".report.json")))
            check_cancel(cancel_event)
            backups, published = {}, []
            try:
                for index, (_, target) in enumerate(targets):
                    backup = Path(tmp)/f"backup-{index}"
                    if target.exists():
                        shutil.copy2(target, backup)
                        backups[target] = backup
                for source, target in targets:
                    os.replace(source, target)
                    published.append(target)
            except Exception:
                for target in reversed(published):
                    if target in backups:
                        os.replace(backups[target], target)
                    else:
                        target.unlink(missing_ok=True)
                raise
        self.records, self.bridges = candidate.records, candidate.bridges
        self.rows, self.summary = candidate.rows, summary
        self.assembly = candidate.assembly
        if self.cache_store and self.cache_identity:
            try:
                with self.cache_store.job(self.cache_identity) as checkpoint:
                    checkpoint.save('overrides', {r['id']: r for r in self.records if r['revision']})
                    for key, bridge in self.bridges.items():
                        checkpoint.save('bridge-' + digest(key), {'value': bridge})
                    checkpoint.save(assembly_name(self.records),
                                    self.assembly)
            except (OSError, RuntimeError, ValueError) as error:
                log(f'PDF 已更新，但局部修正缓存保存失败：{error}。本次会话仍保留修正。')
        return summary


def build_issues(records, joins, row_info=()):
    """Coordinates refer to extracted source views, not inferred musical notes."""
    issues = []
    kept = [r for r in records if r["valid"]]

    def add(record, kind, message, bbox=None, time=None):
        h, w = record["page"].shape
        issues.append({"segment_id": record["id"], "kind": kind, "message": message,
                       "time_seconds": float(time if time is not None else
                                             (record["segment"].start+record["segment"].end)/2),
                       "bbox": bbox or [0, 0, w, h], "image_size": [w, h]})

    for record in records:
        if not record["valid"]:
            add(record, "missing_view", "此片段未提取到六线谱，可能缺少谱面")
        for detail in record.get("diagnostics", []):
            add(record, detail["kind"], detail["message"], detail["bbox"])
    for join in joins:
        a, b = join.get("from_page", 0)-1, join.get("to_page", 0)-1
        if not (0 <= a < len(kept) and 0 <= b < len(kept)):
            continue
        status = join["status"]
        if status.startswith("unresolved"):
            for record, side in ((kept[a], "right"), (kept[b], "left")):
                h, w = record["page"].shape
                add(record, "seam", "接缝未确认，已分别保留；检查半小节是否可接上",
                    [round(w*.75) if side == "right" else 0, 0, round(w*.25), h],
                    record["segment"].end if side == "right" else record["segment"].start)
        elif status == "same_view":
            add(kept[b], "repeat", "相邻画面判为重复，核对是否实际演奏了重复小节")
    for row in row_info:
        if row.get("end_of_run") and row["complete"] < row["units"] and row.get("source_pages"):
            record = kept[row["source_pages"][-1]-1]
            h, w = record["page"].shape
            add(record, "boundary", "末尾小节边界未确认，已保留可见部分",
                [round(w*.75), 0, round(w*.25), h], record["segment"].end)
    return issues
