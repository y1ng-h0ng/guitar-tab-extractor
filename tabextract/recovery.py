"""Bounded local retries; accept only improvements supported by source frames."""
import cv2
import numpy as np
from .clean import clean_frames, observed_staff, stroke_response, notation_response
from .geometry import staff_groups
from .pages import signature
from .support import check_cancel


def evidence_metrics(page, frames, polarity):
    h, w = frames[0].shape[:2]
    small = cv2.resize(page, (w, h), interpolation=cv2.INTER_AREA)
    sample = frames[::max(1, len(frames)//9)][:9]
    evidence = np.median([notation_response(f, polarity, 5) for f in sample], axis=0)
    groups = observed_staff(sample[len(sample)//2], polarity)
    zone = np.zeros((h, w), bool)
    for lines in groups:
        space = float(np.median(np.diff(lines)))
        zone[max(0, round(lines[0]-3*space)):min(h, round(lines[-1]+4*space))] = True
    marks = (evidence >= 18) & zone
    for lines in groups:
        for line in lines:
            marks[max(0, round(line)-1):round(line)+2] = False
    ink = small < 210
    nearby = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    support = cv2.dilate((evidence >= 8).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    edges = np.abs(cv2.Laplacian(small, cv2.CV_32F))
    return {'groups': len(staff_groups(page)), 'source_groups': len(groups),
            'retention': float(nearby[marks].mean()) if marks.any() else 0.,
            'clarity': float(edges[marks].mean()) if marks.any() else 0.,
            'unsupported': float((~support)[ink & zone].mean()) if (ink & zone).any() else 1.}, small, marks


def improves(baseline, candidate, frames, polarity):
    old, a, marks = evidence_metrics(baseline, frames, polarity)
    new, b, _ = evidence_metrics(candidate, frames, polarity)
    retained = cv2.dilate((b < 210).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    old_ink = a < 170
    preserved = float(retained[old_ink].mean()) if old_ink.any() else 1.
    detail = old_ink.copy()
    for lines in observed_staff(frames[len(frames)//2], polarity):
        for y in lines: detail[max(0, round(y)-1):round(y)+2] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(detail.astype(np.uint8), 8)
    component_preserved = 1.
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 2: continue
        component = labels[y:y+h, x:x+w] == index
        component_preserved = min(component_preserved, float(retained[y:y+h,x:x+w][component].mean()))
    # Never reward a blank output, excess texture, changed staff count, or lost
    # existing strokes merely because the diagnostic list becomes shorter.
    valid = (new['groups'] > 0 and new['groups'] == new['source_groups']
             and preserved >= .995 and component_preserved >= .98
             and new['unsupported'] <= max(.03, old['unsupported']+.005)
             and (np.count_nonzero(b < 210) <= max(30, np.count_nonzero(a < 210)*1.30)
                  or old['groups'] == 0))
    better = ((old['groups'] == 0 and new['retention'] >= .9) or
              new['retention']-old['retention'] >= max(.015, (1-old['retention'])*.25) or
              (old['groups'] > 0 and old['clarity'] > 0 and new['clarity'] > old['clarity']*1.15
               and new['retention'] >= old['retention'] and new['unsupported'] <= old['unsupported']+.005))
    return bool(valid and better), {'before': old, 'after': new, 'preserved': preserved,
                                   'minimum_component_preserved': component_preserved}


def sharp_stable_frames(frames, polarity):
    reference = signature(frames[len(frames)//2], polarity).ravel()
    ref_norm = np.linalg.norm(reference)
    ranked = []
    for index, frame in enumerate(frames):
        feature = signature(frame, polarity).ravel()
        similarity = float(feature @ reference / max(1., np.linalg.norm(feature)*ref_norm))
        if similarity >= .94:
            response = stroke_response(frame, polarity, 5)
            ranked.append((float(cv2.Laplacian(response, cv2.CV_32F).var()), index))
    # Keep at least 60% of the denser sample; a single clear cursor cannot win.
    count = max(9, int(np.ceil(len(frames)*.6)))
    if len(ranked) < count:
        return frames
    indices = sorted(i for _, i in sorted(ranked, reverse=True)[:count])
    return [frames[i] for i in indices]


def recover_page(page, evidence_page, frames, polarity, gain, diagnostics, load_more,
                 cancel_event=None, log=print):
    metrics, _, _ = evidence_metrics(page, frames, polarity)
    sharpness = [cv2.Laplacian(stroke_response(f, polarity, 5), cv2.CV_32F).var()
                 for f in frames[::max(1, len(frames)//7)]]
    flagged = bool(diagnostics) or metrics['groups'] == 0 or (
        metrics['source_groups'] > 0 and (metrics['retention'] < .90 or
        max(sharpness) > max(1., float(np.median(sharpness)))*1.8))
    if not flagged:
        return page, evidence_page, diagnostics, None
    log('此片段存在图像疑点，正在增加采样并验证局部补救 ...')
    check_cancel(cancel_event)
    dense = load_more()
    selected = sharp_stable_frames(dense, polarity)
    attempts = []
    for label, samples, factor in [('加密采样', dense, gain),
                                    ('稳定清晰帧', selected, min(2., gain*1.15))]:
        check_cancel(cancel_event)
        issues, evidence = [], {}
        candidate = clean_frames(samples, polarity, detail_gain=factor,
            diagnostics=issues, evidence_out=evidence, cancel_event=cancel_event)
        accepted, scores = improves(page, candidate, dense, polarity)
        attempts.append(dict(method=label, accepted=accepted, metrics=scores, samples=len(samples)))
        if accepted:
            log(f'局部补救通过：{label}，原有笔画保留率 {scores["preserved"]:.1%}。')
            return candidate, evidence['page'], issues, {'accepted': True, 'attempts': attempts}
    log('补救候选未达到可靠改善标准，保留原提取结果并提示核对。')
    return page, evidence_page, diagnostics, {'accepted': False, 'attempts': attempts}


def recover_bridge(previous, current, bridge, retry, cancel_event=None, log=print):
    """Try one wider transition, accepting it only when both seams verify."""
    from .stitch import find_horizontal_shift, _seam, _tracked_match
    groups = staff_groups(previous)
    if len(groups) != 1:
        return bridge

    def verified(value):
        if value is None:
            return False
        middle_groups = staff_groups(value['page'])
        return (len(middle_groups) == 1 and
            _tracked_match(previous, value['page'], groups[0], value['first_shift']) is not None and
            _tracked_match(value['page'], current, middle_groups[0], value['second_shift']) is not None)

    if verified(bridge):
        return bridge
    direct = find_horizontal_shift(previous, current, groups[0])
    direct_ok = direct and (direct['duplicate'] or _seam(previous, current, groups[0], direct['shift']) is not None)
    if bridge is None and direct_ok:
        return bridge
    log('接缝尚未确认，扩大翻页取帧范围重试一次 ...')
    check_cancel(cancel_event)
    candidate = retry()
    check_cancel(cancel_event)
    if verified(candidate):
        log('补救接缝已通过前后两侧验证。')
        return dict(candidate, recovery='wider_transition')
    # Keep failed motion evidence: it may veto a misleading repeated-note match.
    return bridge
