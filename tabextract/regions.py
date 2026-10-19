"""Reversible rectangles in extracted-page pixels, applied after stitching."""
import numpy as np


def validate_region(kind, bbox, shape):
    if kind not in ('erase', 'protect') or len(bbox) != 4:
        raise ValueError('请选择擦除或保护，并框选有效区域')
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bbox):
        raise ValueError('框选坐标必须是整数')
    x, y, w, h = map(int, bbox)
    if x < 0 or y < 0 or min(w, h) < 1 or x+w > shape[1] or y+h > shape[0]:
        raise ValueError('框选区域超出提取图范围')
    return {'kind': kind, 'bbox': [x, y, w, h]}


def apply_regions(canvas, placements):
    """placements: (record, x offset, y offset); protection wins in overlaps.

    Protection restores only observed, temporally supported source pixels.
    Applying at the end prevents overlap copies/bridge cleanup undoing edits.
    """
    result = canvas.copy()
    for kind in ('erase', 'protect'):
        for record, dx, dy in placements:
            reference = record.get('evidence_page', record.get('base_page', record['page']))
            for operation in record.get('regions', []):
                if operation['kind'] != kind:
                    continue
                x, y, w, h = operation['bbox']
                a, b = max(0, x+dx), min(result.shape[1], x+dx+w)
                c, d = max(0, y+dy), min(result.shape[0], y+dy+h)
                if b <= a or d <= c:
                    continue
                if kind == 'erase':
                    result[c:d, a:b] = 255
                else:
                    patch = reference[c-dy:d-dy, a-dx:b-dx]
                    result[c:d, a:b] = np.minimum(result[c:d, a:b], patch)
    return result
