"""Separate static notation from footage using within-page consensus.
No OCR: letters, dots, ties and stems stay image strokes. No area-based deletion
of small components; no vertical clipping to the six staff lines.
"""

import cv2
import numpy as np
from .geometry import staff_groups
from .support import check_cancel


def stroke_response(bgr, polarity="bright", kernel=9):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    neutral = bgr.min(axis=2) if polarity == "bright" else 255 - bgr.max(axis=2)
    if polarity == "dark":
        gray = 255 - gray
    # A pair of 1-D openings erases L/T junctions: each direction mistakes
    # the crossing stroke for background. A 2-D opening preserves those
    # evidenced junctions together with the thin stems and beams.
    response = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, np.ones((kernel, kernel), np.uint8)
    )
    response[neutral < 50] = 0
    return response


def infer_polarity(bgr):
    """Use long-line evidence rather than total dark/bright background area."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    above, below = np.roll(gray, 2, axis=0), np.roll(gray, -2, axis=0)
    scores = []
    for sign in (1, -1):
        ridge = np.minimum(sign * (gray - above), sign * (gray - below)) > 25
        scores.append(float(np.sort(ridge.mean(axis=1))[-12:].sum()))
    return "bright" if scores[0] >= scores[1] else "dark"


def observed_staff(bgr, polarity):
    """Locate staff lines in source evidence before consensus can reject them."""
    response = stroke_response(bgr, polarity, kernel=5).astype(np.int16)
    return staff_groups(
        255 - np.clip(response * 3, 0, 255).astype(np.uint8), coverage=0.18
    )


def _restore_stem_evidence(page, low, median, high, groups, upscale):
    """Recover observed vertical strokes briefly hidden by footage/playheads.

    Relax the normal 60%-support requirement slightly for long vertical
    strokes seen in a majority of frames. Every restored pixel needs evidence;
    no geometric line is drawn across an empty gap in the original notation.
    """
    evidence = ((median >= 8) & (high >= 25)).astype(np.uint8)
    for lines in groups:
        space = float(np.median(np.diff(lines)))
        a = max(0, round(lines[0] - 4 * space))
        b = min(page.shape[0], round(lines[-1] + 4 * space) + 1)
        length = max(3, round(space * 0.85))
        vertical = cv2.morphologyEx(
            evidence, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)
        )
        # Select by intersection with the notation envelope, but keep the
        # entire observed stroke so the envelope itself cannot cut a stem.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
        keep = np.zeros(count, np.uint8)
        for i in range(1, count):
            y, height = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
            keep[i] = y < b and y + height > a
        vertical = keep[labels]
        # Include antialiased sides, but never extend into unsupported pixels.
        vertical = cv2.dilate(vertical, np.ones((1, max(1, upscale)), np.uint8))
        restore = (vertical > 0) & (evidence > 0) & (low < 10)
        alpha = np.clip((high - 8) * (255 / 35), 0, 255)
        page[:] = np.minimum(
            page, np.where(restore, 255 - alpha, 255).astype(np.uint8)
        )


def _restore_compact_evidence(page, majority, high, groups):
    """Keep compact notes visible in a strict majority of the sampled frames.

    This is deliberately narrower than relaxing consensus for the whole page:
    long background edges and cursors are not compact musical marks. Faint
    scenery cannot seed a new component, and every edge needs pixel evidence.
    """
    evidence = (majority >= 10) & (high >= 20)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        evidence.astype(np.uint8), 8
    )
    keep = np.zeros(count, bool)
    for i in range(1, count):
        x, y, width, height, _ = stats[i]
        for lines in groups:
            space = float(np.median(np.diff(lines)))
            if (lines[0] - 4 * space <= y
                    and y + height <= lines[-1] + 4 * space
                    and height <= 1.7 * space and width <= 3 * space):
                region = np.s_[y:y + height, x:x + width]
                component = labels[region] == i
                keep[i] = np.any((majority[region][component] >= 20)
                                 & (high[region][component] >= 60))
                break
    mask = keep[labels]
    alpha = np.clip((high - 8) * (255 / 35), 0, 255)
    page[mask] = np.minimum(page[mask], (255 - alpha[mask]).astype(np.uint8))


def _sharpen_annotations(page, low, high, groups, upscale):
    """Keep holes and gaps in small labels instead of saturating them black.

    Use a modest unsharp mask on the grayscale evidence BEFORE contrast
    clipping. Sharpening the already clipped binary-looking result cannot
    recover a P's counter or separate the strokes of a slide label.
    """
    response = high.astype(np.float32)
    blurred = cv2.GaussianBlur(response, (0, 0), max(0.5, 0.47 * upscale))
    sharp = np.clip(response + (response - blurred), 0, 255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (page < 245).astype(np.uint8), 8
    )
    for i in range(1, count):
        x, y, width, height, _ = stats[i]
        for lines in groups:
            space = float(np.median(np.diff(lines)))
            if (lines[0] - 4 * space <= y
                    and y + height <= lines[-1] + 4 * space
                    and height <= 1.7 * space and width <= 3 * space):
                region = np.s_[y:y + height, x:x + width]
                component = labels[region] == i
                # Fixed contrast scaling can almost erase a faint label or a
                # small note above the top string. Only refine strong glyphs,
                # and fall back if too much of their visible ink would vanish.
                strength = np.percentile(high[region][component], 90)
                if strength < 80:
                    break
                # Match the gain to the observed glyph rather than using one
                # threshold for faint labels and bright fret numbers alike.
                gain = 255 / max(60, strength * 0.8)
                alpha = np.clip((sharp[region] - 8) * gain, 0, 255)
                alpha[low[region] < 10] = 0
                refined = 255 - alpha.astype(np.uint8)
                original_ink = np.count_nonzero(page[region][component] < 180)
                refined_ink = np.count_nonzero(refined[component] < 180)
                if refined_ink >= max(1, original_ink * 0.6):
                    page[region][component] = refined[component]
                break


def _temporal_percentiles(features):
    # The interpolated median of an even sample count can admit a cursor
    # visible in exactly half the frames. Use the lower middle observation
    # for majority evidence; retain interpolated percentiles for contrast.
    n = len(features)
    middle = 100 * ((n - 1) // 2) / (n - 1) if n > 1 else 50
    return np.percentile(np.stack(features), [25, 40, middle, 75], axis=0)


def _covered_percentiles(features, offsets, upscale, minimum=3):
    """Ignore out-of-frame columns when combining registered scrolling frames."""
    height, width = features[0].shape
    bounds = [(max(0, int(np.ceil(dx * upscale))),
               min(width, int(np.floor(width + dx * upscale)))) for dx in offsets]
    cuts = sorted({0, width, *(max(0, min(width, v)) for pair in bounds for v in pair)})
    result = np.zeros((4, height, width), np.float32)
    for left, right in zip(cuts, cuts[1:]):
        usable = [f[:, left:right] for f, (a, b) in zip(features, bounds)
                  if a <= left and b >= right]
        if len(usable) >= minimum:
            result[:, :, left:right] = _temporal_percentiles(usable)
    return result


def clean_frames(frames, polarity="auto", upscale=3, cancel_event=None,
                 horizontal_offsets=None):
    if not frames:
        raise ValueError("没有用于清洗的帧")
    if horizontal_offsets is not None:
        if len(horizontal_offsets) != len(frames) or not np.isfinite(horizontal_offsets).all():
            raise ValueError("滚动帧位移必须与采样帧一一对应且为有限数值")
    if polarity == "auto":
        votes = [
            infer_polarity(frames[i]) for i in (0, len(frames) // 2, len(frames) - 1)
        ]
        polarity = max(set(votes), key=votes.count)
    references = [
        observed_staff(frames[i], polarity)
        for i in sorted(set((0, len(frames) // 2, len(frames) - 1)))
    ]
    references = [groups for groups in references if groups]
    source_groups = max(references, key=len) if references else []
    spacing = [float(np.median(np.diff(g))) for groups in references for g in groups]
    # A fixed three-source-pixel opening removes wide eighth-note beams as
    # background. Scale it with the observed staff, keeping small-score behavior.
    native_kernel = max(3, round(float(np.median(spacing)) / 3) | 1) if spacing else 3
    kernel = (native_kernel * upscale) | 1
    features = []
    for index, frame in enumerate(frames):
        check_cancel(cancel_event)
        image = cv2.resize(
            frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
        feature = stroke_response(image, polarity, kernel=kernel)
        if horizontal_offsets is not None:
            matrix = np.float32([[1, 0, horizontal_offsets[index] * upscale], [0, 1, 0]])
            feature = cv2.warpAffine(feature, matrix, feature.shape[::-1],
                                     flags=cv2.INTER_LINEAR, borderValue=0)
        features.append(feature)
    # Strong strokes need support in 60% of samples, not 75%: the old hard
    # veto could delete an ENTIRE fret number during a brief occlusion. Keep
    # the stricter consensus for faint texture so the relaxed rule does not
    # promote moving background details into apparent musical symbols.
    if horizontal_offsets is None:
        low, supported, median, high = _temporal_percentiles(features)
    else:
        low, supported, median, high = _covered_percentiles(
            features, horizontal_offsets, upscale
        )
    del features
    alpha = np.clip((high - 8) * (255 / 35), 0, 255)
    strong = (low < 10) & (supported >= 10) & (high >= 60)
    nearby = cv2.dilate(
        strong.astype(np.uint8), np.ones((2 * upscale + 1, 2 * upscale + 1), np.uint8)
    ) > 0
    # Keep the antialiased fringe of a recovered glyph, but do not seed new
    # symbols from weak background texture or from a nearby persistent string.
    valid = (low >= 10) | (nearby & (supported >= 10) & (high >= 20))
    alpha[~valid] = 0
    page = 255 - alpha.astype(np.uint8)
    # Accept faint pixels only within evidenced horizontal line bands.
    groups = staff_groups(page)
    if not groups and source_groups:
        # Bootstrap from actually observed source lines, never draw missing
        # lines: the restoration below still requires temporal pixel evidence.
        groups = [(lines + 0.5) * upscale - 0.5 for lines in source_groups]
    for lines in groups:
        for line in lines:
            y = int(round(line))
            a, b = max(0, y - upscale), min(page.shape[0], y + upscale + 1)
            valid = low[a:b] >= 5
            line_alpha = np.clip((high[a:b] - 4) * (255 / 30), 0, 255)
            page[a:b] = np.minimum(
                page[a:b], np.where(valid, 255 - line_alpha, 255).astype(np.uint8)
            )
    if groups:
        _restore_stem_evidence(page, low, median, high, groups, upscale)
        _restore_compact_evidence(page, median, high, groups)
        _sharpen_annotations(page, median, high, groups, upscale)
        # Remove detached scenery far from all staff systems. Components touching
        # the notation envelope are kept WHOLE, even when stems extend below it.
        zone = np.zeros(page.shape[0], bool)
        for lines in groups:
            spacing = float(np.median(np.diff(lines)))
            a = max(0, round(lines[0] - spacing * 4))
            b = min(page.shape[0], round(lines[-1] + spacing * 4) + 1)
            zone[a:b] = True
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (page < 245).astype(np.uint8), 8
        )
        keep = np.zeros(count, bool)
        for i in range(1, count):
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            keep[i] = zone[y : y + h].any()
        page[~keep[labels]] = 255
    return page


def clean_page(bgr_img, upscale=3, **_compat):
    """Single-image compatibility entry point; videos use clean_frames."""
    return clean_frames([bgr_img], upscale=upscale)
