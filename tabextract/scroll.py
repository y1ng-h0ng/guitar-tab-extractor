"""Recover notation visible only during the animated move between stable pages."""

import cv2
import numpy as np

from .clean import clean_frames, observed_staff, stroke_response
from .pages import crop_frame, signature
from .support import check_cancel


def _similarity(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denominator) if denominator > 1e-6 else 0.0


def _peaks(response, radius, count=8):
    work = response.copy()
    found = []
    for _ in range(count):
        index = int(np.argmax(work))
        if work[index] < 0.4:
            break
        found.append(index)
        work[max(0, index - radius):index + radius + 1] = -1
    return found


def _number_strip(frame, polarity, groups, width):
    response = np.maximum(stroke_response(frame, polarity, 7).astype(np.float32) - 5, 0)
    strips = []
    for lines in groups:
        space = float(np.median(np.diff(lines)))
        top = max(0, round(lines[0] - space * 1.15))
        bottom = max(top + 1, round(lines[0] - space * 0.15))
        strips.append(cv2.resize(response[top:bottom], (width, bottom - top)))
    return np.vstack(strips)


def _register_step(previous, current, old_labels, new_labels):
    width = previous.shape[1]
    template_width = max(32, round(width * 0.45))
    response = cv2.matchTemplate(
        previous, current[:, :template_width], cv2.TM_CCORR_NORMED
    )[0]
    index = int(np.argmax(response))
    score = float(response[index])
    radius = max(4, round(width * 0.02))
    other = response.copy()
    other[max(0, index - radius):index + radius + 1] = -1
    margin = score - float(other.max())
    if index == 0 and score >= 0.98:
        return 0.0, score, margin
    if score >= 0.70 and margin >= 0.08:
        objective = lambda x: float(response[x])
    else:
        # Circled chords and repeated stems can correlate at several offsets.
        # Compare the measure-number strip across the ENTIRE available overlap,
        # not just the fixed prefix that may contain one ambiguous digit.
        labels = cv2.matchTemplate(
            old_labels, new_labels[:, :template_width], cv2.TM_CCORR_NORMED
        )[0]
        candidates = set(_peaks(response, radius) + _peaks(labels, radius))
        ranked = {}
        for candidate in candidates:
            for shift in range(max(0, candidate - 2), min(len(response) - 1, candidate + 2) + 1):
                body_score = _similarity(previous[:, shift:], current[:, :width - shift])
                if body_score >= 0.70:
                    label_score = _similarity(old_labels[:, shift:], new_labels[:, :width - shift])
                    ranked[shift] = (label_score, body_score)
        if not ranked:
            return None
        index = max(ranked, key=lambda x: ranked[x][0])
        label_score, score = ranked[index]
        runner_up = max((v[0] for x, v in ranked.items() if abs(x - index) > radius), default=0.0)
        margin = label_score - runner_up
        # Faint numbers can have moderate absolute correlation yet decisively
        # rule out every other offset. Still require a strong separation then.
        if margin < 0.025 or label_score < 0.50 or (label_score < 0.60 and margin < 0.10):
            return None
        objective = lambda x: _similarity(old_labels[:, x:], new_labels[:, :width - x])
    subpixel = 0.0
    if 0 < index < len(response) - 1:
        left, middle, right = objective(index - 1), objective(index), objective(index + 1)
        denominator = left - 2 * middle + right
        if abs(denominator) > 1e-6:
            subpixel = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return index + subpixel, score, margin


def track_scroll(frames, polarity):
    """Measure every consecutive leftward move; reject ambiguous registrations."""
    if len(frames) < 3:
        return None
    groups = observed_staff(frames[0], polarity)
    if not groups:
        return None
    features = [signature(f, polarity, groups) for f in frames]
    width = features[0].shape[1]
    scale = frames[0].shape[1] / width
    labels = [_number_strip(f, polarity, groups, width) for f in frames]
    positions, scores, margins = [0.0], [], []
    for index in range(1, len(frames)):
        registration = _register_step(features[index - 1], features[index],
                                      labels[index - 1], labels[index])
        if registration is None:
            return None
        shift, score, margin = registration
        step = max(0.0, shift * scale)
        if step < 1:
            step = 0.0
        positions.append(positions[-1] + step)
        scores.append(score)
        margins.append(margin)
    # Both ends must include stationary evidence from their stable pages.
    if positions[1] > 1 or positions[-1] - positions[-2] > 1:
        return None
    if np.count_nonzero(np.diff(positions) > 1) < 3:
        return None  # An instantaneous cut supplies no evidence for a gap.
    return {"positions": positions, "minimum_match": min(scores),
            "minimum_margin": min(margins)}


def load_scroll_frames(video_path, region, previous_segment, next_segment,
                       sample_fps=6, cancel_event=None):
    """Read every video frame around a short transition, including both anchors."""
    fps = previous_segment.fps
    guard = max(2, round(fps / sample_fps))
    start = max(previous_segment.first, previous_segment.last - guard)
    end = min(next_segment.last, next_segment.first + guard)
    if end <= start or end - start > fps * 2:
        return [], start, end
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError("无法读取翻页过程")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for index in range(start, end + 1):
            check_cancel(cancel_event)
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"读取翻页帧 {index} 失败")
            frames.append(crop_frame(frame, region))
        return frames, start, end
    finally:
        cap.release()


def make_scroll_bridge(video_path, region, previous_segment, next_segment,
                       polarity, sample_fps=6, cancel_event=None):
    frames, start, end = load_scroll_frames(
        video_path, region, previous_segment, next_segment, sample_fps, cancel_event
    )
    check_cancel(cancel_event)
    tracking = track_scroll(frames, polarity)
    if tracking is None:
        return None
    positions = np.array(tracking.pop("positions"))
    distance = float(positions[-1])
    if not frames[0].shape[1] * 0.05 < distance < frames[0].shape[1] * 1.45:
        return None
    anchor = distance / 2
    columns = anchor + np.arange(frames[0].shape[1])
    coverage = ((columns[None, :] >= positions[:, None])
                & (columns[None, :] < positions[:, None] + frames[0].shape[1])).sum(axis=0)
    if np.any(coverage < 3):
        return None  # Do not silently erase a gap seen too briefly for consensus.
    page = clean_frames(frames, polarity, cancel_event=cancel_event,
                        horizontal_offsets=positions - anchor)
    upscale = page.shape[1] / frames[0].shape[1]
    return {"page": page, "first_shift": anchor * upscale,
            "second_shift": (distance - anchor) * upscale,
            "start_seconds": start / previous_segment.fps,
            "end_seconds": end / previous_segment.fps,
            "samples": len(frames), "scroll_pixels": distance, **tracking}
