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


def _track_direction(frames, polarity):
    """Track leftward motion, omitting isolated unregistrable video frames.

    A skipped frame contributes neither a guessed position nor cleanup pixels:
    the next retained frame must independently match the last reliable frame.
    """
    if len(frames) < 3:
        return None
    groups = observed_staff(frames[0], polarity)
    if not groups:
        return None
    features = [signature(f, polarity, groups) for f in frames]
    width = features[0].shape[1]
    scale = frames[0].shape[1] / width
    labels = [_number_strip(f, polarity, groups, width) for f in frames]
    positions, indices, scores, margins = [0.0], [0], [], []
    previous = 0
    while previous < len(frames) - 1:
        for index in range(previous + 1, min(len(frames), previous + 4)):
            registration = _register_step(features[previous], features[index],
                                          labels[previous], labels[index])
            if registration is not None:
                break
        else:
            return None
        # Never bridge a long unreliable passage with sparse coincidences.
        if index - len(indices) > max(1, len(frames) // 5):
            return None
        shift, score, margin = registration
        step = max(0.0, shift * scale)
        if step < 1:
            step = 0.0
        positions.append(positions[-1] + step)
        indices.append(index)
        scores.append(score)
        margins.append(margin)
        previous = index
    # Both ends must include stationary evidence from their stable pages.
    if positions[1] > 1 or positions[-1] - positions[-2] > 1:
        return None
    if np.count_nonzero(np.diff(positions) > 1) < 3:
        return None  # An instantaneous cut supplies no evidence for a gap.
    return {"positions": positions, "frame_indices": indices,
            "minimum_match": min(scores),
            "minimum_margin": min(margins)}


def _path_score(features, tracking, scale):
    """Check non-adjacent frames, so one repeated motif cannot set the path."""
    indices, positions = tracking["frame_indices"], tracking["positions"]
    width = features[0].shape[1]
    scores = []
    for a in range(len(indices)):
        for b in range(a + 2, len(indices)):
            shift = round((positions[b] - positions[a]) / scale)
            if 5 < shift < width * 0.75:
                scores.append(_similarity(features[indices[a]][:, shift:],
                                          features[indices[b]][:, :width - shift]))
    return float(np.percentile(scores, 25)) if len(scores) >= 3 else 0.0


def track_scroll(frames, polarity):
    """Cross-check forward/backward tracks through repeated musical phrases."""
    forward = _track_direction(frames, polarity)
    # Reverse time and the horizontal axis: the same leftward matcher now
    # uses the other end of the overlap, independently of a misleading prefix.
    reverse = _track_direction([np.ascontiguousarray(f[:, ::-1])
                                for f in frames[::-1]], polarity)
    if reverse is not None:
        positions = np.array(reverse["positions"])
        reverse["positions"] = (positions[-1] - positions[::-1]).tolist()
        reverse["frame_indices"] = [len(frames) - 1 - i
                                    for i in reverse["frame_indices"][::-1]]
    candidates = [("forward", forward), ("reverse", reverse)]
    candidates = [(name, path) for name, path in candidates if path is not None]
    if not candidates:
        return None
    if forward is not None and reverse is not None:
        f = dict(zip(forward["frame_indices"], forward["positions"]))
        r = dict(zip(reverse["frame_indices"], reverse["positions"]))
        if max(abs(f[i] - r[i]) for i in f.keys() & r.keys()) <= 3:
            return {**forward, "tracking_direction": "forward"}
    groups = observed_staff(frames[0], polarity)
    features = [signature(f, polarity, groups) for f in frames]
    scale = frames[0].shape[1] / features[0].shape[1]
    ranked = sorted([(_path_score(features, path, scale), name, path)
                     for name, path in candidates], key=lambda item: item[0], reverse=True)
    score, name, path = ranked[0]
    # Disagreement needs strong whole-path evidence, not a preference for the
    # larger displacement. If both explanations fit, preserve separate views.
    if score < 0.78 or (len(ranked) > 1 and score - ranked[1][0] < 0.06):
        return None
    return {**path, "tracking_direction": name, "path_match": score}


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
    indices = tracking.pop("frame_indices")
    skipped = len(frames) - len(indices)
    frames = [frames[i] for i in indices]
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
            "samples": len(frames), "skipped_frames": skipped,
            "scroll_pixels": distance, **tracking}
