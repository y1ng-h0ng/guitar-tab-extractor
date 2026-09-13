"""Detect page changes from notation features, then sample stable spans."""

from dataclasses import dataclass
import cv2
import numpy as np
from .clean import infer_polarity, observed_staff, stroke_response
from .geometry import staff_groups
from .support import check_cancel


@dataclass
class Segment:
    first: int
    last: int
    fps: float

    @property
    def start(self):
        return self.first / self.fps

    @property
    def end(self):
        return self.last / self.fps


def crop_frame(frame, region):
    x, y, w, h = region
    return frame[y : y + h, x : x + w]


def signature(crop, polarity, groups=None):
    response = stroke_response(crop, polarity, kernel=5).astype(np.float32)
    neutral = crop.min(axis=2) if polarity == "bright" else 255 - crop.max(axis=2)
    response[neutral < 90] = 0
    response = np.maximum(response - 30, 0)
    if groups is None:
        line_page = 255 - np.clip(
            stroke_response(crop, polarity, 5).astype(np.int16) * 3, 0, 255
        ).astype(np.uint8)
        groups = staff_groups(line_page, coverage=0.18)
    if groups:
        top, bottom = groups[0][0], groups[-1][-1]
        space = float(np.median(np.diff(groups[0])))
        a, b = (
            max(0, int(top - space * 3)),
            min(crop.shape[0], int(bottom + space * 4.5)),
        )
        response[:a] = 0
        response[b:] = 0
        for lines in groups:
            for y in lines:
                r = round(y)
                response[max(0, r - 1) : r + 2] = 0
    response = cv2.GaussianBlur(response, (3, 3), 0.6)
    w = min(640, response.shape[1])
    h = max(16, min(240, round(response.shape[0] * w / response.shape[1])))
    return cv2.resize(response, (w, h), interpolation=cv2.INTER_AREA)


def scan_segments(
    video_path,
    region,
    sample_fps=6,
    polarity="auto",
    log=print,
    progress=None,
    cancel_event=None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError("无法打开视频，请确认文件完整且编码受支持")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("视频帧率无效，无法可靠地定位翻页")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / sample_fps))
        auto_mode = polarity == "auto"
        groups = []
        times, differences = [], []
        previous = None
        idx = 0
        while cap.grab():
            check_cancel(cancel_event)
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise OSError(f"读取第 {idx} 帧失败")
                crop = crop_frame(frame, region)
                if not groups:
                    candidate_polarity = infer_polarity(crop) if auto_mode else polarity
                    groups = observed_staff(crop, candidate_polarity)
                    if not groups:
                        # Intro footage and fade-in frames must not vote in
                        # the first page's consensus or determine its polarity.
                        idx += 1
                        continue
                    polarity = candidate_polarity
                    log(f"首个可见六线谱: {idx / fps:.2f} 秒")
                feature = signature(crop, polarity, groups)
                norm = float(np.linalg.norm(feature))
                if previous is not None:
                    old, old_norm = previous
                    if norm < 1 and old_norm < 1:
                        distance = 0.0
                    elif norm < 1 or old_norm < 1:
                        distance = 1.0
                    else:
                        distance = max(
                            0.0, 1 - float((feature * old).sum()) / (norm * old_norm)
                        )
                    differences.append(distance)
                previous = feature, norm
                times.append(idx)
                if progress and len(times) % max(1, round(sample_fps)) == 0:
                    progress(min(idx / max(total, 1), 0.99))
            idx += 1
        if not groups:
            raise RuntimeError("视频中未检测到可见六线谱，请检查框选区域与颜色模式")
        if len(times) < 2:
            raise ValueError("视频太短，至少需要两帧可用采样")
    finally:
        cap.release()
    diff = np.array(differences)
    median = float(np.median(diff))
    mad = float(np.median(abs(diff - median)))
    threshold = min(0.45, max(0.13, median + 8 * 1.4826 * mad))
    change = np.where(diff > threshold)[0]
    starts, ends = [0] + (change + 1).tolist(), change.tolist() + [len(times) - 1]
    segments = [
        Segment(times[a], times[b], fps)
        for a, b in zip(starts, ends)
        if times[b] - times[a] >= fps * 0.45
    ]
    if not segments:
        raise RuntimeError("没有稳定谱面：请提高采样频率或重新框选谱面")
    log(f"采样 {len(times)} 帧，候选稳定段 {len(segments)} 段；按谱面变化检测翻页")
    return segments, polarity


def load_segment_frames(video_path, region, segment, count=25, cancel_event=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError("重新读取视频失败")
    try:
        guard = min(round(segment.fps * 0.12), (segment.last - segment.first) // 6)
        indices = np.unique(
            np.linspace(segment.first + guard, segment.last - guard, count)
            .round()
            .astype(int)
        )
        frames = []
        for idx in indices:
            check_cancel(cancel_event)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"读取视频 {idx / segment.fps:.2f} 秒处失败")
            frames.append(crop_frame(frame, region))
        return frames
    finally:
        cap.release()


def detect_pages(video_path, region, sample_fps=6, log=print, **_compat):
    segments, _ = scan_segments(video_path, region, sample_fps, log=log)
    return [load_segment_frames(video_path, region, s, count=1)[0] for s in segments]
