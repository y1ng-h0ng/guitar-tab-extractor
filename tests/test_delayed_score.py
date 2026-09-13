"""The earliest visible score survives an intro, including a late fade-in."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tabextract.clean import clean_frames
from tabextract.geometry import staff_groups
from tabextract.pages import load_segment_frames, scan_segments


class FrameCapture:
    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 12 if prop == cv2.CAP_PROP_FPS else len(self.frames)

    def set(self, prop, value):
        self.index = int(value)
        return True

    def grab(self):
        self.index += 1
        return self.index <= len(self.frames)

    def retrieve(self):
        return True, self.frames[self.index - 1].copy()

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def release(self):
        pass


class DelayedScoreTests(unittest.TestCase):
    def test_delayed_first_page_is_detected_and_survives_cleanup(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                # Intro is longer than one third of the clip; it cannot be
                # handled by choosing time zero or one fixed fallback frame.
                blank = np.full((245, 640), 25, np.uint8)
                score = blank.copy()
                for y in range(65, 166, 20):
                    cv2.line(score, (10, y), (630, y), 230, 2)
                for x in (110, 180):
                    cv2.line(score, (x, 160), (x, 215), 230, 2)
                cv2.line(score, (110, 215), (180, 215), 230, 5)
                cv2.putText(score, "7", (108, 153), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, 230, 2)
                gray_frames = [blank] * 48 + [
                    cv2.addWeighted(score, a, blank, 1 - a, 0)
                    for a in np.linspace(0.05, 1, 12)
                ] + [score] * 48
                if polarity == "dark":
                    gray_frames = [255 - f for f in gray_frames]
                frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in gray_frames]
                with patch("tabextract.pages.cv2.VideoCapture",
                           side_effect=lambda _: FrameCapture(frames)):
                    segments, detected = scan_segments("fake", (0, 0, 640, 245),
                                                       log=lambda _: None)
                    self.assertEqual(detected, polarity)
                    self.assertGreaterEqual(segments[0].start, 4)
                    self.assertLess(segments[0].start, 5)
                    samples = load_segment_frames("fake", (0, 0, 640, 245), segments[0])
                page = clean_frames(samples, detected)
                self.assertEqual(len(staff_groups(page)), 1)
                self.assertLess(float(page[642:649, 360:510].mean()), 100)


if __name__ == "__main__":
    unittest.main()
