"""Synthetic regressions for lost stems and filled-in technique labels.

Run with: python -m unittest discover -s tests -v
No video or extracted score is stored in the repository.
"""

import unittest

import cv2
import numpy as np

from tabextract.clean import clean_frames
from tabextract.geometry import staff_groups


class DetailPreservationTests(unittest.TestCase):
    def test_occluded_stem_recovers_without_inventing_a_source_gap_or_cursor(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                ink = np.zeros((190, 640), np.uint8)
                for y in range(70, 121, 10):
                    cv2.line(ink, (10, y), (630, y), 230, 1)
                for x in (110, 170):
                    cv2.line(ink, (x, 115), (x, 172), 230, 1)
                # An intentionally empty interval must remain empty.
                ink[139:153, 168:173] = 0
                frames = []
                for i in range(20):
                    gray = np.maximum(ink, 25).copy()
                    if i < 7:
                        # Cover a real stem in 35% of frames; add a transient
                        # bright vertical overlay at a different location.
                        gray[139:153, 108:113] = 25
                        gray[125:175, 300] = 230
                    if polarity == "dark":
                        gray = 255 - gray
                    frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                page = clean_frames(frames, polarity)
                out = cv2.resize(page, (640, 190), interpolation=cv2.INTER_AREA)
                self.assertTrue((out[136:156, 110] < 100).all())
                self.assertTrue((out[142:150, 169:172] > 245).all())
                self.assertTrue((out[135:165, 299:302] > 245).all())
                self.assertEqual(len(staff_groups(page)), 1)

    def test_small_letter_counter_survives_cleanup(self):
        ink = np.zeros((190, 640), np.uint8)
        for y in range(70, 121, 10):
            cv2.line(ink, (10, y), (630, y), 230, 1)
        # A small P with an explicitly known internal counter; blur models
        # antialiasing/compression in the source instead of a perfect binary P.
        cv2.line(ink, (102, 48), (102, 59), 230, 1)
        cv2.rectangle(ink, (102, 48), (107, 54), 230, 1)
        ink = cv2.GaussianBlur(ink, (0, 0), 0.7)
        gray = np.maximum(ink, 25)
        frames = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)] * 12
        page = clean_frames(frames, "bright")
        # The center is white; the left stroke remains clearly visible.
        self.assertGreater(float(page[151:158, 313:318].mean()), 200)
        self.assertLess(float(page[148:165, 305:309].mean()), 100)


if __name__ == "__main__":
    unittest.main()
