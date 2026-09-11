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
    def test_large_score_keeps_thick_beams_and_single_note_flags(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                gray = np.full((245, 640), 25, np.uint8)
                for y in range(65, 166, 20):
                    cv2.line(gray, (10, y), (630, y), 230, 2)
                rhythm = np.zeros_like(gray)
                for x in (110, 180, 260):
                    cv2.line(rhythm, (x, 160), (x, 215), 230, 2)
                cv2.line(rhythm, (110, 215), (180, 215), 230, 5)
                cv2.polylines(rhythm, [np.array([[260, 215], [273, 207],
                                               [276, 197], [270, 189]])],
                              False, 230, 4)
                gray = np.maximum(gray, rhythm)
                if polarity == "dark":
                    gray = 255 - gray
                frames = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)] * 12
                page = clean_frames(frames, polarity)
                out = cv2.resize(page, (640, 245), interpolation=cv2.INTER_AREA)
                for x0, y0, x1, y1 in [(120, 212, 170, 219), (265, 185, 280, 214)]:
                    target = rhythm[y0:y1, x0:x1] > 140
                    kept = out[y0:y1, x0:x1] < 180
                    self.assertGreater(float((target & kept).sum() / target.sum()), 0.95)
                self.assertEqual(len(staff_groups(page)), 1)

    def test_temporarily_occluded_fret_numbers_chords_and_dots_survive(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                ink = np.zeros((180, 480), np.uint8)
                for y in range(65, 116, 10):
                    cv2.line(ink, (10, y), (470, y), 230, 1)
                notes = np.zeros_like(ink)
                for text, x, y in [("7", 110, 103), ("12", 180, 83), ("9", 180, 103)]:
                    cv2.putText(notes, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, 230, 1, cv2.LINE_AA)
                cv2.circle(notes, (220, 138), 1, 230, -1)
                frames = []
                for i in range(20):
                    gray = np.maximum(np.maximum(ink, notes), 25)
                    if i < 7:
                        gray[55:150, 105:228] = 25
                        gray[125:155, 300] = 230  # Transient cursor.
                    else:
                        # A weak piece of moving scenery is not a new note,
                        # even when it appears in a majority of samples.
                        gray[88:100, 330:333] = 65
                    if polarity == "dark":
                        gray = 255 - gray
                    frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                page = clean_frames(frames, polarity)
                out = cv2.resize(page, (480, 180), interpolation=cv2.INTER_AREA)
                # Check each note separately: total ink or measure counts can
                # pass even when an entire individual note has disappeared.
                for x0, y0, x1, y1 in [(110, 90, 124, 104), (180, 70, 201, 84),
                                       (180, 90, 194, 104), (217, 135, 224, 142)]:
                    target = notes[y0:y1, x0:x1] > 140
                    kept = out[y0:y1, x0:x1] < 180
                    self.assertGreater(float((target & kept).sum() / target.sum()), 0.95)
                self.assertTrue((out[130:150, 299:302] > 245).all())
                self.assertTrue((out[89:94, 330:333] > 245).all())

    def test_sharpening_does_not_fade_a_low_contrast_symbol(self):
        gray = np.full((160, 400), 75, np.uint8)
        for y in range(70, 121, 10):
            cv2.line(gray, (10, y), (390, y), 230, 1)
        cv2.putText(gray, "9", (100, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, 102, 1, cv2.LINE_AA)
        frames = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)] * 12
        page = clean_frames(frames, "bright")
        symbol = page[120:165, 297:345]
        self.assertGreater(np.count_nonzero(symbol < 180), 70)

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
