"""Translucent strings must establish scale before cleaning thick rhythms."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tabextract.clean import clean_frames, observed_staff, stroke_response
from tabextract.geometry import staff_groups
from tabextract.pages import scan_segments
from test_delayed_score import FrameCapture


def faint_score(lines=(60, 88, 116, 144, 172, 200), end=630):
    gray = np.full((285, 640), 45, np.uint8)
    for y in lines:
        cv2.line(gray, (10, y), (end, y), 63, 2)
    return gray


class FaintStaffTests(unittest.TestCase):
    def test_faint_six_lines_preserve_thick_beams_and_source_gaps(self):
        for polarity in ('bright', 'dark'):
            with self.subTest(polarity=polarity):
                gray = faint_score()
                # A real gap must not be reconstructed from predicted lines.
                gray[57:64, 390:430] = 45
                for x in (100, 175):
                    cv2.line(gray, (x, 203), (x, 260), 230, 2)
                cv2.line(gray, (100, 260), (175, 260), 230, 6)
                cv2.putText(gray, '7', (105, 168), cv2.FONT_HERSHEY_SIMPLEX, .7, 230, 2)
                if polarity == 'dark': gray = 255 - gray
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                response = stroke_response(frame, polarity, 5).astype(np.int16)
                old = 255 - np.clip(response * 3, 0, 255).astype(np.uint8)
                self.assertFalse(staff_groups(old, coverage=.18))
                self.assertEqual(len(observed_staff(frame, polarity)), 1)
                page = clean_frames([frame] * 9, polarity)
                self.assertEqual(len(staff_groups(page)), 1)
                self.assertLess(float(page[258*3:262*3, 110*3:165*3].mean()), 80)
                self.assertTrue((page[58*3:63*3, 397*3:423*3] > 245).all())

    def test_weak_background_needs_six_long_equally_spaced_lines(self):
        for gray in (faint_score(lines=(60, 88, 116, 144, 172)),
                     faint_score(end=180),
                     faint_score(lines=(60, 82, 120, 148, 172, 220))):
            self.assertFalse(observed_staff(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'bright'))

    def test_repeated_bright_glyph_rows_cannot_override_wide_faint_staff(self):
        gray = faint_score()
        for y in range(133, 179, 9):
            cv2.line(gray, (200, y), (345, y), 230, 1)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        groups = observed_staff(frame, 'bright')
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(float(np.median(np.diff(groups[0]))), 28, delta=1)

    def test_scan_does_not_drop_opening_when_only_strings_are_faint(self):
        gray = faint_score()
        cv2.putText(gray, '7', (100, 168), cv2.FONT_HERSHEY_SIMPLEX, .7, 230, 2)
        for polarity in ('bright', 'dark'):
            frame = cv2.cvtColor(gray if polarity == 'bright' else 255-gray, cv2.COLOR_GRAY2BGR)
            with patch('tabextract.pages.cv2.VideoCapture', side_effect=lambda _: FrameCapture([frame]*48)):
                segments, detected = scan_segments('fake', (0, 0, 640, 285), log=lambda _: None)
            self.assertEqual(detected, polarity)
            self.assertEqual(segments[0].start, 0)


if __name__ == '__main__':
    unittest.main()
