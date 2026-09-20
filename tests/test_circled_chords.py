"""Whole-note chord ovals are notation, not extra measure boundaries."""

import unittest
import cv2
import numpy as np
from tabextract.geometry import bar_lines, staff_groups
from tabextract.stitch import assemble_score


class CircledChordTests(unittest.TestCase):
    def test_three_digit_measure_label_moves_as_a_whole_to_the_next_row(self):
        score = np.full((240, 1250), 255, np.uint8)
        for y in range(70, 171, 20):
            cv2.line(score, (10, y), (1240, y), 0, 1)
        for x in [20, 320, 620, 920, 1220]:
            cv2.line(score, (x, 70), (x, 170), 0, 2)
        cv2.putText(score, '102', (586, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        label = score[50:66, 584:620]
        result = assemble_score([score], bars_per_row=2, log=lambda _: None)
        self.assertEqual(len(result.rows), 2)
        matched = cv2.matchTemplate(result.rows[1], label, cv2.TM_SQDIFF)
        self.assertLess(float(matched.min()), 1)
        top = round(staff_groups(result.rows[0])[0][0])
        self.assertTrue((result.rows[0][:top-3] == 255).all())

    def test_oval_sides_cannot_split_measures_or_chords_across_rows(self):
        score = np.full((260, 1250), 255, np.uint8)
        lines = np.arange(70, 171, 20)
        for y in lines:
            cv2.line(score, (10, y), (1240, y), 0, 1)
        expected = [20, 320, 620, 920, 1220]
        for x in expected:
            cv2.line(score, (x, 70), (x, 170), 0, 2)
        for x in expected[:-1]:
            # Tall, narrow ovals, like six-note full chords in the video.
            cv2.ellipse(score, (x + 33, 120), (15, 62), 0, 0, 360, 0, 2)
            for j,y in enumerate(lines):
                cv2.putText(score, str(j % 4), (x + 26, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
        actual = bar_lines(score, staff_groups(score)[0])
        self.assertEqual(len(actual), len(expected))
        np.testing.assert_allclose(actual, expected, atol=1)
        result = assemble_score([score], bars_per_row=2, log=lambda _: None)
        self.assertEqual(result.complete_measures, 4)
        self.assertEqual(len(result.rows), 2)


if __name__ == '__main__':
    unittest.main()
