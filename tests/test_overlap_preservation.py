"""Repeated notes with different measure numbers must not be deduplicated."""

import unittest

import cv2
import numpy as np

from tabextract.stitch import assemble_score


class OverlapPreservationTests(unittest.TestCase):
    def test_only_true_overlap_can_remove_repeated_music(self):
        score = np.full((175, 1200), 255, np.uint8)
        for y in range(70, 121, 10):
            cv2.line(score, (0, y), (1199, y), 0, 1)
        for i, x in enumerate(range(50, 1200, 150)):
            cv2.line(score, (x, 70), (x, 120), 0, 1)
            cv2.putText(score, str(11 + i), (x - 8, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
            cv2.putText(score, str([3, 7, 12, 5, 9, 14, 2, 8][i]), (x + 30, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.line(score, (x + 30, 110), (x + 30, 150), 0, 1)
        first, second = score[:, :800].copy(), score[:, 400:].copy()
        actual_overlap = assemble_score([first, second], log=lambda _: None)
        self.assertEqual(len(actual_overlap.runs), 1)
        self.assertEqual(actual_overlap.joins[0]["status"], "joined")
        # Exactly the same musical phrase appears again at later measure
        # numbers. Global image similarity used to cut away these real notes.
        second[:70] = 255
        for i, x in enumerate(range(50, 1200, 150)):
            if x >= 400:
                cv2.putText(second, str(31 + i), (x - 408, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
        repeated_music = assemble_score([first, second], log=lambda _: None)
        self.assertEqual(len(repeated_music.runs), 2)
        np.testing.assert_array_equal(repeated_music.runs[0], first)
        np.testing.assert_array_equal(repeated_music.runs[1], second)


if __name__ == "__main__":
    unittest.main()
