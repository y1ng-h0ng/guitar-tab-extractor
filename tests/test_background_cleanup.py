"""Moving notation survives removal of screen-fixed footage edges."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tabextract.background import motion_background_mask, remove_motion_background
from tabextract.clean import clean_frames
from tabextract.stitch import assemble_score
from test_scroll_bridge import scrolling_score


class BackgroundCleanupTests(unittest.TestCase):
    def test_fixed_cable_is_removed_without_cutting_notes_stems_or_ties(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                score, _, positions = scrolling_score("bright")
                cv2.ellipse(score, (285, 51), (50, 18), 0, 180, 360, 230, 1)
                cv2.putText(score, "P", (275, 28), cv2.FONT_HERSHEY_SIMPLEX, .4, 230, 1)
                cv2.circle(score, (360, 42), 1, 230, -1)
                noise = np.zeros((220, 640), np.uint8)
                cv2.polylines(noise, [np.array([[200, 0], [255, 14], [280, 80],
                                              [310, 135], [360, 205]])], False, 210, 2)
                frames = []
                for x in positions:
                    gray = np.maximum(score[:, x:x + 640], noise)
                    if polarity == "dark":
                        gray = 255 - gray
                    frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                original = clean_frames([frames[0]] * 8, polarity)
                mask = motion_background_mask(frames, positions, polarity)
                result = remove_motion_background(original, mask)
                out = cv2.resize(result, (640, 220), interpolation=cv2.INTER_AREA)
                old = cv2.resize(original, (640, 220), interpolation=cv2.INTER_AREA)
                notes = score[:, :640] > 140
                # Compare individual notation regions, not just total ink.
                for x0, y0, x1, y1 in [(225, 27, 340, 60), (350, 37, 367, 47),
                                       (345, 111, 375, 194), (192, 104, 226, 133)]:
                    target = notes[y0:y1, x0:x1] & (old[y0:y1, x0:x1] < 180)
                    self.assertGreater(np.count_nonzero(target), 0)
                    self.assertGreater(np.mean(out[y0:y1, x0:x1][target] < 180), .95)
                scenery = (noise > 140) & ~cv2.dilate(
                    notes.astype(np.uint8), np.ones((9, 9), np.uint8)
                ).astype(bool)
                scenery[:15] = False  # Left boundary has less independent coverage.
                self.assertGreater(np.mean(out[scenery] > 220), .75)
                # No input mutation: an unverified bridge must retain originals.
                np.testing.assert_array_equal(original, clean_frames([frames[0]] * 8, polarity))

    def test_stationary_duration_does_not_outvote_motion_and_edges_stay(self):
        _, frames, positions = scrolling_score("bright")
        original = motion_background_mask(frames, positions, "bright")
        repeated = motion_background_mask(frames[:1] * 20 + frames + frames[-1:] * 30,
                                           [0] * 20 + positions + [680] * 30, "bright")
        np.testing.assert_array_equal(original, repeated)
        self.assertFalse(original[:, :20].any())
        self.assertFalse(original[:, -40:].any())
        self.assertIsNone(motion_background_mask([frames[0]] * 8, [0] * 8, "bright"))
        self.assertIsNone(motion_background_mask([frames[0]] * 3 + [frames[-1]] * 3,
                                                 [0] * 3 + [680] * 3, "bright"))

    def test_unsupported_small_glyphs_and_long_stems_are_kept_whole(self):
        gray = np.full((180, 480), 25, np.uint8)
        for y in range(70, 121, 10):
            cv2.line(gray, (5, y), (475, y), 230, 1)
        cv2.putText(gray, "P", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .45, 230, 1)
        cv2.circle(gray, (150, 45), 1, 230, -1)
        cv2.line(gray, (210, 28), (210, 167), 230, 1)
        original = clean_frames([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)] * 8, "bright")
        result = remove_motion_background(original, np.ones((180, 480), np.uint8))
        for a, b, c, d in [(94, 37, 118, 53), (146, 41, 155, 49), (208, 25, 213, 170)]:
            np.testing.assert_array_equal(result[b*3:d*3, a*3:c*3], original[b*3:d*3, a*3:c*3])

    def test_cleanup_requires_both_seams_and_mask_stays_out_of_report(self):
        score, _, _ = scrolling_score("bright")
        score = np.where(score > 140, 0, 255).astype(np.uint8)
        first, middle, last = score[:, :640], score[:, 340:980], score[:, 680:]
        bridge = {"page": middle, "first_shift": 340, "second_shift": 340,
                  "samples": 14, "background_mask": np.ones((220, 1320), np.uint8)}
        valid = assemble_score([first, last], log=lambda _: None, bridge_provider=lambda *_: bridge)
        self.assertEqual(valid.joins[0]["status"], "joined_scroll")
        self.assertNotIn("background_mask", valid.joins[0])
        bridge["second_shift"] = 280
        with patch("tabextract.stitch.remove_motion_background") as cleanup:
            failed = assemble_score([first, last], log=lambda _: None, bridge_provider=lambda *_: bridge)
            cleanup.assert_not_called()
        np.testing.assert_array_equal(failed.runs[0], first)
        np.testing.assert_array_equal(failed.runs[1], last)


if __name__ == "__main__":
    unittest.main()
