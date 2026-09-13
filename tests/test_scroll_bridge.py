"""Scrolling-only notes must survive when the two resting views do not overlap."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tabextract.clean import clean_frames
from tabextract.pages import Segment
from tabextract.scroll import make_scroll_bridge, track_scroll
from tabextract.stitch import assemble_score


def scrolling_score(polarity):
    score = np.full((220, 1320), 25, np.uint8)
    for y in range(65, 146, 16):
        cv2.line(score, (0, y), (1319, y), 230, 1)
    for i, x in enumerate(range(20, 1320, 150)):
        cv2.line(score, (x, 65), (x, 145), 230, 1)
        cv2.putText(score, str(11 + i), (x - 9, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 230, 1)
        cv2.putText(score, str([3, 7, 12, 5, 8, 14, 2, 10, 6][i]), (x + 30, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 230, 1)
        cv2.line(score, (x + 30, 135), (x + 30, 190), 230, 1)
    # Neither the initial [0,640) view nor final [680,1320) view shows this note.
    cv2.putText(score, "9", (651, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 230, 1)
    cv2.line(score, (653, 150), (653, 195), 230, 1)
    positions = [0, 0, 20, 80, 160, 240, 320, 400, 480, 560, 620, 660, 680, 680]
    gray = score if polarity == "bright" else 255 - score
    frames = [cv2.cvtColor(gray[:, x:x + 640], cv2.COLOR_GRAY2BGR) for x in positions]
    return score, frames, positions


class ScrollBridgeTests(unittest.TestCase):
    def test_motion_bridge_preserves_a_note_missing_from_both_resting_views(self):
        for polarity in ("bright", "dark"):
            with self.subTest(polarity=polarity):
                score, frames, expected = scrolling_score(polarity)
                tracking = track_scroll(frames, polarity)
                self.assertIsNotNone(tracking)
                positions = np.array(tracking["positions"])
                np.testing.assert_allclose(positions, expected, atol=1.5)
                distance = positions[-1]
                middle = clean_frames(frames, polarity,
                                      horizontal_offsets=positions - distance / 2)
                first = clean_frames([frames[0]] * 8, polarity)
                last = clean_frames([frames[-1]] * 8, polarity)
                bridge = {"page": middle, "first_shift": distance * 1.5,
                          "second_shift": distance * 1.5, "samples": len(frames)}
                result = assemble_score([first, last], log=lambda _: None,
                                        bridge_provider=lambda *_: bridge)
                self.assertEqual(len(result.runs), 1)
                self.assertEqual(result.joins[0]["status"], "joined_scroll")
                out = cv2.resize(result.runs[0], (1320, 220), interpolation=cv2.INTER_AREA)
                target = score[130:198, 647:669] > 140
                kept = out[130:198, 647:669] < 180
                self.assertGreater(float((target & kept).sum() / target.sum()), 0.90)

    def test_an_instant_cut_is_not_evidence_for_the_missing_gap(self):
        _, frames, _ = scrolling_score("bright")
        self.assertIsNone(track_scroll([frames[0]] * 3 + [frames[-1]] * 3, "bright"))

    def test_unverified_second_seam_does_not_partially_modify_the_score(self):
        _, frames, positions = scrolling_score("bright")
        middle = clean_frames(frames, "bright",
                              horizontal_offsets=np.array(positions) - 340)
        first = clean_frames([frames[0]] * 8, "bright")
        last = clean_frames([frames[-1]] * 8, "bright")
        bridge = {"page": middle, "first_shift": 1020,
                  "second_shift": 1230, "samples": len(frames)}
        result = assemble_score([first, last], log=lambda _: None,
                                bridge_provider=lambda *_: bridge)
        self.assertEqual(len(result.runs), 2)
        np.testing.assert_array_equal(result.runs[0], first)
        np.testing.assert_array_equal(result.runs[1], last)

    def test_a_gap_without_enough_visible_samples_is_not_silently_erased(self):
        _, frames, _ = scrolling_score("bright")
        fast = [frames[i] for i in (0, 1, 6, 7, 12, 13)]
        self.assertIsNotNone(track_scroll(fast, "bright"))
        with patch("tabextract.scroll.load_scroll_frames", return_value=(fast, 0, 5)):
            bridge = make_scroll_bridge("fake", (0, 0, 640, 220),
                                        Segment(0, 1, 30), Segment(4, 5, 30), "bright")
        self.assertIsNone(bridge)


if __name__ == "__main__":
    unittest.main()
