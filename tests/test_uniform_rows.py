"""Uniform row widths must include dense music and the short final row."""

import unittest

import cv2
import numpy as np

from tabextract.geometry import staff_groups
from tabextract.pdfout import _add_final_barline, layout_rows
from tabextract.rowlayout import justify_row
from tabextract.stitch import assemble_score


def score_with_uneven_measures(connected=False):
    widths = [180, 200, 170, 220, 190, 130, 145, 135, 150, 140, 125, 130, 150]
    bars = np.r_[20, 20 + np.cumsum(widths)]
    score = np.full((170, int(bars[-1]) + 22), 255, np.uint8)
    for y in range(60, 111, 10):
        cv2.line(score, (20, y), (int(bars[-1]), y), 0, 1)
    for x in bars:
        cv2.line(score, (int(x), 60), (int(x), 110), 0, 1)
    for x in bars[:-1]:
        cv2.putText(score, '7', (int(x) + 30, 92), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
        cv2.line(score, (int(x) + 32, 96), (int(x) + 32, 145), 0, 1)
    if connected:
        # A continuous annotation across every column leaves no safe place
        # to insert whitespace. It must stay continuous in the fallback.
        cv2.line(score, (0, 35), (score.shape[1] - 1, 35), 0, 1)
    return score


class UniformRowTests(unittest.TestCase):
    def test_every_row_including_short_tail_matches_longest_natural_row(self):
        result = assemble_score([score_with_uneven_measures()], bars_per_row=5, log=lambda _: None)
        self.assertLess(result.row_info[-1]['units'], 5)
        target = max(x['natural_width'] for x in result.row_info)
        self.assertGreater(len({x['natural_width'] for x in result.row_info}), 1)
        self.assertEqual({r.shape[1] for r in result.rows}, {target})
        self.assertTrue(all(x['horizontal_scale'] == 1 for x in result.row_info))
        self.assertTrue(all(x['width'] == target == x['target_width'] for x in result.row_info))
        rows = result.rows[:-1] + [_add_final_barline(result.rows[-1])]
        widths = [r['width'] for p in layout_rows(rows, result.staff_spacing) for r in p]
        self.assertLess(max(widths) - min(widths), .01)

    def test_narrow_gaps_keep_every_source_pixel_and_do_not_stretch_a_dot(self):
        row = np.full((120, 200), 255, np.uint8)
        lines = np.arange(35, 86, 10)
        for y in lines:
            row[y, :] = 0
        for x in range(10, 191, 10):
            cv2.rectangle(row, (x, 92), (x + 6, 108), 0, -1)
        # This dot lies entirely inside the old ignored line band.
        cv2.circle(row, (98, 55), 1, 0, -1)
        out, inserts = justify_row(row, lines, 330)
        self.assertEqual(out.shape[1], 330)
        self.assertGreater(len(inserts), 1)
        pieces, end, added = [], 0, 0
        for item in inserts:
            x, n = item['source_x'], item['pixels']
            self.assertFalse(97 <= x <= 99)
            cut = x + added
            pieces.append(out[:, end:cut])
            end, added = cut + n, added + n
        pieces.append(out[:, end:])
        np.testing.assert_array_equal(np.hstack(pieces), row)

    def test_connected_annotation_uses_whole_row_fallback_without_clipping(self):
        result = assemble_score([score_with_uneven_measures(True)], bars_per_row=5, log=lambda _: None)
        target = max(x['natural_width'] for x in result.row_info)
        self.assertEqual({r.shape[1] for r in result.rows}, {target})
        stretched = [(row, info) for row, info in zip(result.rows, result.row_info)
                     if info['horizontal_scale'] > 1]
        self.assertTrue(stretched)
        for row, info in stretched:
            self.assertFalse(info['space_insertions'])
            top = int(round(staff_groups(row)[0][0]))
            self.assertGreater(np.max((row[:top] < 180).sum(axis=1)), target * .85)


if __name__ == '__main__':
    unittest.main()
