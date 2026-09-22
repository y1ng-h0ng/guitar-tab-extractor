"""Prefer an extra measure and bounded compression over long stretched bars."""

import unittest
import cv2
import numpy as np

from tabextract.rowlayout import borrow_measure, choose_auto_breaks, compact_row
from tabextract.stitch import assemble_score
from test_auto_layout import units_with_widths
from test_uniform_rows import score_with_uneven_measures


class CompactLayoutTests(unittest.TestCase):
    def test_auto_adds_a_measure_instead_of_stretching_four_long_bars(self):
        # Four bars fill only 86% of the nominal line; five need 8% compression.
        units = units_with_widths([160] * 20)
        groups = choose_auto_breaks(units, [False] * 19, 10)
        self.assertEqual([b - a for a, b in groups], [5, 5, 5, 5])

    def test_manual_can_borrow_one_without_splitting_tie_or_creating_short_tail(self):
        units = units_with_widths([120] * 5 + [230] * 5)
        groups = [(0, 5), (5, 10)]
        self.assertEqual(borrow_measure(units, [False] * 10, groups, 800, 5),
                         [(0, 6), (6, 10)])
        crossings = [False] * 10; crossings[5] = True
        self.assertEqual(borrow_measure(units, crossings, groups, 800, 5), groups)
        tail = [(0, 5), (5, 8)]
        self.assertEqual(borrow_measure(units, [False] * 10, tail, 800, 5), tail)
        too_wide = units_with_widths([120] * 5 + [400] * 5)
        self.assertEqual(borrow_measure(too_wide, [False] * 10, groups, 800, 5), groups)

    def test_compaction_preserves_digits_dots_stems_and_ties_pixel_for_pixel(self):
        row = np.full((160, 800), 255, np.uint8)
        lines = np.arange(45, 96, 10)
        for y in lines:
            cv2.line(row, (10, y), (790, y), 0, 1)
        for x in (40, 290, 540):
            cv2.putText(row, '8', (x, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, 0, 1)
            cv2.line(row, (x + 8, 87), (x + 8, 138), 0, 1)
            cv2.circle(row, (x + 36, 65), 1, 0, -1)
            cv2.ellipse(row, (x + 85, 38), (24, 9), 0, 180, 360, 0, 1)
        out, removals = compact_row(row, lines, 730)
        self.assertEqual(out.shape[1], 730)
        self.assertTrue(removals)
        keep = np.ones(row.shape[1], bool)
        nonstaff = row < 245
        nonstaff[lines] = False
        for removal in removals:
            x, n = removal['source_x'], removal['pixels']
            self.assertFalse(nonstaff[:, x:x + n].any())
            keep[x:x + n] = False
        np.testing.assert_array_equal(out, row[:, keep])
        self.assertEqual(int(nonstaff[:, keep].sum()), int(nonstaff.sum()))
        # Nothing may be cut through a continuous annotation.
        row[20, :] = 0
        untouched, removals = compact_row(row, lines, 730)
        self.assertFalse(removals)
        np.testing.assert_array_equal(untouched, row)

    def test_auto_and_manual_keep_all_measures_and_limit_compression(self):
        for target in (None, 5):
            with self.subTest(target=target):
                result = assemble_score([score_with_uneven_measures()], bars_per_row=target, log=lambda _: None)
                self.assertEqual(result.measures, 13)
                self.assertEqual(sum(r['units'] for r in result.row_info), 13)
                self.assertEqual(len({row.shape[1] for row in result.rows}), 1)
                self.assertTrue(any(r['width'] < r['natural_width'] for r in result.row_info))
                self.assertTrue(all(r['width'] / r['natural_width'] >= .85 for r in result.row_info))
                self.assertTrue(all(r['horizontal_scale'] >= .85 for r in result.row_info))


if __name__ == '__main__':
    unittest.main()
