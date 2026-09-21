"""Automatic layout adapts to density while covering every measure in order."""

import contextlib
import io
import unittest
from unittest.mock import patch

import main
from tabextract.rowlayout import choose_auto_breaks
from tabextract.stitch import assemble_score
from tabextract.support import parse_bars_per_row
from test_uniform_rows import score_with_uneven_measures


def units_with_widths(widths):
    units, x = [], 0
    for width in widths:
        units.append((None, None, x, x + width))
        x += width
    return units


class AutoLayoutTests(unittest.TestCase):
    def test_auto_and_manual_option_validation(self):
        for value in (None, '', '  ', 'auto', ' AUTO ', '自动'):
            self.assertIsNone(parse_bars_per_row(value))
        for value in (1, 5, 8, ' 5 '):
            self.assertEqual(parse_bars_per_row(value), int(value))
        for value in (0, 9, True, 5.0, '2.5', 'nine'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_bars_per_row(value)

    def test_density_changes_count_and_resolution_does_not(self):
        sparse = choose_auto_breaks(units_with_widths([100] * 24), [False] * 23, 10)
        dense = choose_auto_breaks(units_with_widths([240] * 24), [False] * 23, 10)
        self.assertLess(len(sparse), len(dense))
        widths = [100, 120, 90, 230, 260, 180, 80, 100] * 4 + [140]
        groups = choose_auto_breaks(units_with_widths(widths), [False] * 32, 10)
        self.assertEqual(groups, choose_auto_breaks(
            units_with_widths([w * 3 for w in widths]), [False] * 32, 30))
        self.assertEqual([i for a, b in groups for i in range(a, b)], list(range(33)))
        self.assertTrue(all(1 <= b - a <= 8 for a, b in groups))
        self.assertEqual(choose_auto_breaks([], [], 10), [])

    def test_crossing_mark_moves_a_break_without_losing_measures(self):
        units = units_with_widths([150] * 20)
        crossings = [False] * 19
        baseline = choose_auto_breaks(units, crossings, 10)
        boundary = baseline[0][1]
        crossings[boundary - 1] = True
        groups = choose_auto_breaks(units, crossings, 10)
        self.assertNotIn(boundary, [b for _, b in groups[:-1]])
        self.assertEqual([i for a, b in groups for i in range(a, b)], list(range(20)))

    def test_assembly_defaults_to_auto_and_keeps_manual_override(self):
        score = score_with_uneven_measures()
        auto = assemble_score([score], log=lambda _: None)
        manual = assemble_score([score], bars_per_row=5, log=lambda _: None)
        self.assertEqual(auto.measures, manual.measures)
        self.assertEqual(sum(r['units'] for r in auto.row_info), auto.measures)
        self.assertEqual(len({r.shape[1] for r in auto.rows}), 1)
        self.assertTrue(all(r['target_units'] is None for r in auto.row_info))
        self.assertTrue(all(r['target_units'] == 5 for r in manual.row_info))
        self.assertNotEqual([r['units'] for r in auto.row_info],
                            [r['units'] for r in manual.row_info])

    def test_cli_default_auto_explicit_auto_and_manual(self):
        summary = dict(video_pages=1, score_rows=1, pdf_pages=1,
                       output_pdf='example.pdf', warnings=[])
        for args, expected in (([], None), (['--bars-per-row', 'auto'], None),
                               (['--bars-per-row', '5'], 5)):
            with patch('main.run_pipeline', return_value=summary) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(['example.mp4', *args]), 0)
                self.assertEqual(run.call_args.kwargs['bars_per_row'], expected)
        with patch('main.run_pipeline') as run, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.main(['example.mp4', '--bars-per-row', '9'])
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
