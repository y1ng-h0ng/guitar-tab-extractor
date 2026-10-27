"""Local edits must never re-clean other segments or damage a prior export."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tabextract import pipeline
from tabextract.clean import clean_frames, observed_staff, infer_polarity, detail_issues
from tabextract.pages import Segment
from tabextract.review import ReviewSession
from tabextract.stitch import assemble_score
from tabextract.support import CancelledError


def frame(number=7, faint=False):
    gray = np.full((180, 480), 15, np.uint8)
    for y in range(65, 116, 10):
        cv2.line(gray, (10, y), (470, y), 35 if faint else 220, 1)
    for x in (30, 170, 310, 455):
        cv2.line(gray, (x, 65), (x, 115), 35 if faint else 220, 1)
    cv2.putText(gray, str(number), (90, 103), cv2.FONT_HERSHEY_SIMPLEX, .45,
                35 if faint else 220, 1)
    cv2.line(gray, (130, 110), (130, 146), 35 if faint else 220, 1)
    cv2.circle(gray, (150, 139), 1, 35 if faint else 220, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class LocalReviewTests(unittest.TestCase):
    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video = self.folder/'source.mp4'; self.video.touch()
        self.pdf = self.folder/'score.pdf'
        self.session = ReviewSession()
        self.segments = [Segment(0, 10, 10), Segment(20, 30, 10), Segment(40, 50, 10)]
        self.loader = self.start_patch(patch.object(pipeline, 'load_segment_frames',
            side_effect=lambda source, region, segment, **kw: [frame(7+segment.first//10)]*5))
        self.start_patch(patch.object(pipeline, 'inspect_video', return_value=((0, 0, 480, 180), .9, frame())))
        self.scanner = self.start_patch(patch.object(pipeline, 'scan_segments', return_value=(self.segments, 'bright')))
        self.bridge = self.start_patch(patch.object(pipeline, 'make_scroll_bridge', return_value=None))
        pipeline.run_pipeline(self.video, self.pdf, review_session=self.session,
                              generate_report=True, log=lambda _: None)
        self.loader.reset_mock(); self.scanner.reset_mock(); self.bridge.reset_mock()

    def test_only_selected_frames_reload_and_other_pages_are_identical(self):
        originals = [r['page'].copy() for r in self.session.records]
        identities = [id(r) for r in self.session.records]
        summary = self.session.reprocess(2, 2.1, 2.9, 'bright', 1.3, log=lambda _: None)
        self.scanner.assert_not_called()
        self.loader.assert_called_once()
        segment = self.loader.call_args.args[2]
        self.assertEqual((segment.first, segment.last), (21, 29))
        for i in (0, 2):
            self.assertEqual(id(self.session.records[i]), identities[i])
            np.testing.assert_array_equal(self.session.records[i]['page'], originals[i])
        self.assertEqual(self.session.records[1]['detail_gain'], 1.3)
        self.assertEqual(summary['output_pdf'], str(self.pdf))
        report = json.loads(self.pdf.with_suffix('.report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['segment_settings'][1]['detail_gain'], 1.3)
        self.assertFalse(list(self.folder.glob('.tabextract-review-*')))
        self.assertEqual(self.bridge.call_count, 2)
        self.assertTrue(all(c.kwargs['detail_gain'] == 1.15 for c in self.bridge.call_args_list))
        self.bridge.reset_mock()
        # Default restoration updates only this segment and recomputes its two bridges.
        self.session.reprocess(2, 2, 3, 'bright', 1, log=lambda _: None)
        self.assertEqual(self.bridge.call_count, 2)

    def test_failure_and_cancellation_preserve_current_pdf_report_and_state(self):
        old_pdf = self.pdf.read_bytes()
        old_report = self.pdf.with_suffix('.report.json').read_bytes()
        old_record = self.session.records[1]
        with patch.object(pipeline, 'export_rows', side_effect=RuntimeError('export failed')):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                self.session.reprocess(2, 2, 3, 'bright', 1.2, log=lambda _: None)
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(CancelledError):
            self.session.reprocess(2, 2, 3, 'bright', 1.2, log=lambda _: None, cancel_event=cancel)
        self.assertIs(self.session.records[1], old_record)
        self.assertEqual(self.pdf.read_bytes(), old_pdf)
        self.assertEqual(self.pdf.with_suffix('.report.json').read_bytes(), old_report)
        self.assertFalse(list(self.folder.glob('.tabextract-review-*')))

    def test_report_publish_failure_rolls_back_pdf_and_state(self):
        import os
        old_pdf = self.pdf.read_bytes(); old_record = self.session.records[1]
        original_replace = os.replace
        def replace(source, destination):
            if Path(destination) == self.pdf.with_suffix('.report.json'):
                raise PermissionError('report locked')
            return original_replace(source, destination)
        with patch('tabextract.review.os.replace', side_effect=replace):
            with self.assertRaisesRegex(PermissionError, 'report locked'):
                self.session.reprocess(2, 2, 3, 'bright', 1.2, log=lambda _: None)
        self.assertEqual(self.pdf.read_bytes(), old_pdf)
        self.assertIs(self.session.records[1], old_record)

    def test_invalid_settings_do_not_load_or_modify_anything(self):
        for start, end, polarity, gain in ((-1, 3, 'bright', 1), (2, 4, 'bright', 1),
                                         (2.9, 2.1, 'bright', 1), (2, 3, 'auto', float('nan')),
                                         (2, 3, 'other', 1), (2, 3, 'bright', 3)):
            with self.assertRaises(ValueError):
                self.session.reprocess(2, start, end, polarity, gain, log=lambda _: None)
        self.loader.assert_not_called()

    def test_empty_local_result_cannot_drop_a_previously_visible_segment(self):
        old = self.pdf.read_bytes(); record = self.session.records[1]
        with patch.object(pipeline, 'clean_frames', return_value=np.full((540, 1440), 255, np.uint8)):
            with self.assertRaisesRegex(RuntimeError, '原片段和 PDF 已保留'):
                self.session.reprocess(2, 2, 3, 'dark', 1, log=lambda _: None)
        self.assertEqual(self.pdf.read_bytes(), old)
        self.assertIs(self.session.records[1], record)

    def test_skipped_segment_can_be_recovered_without_renumbering_other_ids(self):
        self.loader.side_effect = lambda source, region, segment, **kw: [
            np.full((180, 480, 3), 15, np.uint8) if segment.first == 20 else frame(7)]*5
        pipeline.run_pipeline(self.video, self.pdf, review_session=self.session, log=lambda _: None, overwrite=True)
        self.assertFalse(self.session.records[1]['valid'])
        self.assertEqual([v['segment_id'] for v in self.session.summary['source_views']], [1, 3])
        last = self.session.records[2]
        self.loader.side_effect = lambda *a, **kw: [frame(12)]*5
        summary = self.session.reprocess(2, 2, 3, 'bright', 1, log=lambda _: None)
        self.assertEqual([v['segment_id'] for v in summary['source_views']], [1, 2, 3])
        self.assertIs(self.session.records[2], last)
        self.assertFalse(any(i['kind'] == 'missing_view' for i in summary['issues']))

    def test_nonadjacent_transition_is_reused(self):
        self.scanner.return_value = (self.segments+[Segment(60, 70, 10)], 'bright')
        pipeline.run_pipeline(self.video, self.pdf, review_session=self.session, log=lambda _: None, overwrite=True)
        self.bridge.reset_mock()
        self.session.reprocess(2, 2.1, 2.9, 'bright', 1, log=lambda _: None)
        self.assertEqual(self.bridge.call_count, 2)
        self.assertEqual([(c.args[2].first, c.args[3].first) for c in self.bridge.call_args_list], [(0, 20), (20, 40)])

    def test_issues_point_to_stable_segment_and_relevant_output_rows(self):
        report = self.session.summary
        self.assertTrue(report['issues'])
        for issue in report['issues']:
            record = self.session.records[issue['segment_id']-1]
            self.assertLessEqual(record['original'].start, issue['time_seconds'])
            self.assertGreaterEqual(record['original'].end, issue['time_seconds'])
            x, y, w, h = issue['bbox']; ih, iw = record['page'].shape
            self.assertTrue(0 <= x < iw and 0 <= y < ih and x+w <= iw+1 and y+h <= ih+1)
        self.assertTrue(all(r['source_pages'] for r in report['rows']))


class FaintDetailAndOverlapTests(unittest.TestCase):
    def test_review_flags_missing_stroke_but_not_retained_glyph_fringe(self):
        page = np.full((180, 480), 255, np.uint8)
        evidence = np.zeros(page.shape, np.float32)
        page[78:82, 101:110] = 0
        evidence[78:82, 100:110] = 25
        evidence[78:82, 200:210] = 25
        issues = detail_issues(page, evidence, evidence*2, [np.arange(65, 116, 10)], 1)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['bbox'], [200, 78, 10, 4])

    def test_dark_translucent_staff_notes_stems_and_dot_keep_real_gaps(self):
        for polarity in ('bright', 'dark'):
            image = frame(faint=True)
            image[125:131, 128:133] = 15
            if polarity == 'dark': image = 255-image
            self.assertEqual(infer_polarity(image), polarity)
            self.assertTrue(observed_staff(image, polarity))
            output = clean_frames([image]*9, polarity, upscale=1)
            self.assertLess(int(output[139, 150]), 100)
            self.assertLess(int(output[140, 130]), 100)
            self.assertTrue((output[126:130, 129:132] > 245).all())
            self.assertGreater(np.count_nonzero(output[92:104, 90:102] < 150), 8)

    def test_interior_overlap_survives_a_damaged_prefix_without_losing_half_bar(self):
        score = np.full((180, 1600), 255, np.uint8)
        for y in range(65, 116, 10): cv2.line(score, (0, y), (1599, y), 0, 1)
        for i, x in enumerate(range(25, 1500, 70)):
            if i % 2 == 0: cv2.line(score, (x, 65), (x, 115), 0, 1)
            cv2.putText(score, str((i*7)%23), (x+10, 99), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
            cv2.line(score, (x+10, 110), (x+10, 138+(i%3)*4), 0, 1)
        first, second = score[:, :1000].copy(), score[:, 400:1400].copy()
        # Prefix is damaged, but the common bar and interior notes are intact.
        second[:65, :140] = 255; second[80:110, :140] = 255
        result = assemble_score([first, second], log=lambda _: None)
        self.assertEqual(result.joins[0]['status'], 'joined')
        self.assertEqual(result.joins[0]['shift'], 400)
        np.testing.assert_array_equal(result.runs[0], score[:, :1400])


if __name__ == '__main__': unittest.main()
