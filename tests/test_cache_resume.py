"""Interrupt/restart/reflow must preserve completed work and local corrections."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from tabextract import pipeline
from tabextract.cache import CacheStore, Checkpoint
from tabextract.pages import Segment, scan_segments
from tabextract.review import ReviewSession
from tabextract.support import CancelledError, time_bounds
from test_segment_review import frame
from test_delayed_score import FrameCapture


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cache = CacheStore(self.folder / 'cache')

    def test_safe_roundtrip_corruption_and_active_cleanup(self):
        original = self.folder / 'original.mp4'; original.write_bytes(b'keep')
        with self.cache.job({'kind': 'test'}) as checkpoint:
            checkpoint.save('data', {(1, 2): [Segment(1, 9, 3), np.arange(9).reshape(3, 3), None]})
            restored = checkpoint.load('data')
            self.assertEqual(restored[(1, 2)][0].end, 3)
            np.testing.assert_array_equal(restored[(1, 2)][1], np.arange(9).reshape(3, 3))
            self.assertEqual(self.cache.prune(clear=True), 0)
            checkpoint.file('data').write_bytes(b'truncated')
            self.assertIsNone(checkpoint.load('data'))
            with self.assertRaises(ValueError): checkpoint.save('../outside', 3)
        self.assertGreater(self.cache.prune(clear=True), 0)
        self.assertFalse(self.cache.entries())
        self.assertEqual(original.read_bytes(), b'keep')

    def test_expiry_and_size_limit_keep_recent_entries(self):
        for i in range(3):
            with self.cache.job({'kind': 'test', 'i': i}): pass
        entries = self.cache.entries()
        oldest = entries[0]
        checkpoint = Checkpoint(self.cache.root/oldest['key'])
        meta = checkpoint.load('entry'); meta['last_used'] = 0; checkpoint.save('entry', meta)
        self.cache.prune()
        self.assertEqual(len(self.cache.entries()), 2)
        self.cache.prune(max_bytes=1)
        self.assertFalse(self.cache.entries())

    def patch_pipeline(self):
        video = self.folder/'source.mp4'; video.write_bytes(b'synthetic-source')
        self.video, self.pdf = video, self.folder/'score.pdf'
        self.segments = [Segment(0, 10, 10), Segment(20, 30, 10), Segment(40, 50, 10)]
        mocks = []
        for target, kwargs in (
            ('inspect_video', dict(return_value=((0, 0, 480, 180), .9, frame()))),
            ('scan_segments', dict(return_value=(self.segments, 'bright'))),
            ('load_segment_frames', dict(side_effect=lambda source, region, segment, **kw: [frame(7+segment.first//10)]*3)),
            ('make_scroll_bridge', dict(return_value=None)),
        ):
            patcher = patch.object(pipeline, target, **kwargs)
            mocks.append(patcher.start()); self.addCleanup(patcher.stop)
        return mocks

    def run_score(self, **kwargs):
        return pipeline.run_pipeline(self.video, self.pdf, cache_store=self.cache, log=lambda _: None, **kwargs)

    def test_resume_after_second_segment_failure_then_reflow_without_analysis(self):
        locate, scan, load, bridge = self.patch_pipeline()
        normal = load.side_effect
        def interrupted(source, region, segment, **kw):
            if segment.first == 20: raise CancelledError('interrupted')
            return normal(source, region, segment, **kw)
        load.side_effect = interrupted
        with self.assertRaises(CancelledError): self.run_score()
        self.assertFalse(self.pdf.exists())
        locate.reset_mock(); scan.reset_mock(); load.reset_mock()
        load.side_effect = normal
        self.run_score()
        locate.assert_not_called(); scan.assert_not_called()
        self.assertEqual([c.args[2].first for c in load.call_args_list], [20, 40])
        load.reset_mock(); bridge.reset_mock()
        with patch('tabextract.stitch._horizontal_runs', side_effect=AssertionError('must reuse stitching')):
            summary = self.run_score(bars_per_row=2, dpi=150)
        load.assert_not_called(); bridge.assert_not_called()
        self.assertEqual(summary['bars_per_row'], 2)
        self.assertTrue(self.pdf.is_file())

    def test_local_override_survives_restart_but_failed_edit_does_not(self):
        self.patch_pipeline()
        review = ReviewSession()
        self.run_score(review_session=review)
        review.reprocess(2, 2.1, 2.9, 'bright', 1.3, log=lambda _: None)
        with patch.object(pipeline, 'export_rows', side_effect=RuntimeError('export failed')):
            with self.assertRaises(RuntimeError):
                review.reprocess(2, 2.1, 2.9, 'bright', 1.7, log=lambda _: None)
        restarted = ReviewSession()
        with patch.object(pipeline, 'load_segment_frames', side_effect=AssertionError('must reuse extraction')), \
             patch('tabextract.stitch._horizontal_runs', side_effect=AssertionError('must reuse edited stitching')):
            summary = self.run_score(review_session=restarted, bars_per_row=2)
        self.assertEqual(summary['segment_settings'][1]['detail_gain'], 1.3)
        np.testing.assert_array_equal(restarted.records[1]['page'], review.records[1]['page'])

    def test_changed_source_or_extraction_options_invalidate_cache(self):
        locate, scan, load, bridge = self.patch_pipeline()
        self.run_score()
        load.reset_mock()
        self.video.write_bytes(b'different-source')
        self.run_score()
        self.assertEqual(load.call_count, 3)
        load.reset_mock()
        self.run_score(sample_fps=3)
        self.assertEqual(load.call_count, 3)

    def test_partial_scan_restores_features_and_absolute_times(self):
        frames = [frame()] * 180
        cancel = threading.Event()
        with self.cache.job({'kind': 'scan'}) as checkpoint:
            def stop(p):
                if p > .45: cancel.set()
            with patch('tabextract.pages.cv2.VideoCapture', side_effect=lambda _: FrameCapture(frames)):
                with self.assertRaises(CancelledError):
                    scan_segments('fake', (0, 0, 480, 180), checkpoint=checkpoint,
                                  cancel_event=cancel, progress=stop, log=lambda _: None)
                saved = checkpoint.load('scan-progress')
                self.assertGreater(saved[0], 0)
                cancel.clear()
                resumed = scan_segments('fake', (0, 0, 480, 180), checkpoint=checkpoint, log=lambda _: None)
                fresh = scan_segments('fake', (0, 0, 480, 180), log=lambda _: None)
                self.assertEqual(resumed, fresh)
                ranged, _ = scan_segments('fake', (0, 0, 480, 180), time_range=(2, 4), log=lambda _: None)
                self.assertGreaterEqual(ranged[0].start, 2)
                self.assertLess(ranged[-1].end, 4)

    def test_time_validation(self):
        self.assertEqual(time_bounds(('1:02.5', '2:00'), 180), (62.5, 120))
        for values in ((-1, 2), (2, 1), ('nan', ''), ('1:80', ''), (0, 999)):
            with self.assertRaises(ValueError): time_bounds(values, 180)


if __name__ == '__main__': unittest.main()
