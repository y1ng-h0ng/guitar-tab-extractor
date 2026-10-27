"""Small-stroke safety, editing provenance, transaction rollback and undo."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import cv2
import numpy as np
from tabextract.clean import clean_frames
from tabextract.regions import apply_regions, validate_region
from tabextract.recovery import improves, recover_page, recover_bridge, sharp_stable_frames
from tabextract.support import CancelledError
from tabextract.stitch import assemble_score
from tabextract import pipeline
from tabextract.pages import Segment
from tabextract.review import ReviewSession
from tabextract.cache import CacheStore
from test_segment_review import frame


class PixelSafetyTests(unittest.TestCase):
    def test_rectangles_protect_priority_clipping_and_no_invented_pixels(self):
        base = np.full((30, 50), 255, np.uint8); base[8:12, 10:18] = 0
        evidence = base.copy(); evidence[16:18, 22:24] = 60
        record = dict(page=base, evidence_page=evidence, regions=[
            dict(kind='protect', bbox=[10, 8, 14, 10]), dict(kind='erase', bbox=[5, 5, 25, 20])])
        result = apply_regions(base, [(record, 0, 0)])
        self.assertEqual(result[9, 11], 0)
        self.assertEqual(result[16, 22], 60)
        self.assertEqual(result[15, 22], 255)
        self.assertTrue(np.array_equal(base[8:12,10:18], np.zeros((4,8))))
        shifted = apply_regions(np.full((30,50),255,np.uint8), [(record,-12,0)])
        self.assertEqual(shifted[9, 0], 0)
        for bbox in ([-1,0,10,2], [0,0,100,10], [1.,2,3,4]):
            with self.assertRaises(ValueError): validate_region('erase',bbox,base.shape)

    def test_duplicate_overlap_cannot_reintroduce_erased_ink(self):
        page = clean_frames([frame()]*3)
        # Same view gets deduplicated; editing its second copy must still apply.
        records = [dict(page=page, regions=[]), dict(page=page, regions=[dict(kind='erase',bbox=[265,260,65,60])])]
        result = assemble_score([page,page], page_records=records, log=lambda _:None)
        self.assertEqual(result.joins[0]['status'],'same_view')
        self.assertTrue((result.runs[0][260:320,265:330] == 255).all())
        records[0]['regions'] = [dict(kind='protect',bbox=[265,260,65,60])]
        protected = assemble_score([page,page], page_records=records, log=lambda _:None)
        np.testing.assert_array_equal(protected.runs[0][260:320,265:330], page[260:320,265:330])

    def test_recovery_rejects_blank_lost_dot_and_unsupported_texture(self):
        source = frame()
        page = clean_frames([source]*5)
        for candidate in (np.full_like(page,255), np.zeros_like(page)):
            self.assertFalse(improves(page,candidate,[source]*5,'bright')[0])
        lost = page.copy(); lost[405:430,438:465] = 255
        accepted, scores = improves(page,lost,[source]*5,'bright')
        self.assertFalse(accepted)
        self.assertLess(scores['minimum_component_preserved'], .98)

    def test_multisystem_overlap_keeps_earlier_protection(self):
        single = clean_frames([frame()]*3)
        page = np.vstack([single, np.full((60,single.shape[1]),255,np.uint8), single])
        bbox = [265,260,65,60]
        records = [dict(page=page,regions=[dict(kind='protect',bbox=bbox)]),
                   dict(page=page,regions=[dict(kind='erase',bbox=bbox)])]
        result = assemble_score([page,page],page_records=records,log=lambda _:None)
        self.assertEqual(len(result.runs),2)
        np.testing.assert_array_equal(result.runs[0][260:320,265:330],page[260:320,265:330])

    def test_evidenced_missing_glyph_is_accepted_and_retry_count_is_bounded(self):
        source = frame(); target = clean_frames([source]*5)
        damaged = target.copy(); damaged[265:320,265:330] = 255
        self.assertTrue(improves(damaged,target,[source]*5,'bright')[0])
        diagnostics = [dict(kind='faint_detail',bbox=[265,265,65,55])]
        with patch('tabextract.recovery.clean_frames') as cleaner:
            def clean(*args, **kwargs):
                kwargs['evidence_out']['page'] = target.copy()
                return target.copy()
            cleaner.side_effect = clean
            recovered, _, _, info = recover_page(damaged,damaged,[source]*5,'bright',1.,diagnostics,
                                                  lambda:[source]*13,log=lambda _:None)
            self.assertTrue(info['accepted']); self.assertEqual(cleaner.call_count,1)
            np.testing.assert_array_equal(recovered,target)
        with patch('tabextract.recovery.clean_frames',side_effect=clean) as cleaner:
            _,_,_,info = recover_page(target,target,[source]*5,'bright',1.,diagnostics,
                                      lambda:[source]*13,log=lambda _:None)
            self.assertFalse(info['accepted']); self.assertEqual(cleaner.call_count,2)

    def test_sharp_selection_keeps_temporal_majority(self):
        frames = [frame()]*10 + [cv2.GaussianBlur(frame(),(5,5),2)]*5
        selected = sharp_stable_frames(frames,'bright')
        self.assertGreaterEqual(len(selected),9)


class RegionTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name); self.video = folder/'video.mp4'; self.video.write_bytes(b'test')
        self.pdf = folder/'score.pdf'; self.cache = CacheStore(folder/'cache')
        self.review = ReviewSession()
        for name, kwargs in (
            ('inspect_video',dict(return_value=((0,0,480,180),1.,frame()))),
            ('scan_segments',dict(return_value=([Segment(0,10,10),Segment(20,30,10)],'bright'))),
            ('load_segment_frames',dict(return_value=[frame()]*5)),
            ('make_scroll_bridge',dict(return_value=None))):
            patcher = patch.object(pipeline,name,**kwargs); patcher.start(); self.addCleanup(patcher.stop)
        pipeline.run_pipeline(self.video,self.pdf,cache_store=self.cache,review_session=self.review,log=lambda _:None)

    def test_edit_export_undo_and_cache_restore_without_video_reads(self):
        original = self.review.records[1]['page'].copy()
        untouched = self.review.records[0]
        with patch.object(pipeline,'load_segment_frames',side_effect=AssertionError('unexpected video read')), \
             patch.object(pipeline,'make_scroll_bridge',side_effect=AssertionError('unexpected bridge read')):
            self.review.edit_regions(2,'erase',[265,260,65,60],log=lambda _:None)
            self.assertIs(self.review.records[0],untouched)
            self.assertTrue((self.review.records[1]['page'][260:320,265:330] == 255).all())
            restarted = ReviewSession()
            pipeline.run_pipeline(self.video,self.pdf,cache_store=self.cache,review_session=restarted,log=lambda _:None,overwrite=True)
            self.assertEqual(len(restarted.records[1]['regions']),1)
            restarted.edit_regions(2,undo=True,log=lambda _:None)
            np.testing.assert_array_equal(restarted.records[1]['page'],original)
            self.assertEqual(restarted.summary['region_edits'],[])

    def test_failed_export_and_invalid_rectangle_preserve_pdf_and_history(self):
        pdf = self.pdf.read_bytes(); before = self.review.records[1]
        with patch.object(pipeline,'export_rows',side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                self.review.edit_regions(2,'erase',[265,260,65,60],log=lambda _:None)
        self.assertEqual(self.pdf.read_bytes(),pdf); self.assertIs(self.review.records[1],before)
        with self.assertRaises(ValueError):
            self.review.edit_regions(2,'erase',[0,0,1440,540],log=lambda _:None)
        self.assertIs(self.review.records[1],before)


class BridgeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.page = clean_frames([frame()]*3)
        self.bridge = dict(page=self.page, first_shift=100, second_shift=100)

    def test_retry_requires_both_seams_and_runs_only_once(self):
        with patch('tabextract.stitch.find_horizontal_shift', return_value=None), \
             patch('tabextract.stitch._tracked_match', side_effect=[({}, (1, 1)), None]):
            retry = Mock(return_value=self.bridge)
            self.assertIsNone(recover_bridge(self.page,self.page,None,retry,log=lambda _:None))
            retry.assert_called_once_with()
        with patch('tabextract.stitch.find_horizontal_shift', return_value=None), \
             patch('tabextract.stitch._tracked_match', return_value=({}, (1, 1))):
            retry = Mock(return_value=self.bridge)
            result = recover_bridge(self.page,self.page,None,retry,log=lambda _:None)
            self.assertEqual(result['recovery'],'wider_transition')
            retry.assert_called_once_with()

    def test_failed_retry_preserves_conflicting_motion_and_valid_join_skips_retry(self):
        retry = Mock(return_value=None)
        with patch('tabextract.stitch.find_horizontal_shift', return_value={'duplicate':True}), \
             patch('tabextract.stitch._tracked_match', return_value=None):
            result = recover_bridge(self.page,self.page,self.bridge,retry,log=lambda _:None)
            self.assertIs(result,self.bridge)
            retry.assert_called_once_with()
            retry.reset_mock()
            self.assertIsNone(recover_bridge(self.page,self.page,None,retry,log=lambda _:None))
            retry.assert_not_called()
        with patch('tabextract.stitch._tracked_match', return_value=({}, (1, 1))):
            self.assertIs(recover_bridge(self.page,self.page,self.bridge,retry),self.bridge)
            retry.assert_not_called()

    def test_cancel_before_retry_does_not_read_video(self):
        stop = threading.Event(); stop.set(); retry = Mock()
        with patch('tabextract.stitch.find_horizontal_shift', return_value=None):
            with self.assertRaises(CancelledError):
                recover_bridge(self.page,self.page,None,retry,stop,log=lambda _:None)
        retry.assert_not_called()


if __name__ == '__main__': unittest.main()
