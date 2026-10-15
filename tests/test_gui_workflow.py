"""Exercise the real Tk controls and drop binding without showing a window."""

from pathlib import Path
import tempfile
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gui


class GuiWorkflowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = gui.tk.Tk()
        except gui.tk.TclError as error:
            self.skipTest(f'Tk display unavailable: {error}')
        self.root.withdraw()
        self.addCleanup(self.close_root)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / '测试 视频 {第一首}.MP4'
        self.video.touch()
        self.app = gui.App(self.root)
        self.addCleanup(self.app.source.close)
        self.addCleanup(self.app.close_review)

    def close_root(self):
        self.root.update_idletasks()
        for callback in self.root.tk.splitlist(self.root.tk.call('after', 'info')):
            self.root.after_cancel(callback)
        self.root.destroy()
        # Destroyed Tk roots from earlier tests must be collected on this
        # thread, before later downloader tests start their worker threads.
        self.app = None
        gc.collect()

    def drop(self, *paths):
        # Convert to an actual Tcl list string, as Explorer's native drop does.
        data = self.root.tk.call('format', '%s', tuple(map(str, paths)))
        return self.app.drop_video(SimpleNamespace(data=data))

    def test_drag_to_one_click_uses_auto_or_optional_manual_count(self):
        if gui.TkinterDnD is None:
            self.skipTest('tkinterdnd2 not installed')
        self.assertTrue(self.app.drop_enabled)
        self.assertTrue(self.app.video_entry.dnd_bind('<<Drop>>'))
        self.assertEqual(self.app.bars.get(), '自动')
        self.assertEqual(self.drop(self.video), 'copy')
        self.assertEqual(self.app.video.get(), str(self.video))
        self.assertEqual(self.app.output.get(), str(self.video.with_name(self.video.stem + '_吉他谱.pdf')))
        for value, expected in (('自动', None), ('', None), ('5', 5)):
            self.app.bars.set(value)
            with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), patch('gui.run_pipeline') as run:
                self.app.run_btn.invoke()
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs['bars_per_row'], expected)
                self.assertEqual(run.call_args.args[0], str(self.video))
                self.assertIsNone(run.call_args.kwargs['region'])

    def test_replacing_video_resets_region_and_tracks_default_output(self):
        self.drop(self.video)
        self.app.region = (1, 2, 300, 200)
        self.app.region_source = str(self.video)
        other = self.video.with_name('另一首.mp4'); other.touch()
        self.assertEqual(self.drop(other), 'copy')
        self.assertIsNone(self.app.region)
        self.assertIsNone(self.app.region_source)
        self.assertEqual(self.app.output.get(), str(other.with_name('另一首_吉他谱.pdf')))
        self.app.output.set(str(other.with_name('自定.pdf')))
        self.drop(self.video)
        self.assertEqual(self.app.output.get(), str(other.with_name('自定.pdf')))

    def test_invalid_or_busy_drops_do_not_replace_selected_video(self):
        self.drop(self.video)
        for paths in ((self.video, self.video), (self.video.parent,),
                      (self.video.with_suffix('.txt'),)):
            self.assertEqual(self.drop(*paths), 'refuse_drop')
            self.assertEqual(self.app.video.get(), str(self.video))
        self.app.worker = Mock(is_alive=Mock(return_value=True))
        self.assertEqual(self.drop(self.video), 'refuse_drop')
        with patch('gui.run_pipeline') as run:
            self.app.start()
            run.assert_not_called()

    def test_pasted_url_tracks_output_and_preserves_preview_region(self):
        self.drop(self.video)
        url = 'https://example.com/watch?v=123'
        self.app.video.set('分享谱子：' + url + ' 复制链接')
        self.assertEqual(self.app.output.get(), gui.default_output(url))
        self.assertEqual(self.app.valid_video(), url)
        region = (1, 2, 300, 200)
        with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), \
                patch.object(self.app.source, 'resolve', return_value=str(self.video)) as resolve, \
                patch('gui.inspect_video', return_value=(region, {}, 'frame')):
            self.app.preview()
            self.assertEqual(resolve.call_args.args, (url,))
            self.assertEqual(self.app.events.get_nowait(), ('preview', ('frame', region, url)))
        self.app.apply_region(region, url)
        with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), \
                patch.object(self.app.source, 'resolve', return_value=str(self.video)), \
                patch.object(self.app.source, 'close') as close, patch('gui.run_pipeline') as run:
            self.app.start()
            self.assertEqual(run.call_args.args[0], str(self.video))
            self.assertEqual(run.call_args.kwargs['region'], region)
            self.assertEqual(run.call_args.kwargs['source_url'], url)
            # Active review needs the downloaded source for local reprocessing.
            close.assert_not_called()
            self.assertIs(run.call_args.kwargs['review_session'], self.app.review)
        self.app.video.set('https://example.com/another')
        self.assertIsNone(self.app.region)
        self.assertEqual(self.app.output.get(), gui.default_output(self.app.video.get()))

    def test_partial_url_typing_and_custom_output_are_safe(self):
        for text in ('h', 'https:', 'https://', 'https://example.com'):
            self.app.video.set(text)
        self.assertEqual(self.app.output.get(), gui.default_output('https://example.com'))
        self.app.output.set('自定义.pdf')
        self.app.video.set('https://example.com/new')
        self.assertEqual(self.app.output.get(), '自定义.pdf')

    def test_link_selection_range_and_restore_reach_pipeline(self):
        url = 'https://example.com/list'
        self.app.video.set(url)
        metadata = dict(title='第二段', duration=30., width=1280, height=720,
                        part=2, parts=[{'index': 1}, {'index': 2}])
        self.app.apply_link_options(metadata, '720')
        self.app.range_start.set('0:05'); self.app.range_end.set('15')
        with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), \
                patch.object(self.app.source, 'resolve', return_value=str(self.video)) as resolve, \
                patch('gui.run_pipeline') as run:
            self.app.start()
            self.assertEqual(resolve.call_args.kwargs['part'], 2)
            self.assertEqual(resolve.call_args.kwargs['format_id'], '720')
            self.assertEqual(run.call_args.kwargs['time_range'], (5, 15))
            self.assertIs(run.call_args.kwargs['cache_store'], self.app.cache)
        request = dict(source=url, out_pdf='恢复.pdf', sample_fps=3, dpi=150,
                       bars_per_row=5, region=[0, 0, 300, 100], polarity='bright',
                       time_range=[5, 15], source_options={'part': 2, 'format_id': '720'})
        self.app.restore_request(request)
        self.assertEqual(self.app.bars.get(), '5')
        self.assertEqual(self.app.selected_range(), (5, 15))
        self.assertEqual(self.app.link_options['part'], 2)
        self.assertEqual(self.app.region_source, url)

    def test_link_dialog_failed_part_fetch_does_not_mix_old_formats(self):
        from tabextract.source_ui import LinkDialog
        info = dict(title='分 P 1', duration=30., width=1280, height=720, part=1,
            parts=[{'index': 1, 'title': 'first'}, {'index': 2, 'title': 'second'}],
            formats=[{'id': '720', 'label': '720p'}])
        win = LinkDialog(self.app, info, None)
        self.addCleanup(win.destroy)
        win.part.current(1)
        win.busy(True); win.busy(False)
        self.assertEqual(win.part.current(), 0)
        self.root.update_idletasks()
        self.assertLessEqual(win.apply_btn.winfo_rooty() + win.apply_btn.winfo_height(),
                             win.winfo_rooty() + win.winfo_height())

    def test_failed_download_never_calls_pipeline_and_releases_cache(self):
        self.app.video.set('https://example.com/video')
        with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), \
                patch.object(self.app.source, 'resolve', side_effect=RuntimeError('链接获取失败')), \
                patch.object(self.app.source, 'close') as close, patch('gui.run_pipeline') as run:
            with self.assertRaisesRegex(RuntimeError, '链接获取失败'):
                self.app.start()
            run.assert_not_called()
            close.assert_called_once()

    def test_review_navigates_issue_rows_and_dispatches_only_selected_segment(self):
        import numpy as np
        from tabextract.pages import Segment
        self.app.review.records = [dict(id=1, original=Segment(0, 10, 10),
            segment=Segment(0, 10, 10), polarity='bright', detail_gain=1.,
            page=np.full((180, 480), 255, np.uint8), valid=True)]
        report = dict(pdf_pages=1, score_rows=1, video_pages=1, warnings=[],
            source_views=[dict(page=1, segment_id=1)], rows=[dict(source_pages=[1])],
            issues=[dict(segment_id=1, message='测试接缝', time_seconds=.4, bbox=[10, 20, 30, 40])])
        self.app.review.summary = self.app.summary = report
        self.app.review.rows = [np.full((100, 500), 255, np.uint8)]
        self.app.review.config = dict(polarity='bright', video_path=str(self.video))
        self.app.show_report()
        win = self.app.review_window
        win.tree.selection_set('segment-1-0'); win.select()
        self.assertEqual(win.time.get(), '0.400')
        self.assertEqual(win.description.get(), '测试接缝')
        self.assertEqual(win.row.get(), '1')
        self.assertIsNotNone(win.photo)
        win.show_row(); self.assertEqual(win.mode.get(), '生成谱行')
        with patch.object(self.app, 'reprocess_segment') as process:
            win.gain.set('1.4'); win.apply_btn.invoke()
            process.assert_called_once_with(1, 0., 1., 'bright', 1.4)
        self.app.close_review()
        self.assertFalse(self.app.review.records)
        self.assertIsNone(self.app.review.summary)


if __name__ == '__main__':
    unittest.main()
