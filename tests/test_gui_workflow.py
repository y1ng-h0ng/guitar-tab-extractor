"""Exercise the real Tk controls and drop binding without showing a window."""

from pathlib import Path
import tempfile
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

    def close_root(self):
        for callback in self.root.tk.splitlist(self.root.tk.call('after', 'info')):
            self.root.after_cancel(callback)
        self.root.destroy()

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
            close.assert_called_once()
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

    def test_failed_download_never_calls_pipeline_and_releases_cache(self):
        self.app.video.set('https://example.com/video')
        with patch.object(self.app, 'launch', side_effect=lambda fn: fn()), \
                patch.object(self.app.source, 'resolve', side_effect=RuntimeError('链接获取失败')), \
                patch.object(self.app.source, 'close') as close, patch('gui.run_pipeline') as run:
            with self.assertRaisesRegex(RuntimeError, '链接获取失败'):
                self.app.start()
            run.assert_not_called()
            close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
