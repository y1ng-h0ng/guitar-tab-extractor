"""Real localhost downloads, cancellation, resource ownership and CLI dispatch."""

import contextlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main
from tabextract.download import VideoSource, default_output, safe_title, video_url, probe_video, download_error, _selection
from tabextract.cache import CacheStore
from tabextract.support import CancelledError


class LinkInputTests(unittest.TestCase):
    def test_bilibili_watchlater_and_part_override(self):
        url = video_url('https://www.bilibili.com/list/watchlater?oid=1&bvid=BV1LA411S7Zh')
        self.assertIn('/video/BV1LA411S7Zh', url)
        selected, flags = _selection(url, 3)
        self.assertIn('p=3', selected)
        self.assertIn('--no-playlist', flags)
        self.assertEqual(_selection('https://example.com/list', 2)[1], ['--yes-playlist', '--playlist-items', '2'])

    def test_probe_lists_parts_and_only_selected_formats(self):
        listing = {'entries': [{'title': 'first'}, {'title': 'second'}]}
        detail = {'title': 'second', 'duration': 12, 'width': 1280, 'height': 720,
                  'formats': [{'format_id': 'audio', 'vcodec': 'none'},
                              {'format_id': '720', 'width': 1280, 'height': 720, 'vcodec': 'avc1'}]}
        url = 'https://www.bilibili.com/video/BV123/?p=2'
        with patch('tabextract.download._probe_json', side_effect=[listing, detail]) as probe:
            info = probe_video(url)
            self.assertEqual(info['part'], 2)
            self.assertEqual([f['id'] for f in info['formats']], ['720'])
            self.assertIn('p=2', probe.call_args.args[0])
            self.assertNotIn('p=2', probe.call_args_list[0].args[0])
        with patch('tabextract.download._probe_json', return_value=detail) as probe:
            probe_video(url, part=1, previous=info)
            self.assertEqual(probe.call_count, 1)
            self.assertIn('p=1', probe.call_args.args[0])
        with self.assertRaises(ValueError): probe_video(url, part=3, previous=info)

    def test_errors_distinguish_access_network_and_platform(self):
        for detail, kind in [('Sign in required', 'login'), ('HTTP Error 404', 'unavailable'),
                             ('Connection timed out', 'network'), ('Unsupported URL', 'unsupported'),
                             ('HTTP Error 403 Forbidden', 'platform'), ('requested format unavailable', 'quality')]:
            self.assertEqual(download_error(detail).kind, kind)
        self.assertNotIn('secret', str(download_error('HTTP 404 https://example.com/?secret=123')))

    def test_share_text_signed_urls_and_local_paths(self):
        url = 'https://example.com/watch?v=abc&token=x%2Fy'
        self.assertEqual(video_url('分享这首吉他曲 “' + url + '”，复制后打开'), url)
        self.assertEqual(video_url("'" + url + "'"), url)
        self.assertEqual(video_url('https://example.com/谱面.mp4'), 'https://example.com/谱面.mp4')
        self.assertIsNone(video_url(r'C:\视频\夏日.mp4'))
        for invalid in ('ftp://host/video', 'https://', 'https://a.com https://b.com',
                        'https://user:password@example.com/video'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                video_url(invalid)
        self.assertEqual(default_output('a.mp4'), 'a_吉他谱.pdf')
        self.assertEqual(Path(default_output(url)).parent, Path.cwd() / 'output')
        self.assertNotEqual(default_output(url), default_output(url + 'z'))

    def test_title_cannot_escape_owned_directory(self):
        for title in ('../../my:file', 'CON', 'NUL.txt', '...', 'a' * 79 + '.name'):
            result = safe_title(title)
            self.assertEqual(Path(result).name, result)
            self.assertTrue(result and not result.endswith(('.', ' ')))
            self.assertNotIn(':', result)
        self.assertEqual(safe_title('夏日潜行'), '夏日潜行')

    def test_cli_resolves_before_pipeline_and_always_cleans_up(self):
        summary = dict(video_pages=1, score_rows=1, pdf_pages=1,
                       output_pdf='song.pdf', warnings=[])
        url = 'https://example.com/song'
        for error in (None, RuntimeError('download failed')):
            with patch('main.VideoSource') as source, patch('main.run_pipeline', return_value=summary) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                source.return_value.resolve.return_value = 'download.mp4'
                source.return_value.resolve.side_effect = error
                result = main.main([url, '-o', 'song.pdf'])
                self.assertEqual(result, 1 if error else 0)
                source.return_value.close.assert_called_once()
                if error:
                    run.assert_not_called()
                else:
                    self.assertEqual(run.call_args.args, ('download.mp4', 'song.pdf'))
                    self.assertEqual(run.call_args.kwargs['source_url'], url)
                    self.assertIsNone(run.call_args.kwargs['bars_per_row'])
        with patch('main.VideoSource') as source, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main.main([url, '--dpi', '0']), 1)
            source.return_value.resolve.assert_not_called()


class LocalHttpDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        path = cls.folder / 'score.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
        if not writer.isOpened():
            cls.temp.cleanup()
            raise unittest.SkipTest('MP4 encoder unavailable')
        frame = np.full((240, 320, 3), 240, np.uint8)
        cv2.putText(frame, 'GUITAR 123', (15, 120), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        for _ in range(10): writer.write(frame)
        writer.release()
        (cls.folder / 'invalid.mp4').write_bytes(b'not a video')
        (cls.folder / 'watch.html').write_text(
            '<html><head><title>Guitar tutorial</title></head>'
            '<body><video controls src="score.mp4"></video></body></html>', encoding='utf-8')
        cls.stalled = threading.Event(); cls.release = threading.Event()
        cls.requests = []

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                cls.requests.append(self.path)
                if self.path == '/stall.mp4':
                    cls.stalled.set()
                    cls.release.wait(15)
                    return
                super().do_GET()

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=cls.temp.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.release.set(); cls.server.shutdown(); cls.server.server_close()
        cls.thread.join(); cls.temp.cleanup()

    def test_download_cache_and_owned_cleanup_preserve_original(self):
        source = VideoSource(); self.addCleanup(source.close)
        progress = []
        path = Path(source.resolve(self.base + '/score.mp4', log=lambda _: None,
                                   progress_cb=lambda pct, _: progress.append(pct)))
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), (self.folder / 'score.mp4').read_bytes())
        requests = len(self.requests)
        self.assertEqual(source.resolve(self.base + '/score.mp4'), str(path))
        self.assertEqual(len(self.requests), requests)
        self.assertTrue(progress and max(progress) == 100)
        owned = path.parent
        source.close()
        self.assertFalse(owned.exists())
        self.assertTrue((self.folder / 'score.mp4').exists())

    def test_persistent_download_survives_restart_and_cleanup_skips_live_source(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheStore(directory)
            source = VideoSource(cache)
            try:
                path = Path(source.resolve(self.base+'/score.mp4', log=lambda _: None))
                self.assertEqual(cache.prune(clear=True), 0)
                source.close()
                self.assertTrue(path.exists())
                restarted = VideoSource(cache)
                try:
                    with patch('tabextract.download.download_video', side_effect=AssertionError('must reuse download')):
                        self.assertEqual(restarted.resolve(self.base+'/score.mp4', log=lambda _: None), str(path))
                finally: restarted.close()
                self.assertGreater(cache.prune(clear=True), 0)
                self.assertFalse(path.exists())
                self.assertTrue((self.folder/'score.mp4').exists())
            finally: source.close()

    def test_probe_real_http_video_without_downloading(self):
        info = probe_video(self.base+'/watch.html')
        self.assertTrue(info['title'])
        self.assertEqual(info['part'], 1)

    def test_invalid_video_is_rejected_and_temporary_directory_removed(self):
        source = VideoSource(); self.addCleanup(source.close)
        with self.assertRaisesRegex(RuntimeError, '无法读取视频画面'):
            source.resolve(self.base + '/invalid.mp4', log=lambda _: None)
        self.assertIsNone(source.temp)
        self.assertIsNone(source.path)

    def test_webpage_video_extraction_and_http_error(self):
        source = VideoSource(); self.addCleanup(source.close)
        path = Path(source.resolve(self.base + '/watch.html', log=lambda _: None))
        self.assertEqual(path.read_bytes(), (self.folder / 'score.mp4').read_bytes())
        owned = path.parent
        with self.assertRaisesRegex(RuntimeError, '视频获取失败'):
            source.resolve(self.base + '/missing.mp4', log=lambda _: None)
        self.assertFalse(owned.exists())
        self.assertIsNone(source.temp)

    def test_cancel_during_stalled_network_does_not_wait_for_timeout(self):
        source = VideoSource(); self.addCleanup(source.close)
        cancel = threading.Event(); failures = []
        def run():
            try: source.resolve(self.base + '/stall.mp4', cancel_event=cancel, log=lambda _: None)
            except BaseException as error: failures.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(self.stalled.wait(20), 'Downloader did not reach HTTP server')
            start = time.monotonic(); cancel.set(); worker.join(8)
            self.assertFalse(worker.is_alive(), 'Cancellation left downloader running')
            self.assertLess(time.monotonic() - start, 8)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], CancelledError)
            self.assertIsNone(source.temp)
        finally:
            cancel.set(); self.release.set(); worker.join(20)


if __name__ == '__main__':
    unittest.main()
