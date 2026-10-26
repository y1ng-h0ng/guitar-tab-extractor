import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tabextract.runtime import default_cache, linked_output_dir, downloader_command
from tabextract.cache import CacheStore
from tabextract.download import default_output


class FrozenRuntimeTests(unittest.TestCase):
    def test_frozen_paths_are_user_writable_and_independent_of_install_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.frozen',True,create=True), \
             patch.dict('os.environ',{'LOCALAPPDATA':tmp,'GUITAR_TAB_CACHE':''}):
            self.assertEqual(default_cache(),Path(tmp)/'GuitarTabExtractor/cache')
            self.assertEqual(CacheStore().root,default_cache().resolve())
            self.assertEqual(Path(default_output('https://example.com/video')).parent,
                             Path(tmp)/'GuitarTabExtractor/output')
            self.assertEqual(default_output('song.mp4'),'song_吉他谱.pdf')
            self.assertEqual(CacheStore(Path(tmp)/'custom').root,Path(tmp)/'custom')

    def test_frozen_download_requires_bundled_console_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.frozen',True,create=True), \
             patch('sys.executable',str(Path(tmp)/'GuitarTabExtractor.exe')):
            with self.assertRaisesRegex(RuntimeError,'下载组件不完整'):
                downloader_command()
            worker=Path(tmp)/'tabextract-worker.exe'; worker.touch()
            self.assertEqual(downloader_command(),[str(worker),'--download-worker'])

    def test_source_edition_keeps_existing_behavior(self):
        with patch('sys.frozen',False,create=True),patch('sys.executable','python-test'):
            self.assertEqual(downloader_command(),['python-test','-m','yt_dlp'])
            self.assertEqual(linked_output_dir(),Path.cwd()/'output')
            self.assertEqual(default_cache().name,'.tabextract-cache')
