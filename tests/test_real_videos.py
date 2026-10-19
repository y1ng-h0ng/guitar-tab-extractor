"""Enable with TABEXTRACT_TEST_VIDEOS=<directory with the four original MP4s>."""
import os
from pathlib import Path
import tempfile
import unittest
from tabextract.benchmark import run_benchmark


class RealVideoRegression(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TABEXTRACT_TEST_VIDEOS'),
                         '未配置真实视频目录；设置 TABEXTRACT_TEST_VIDEOS 启用真实视频回归')
    def test_original_video_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(os.environ.get('TABEXTRACT_TEST_OUTPUT',temporary))
            report = run_benchmark(os.environ['TABEXTRACT_TEST_VIDEOS'],output)
            self.assertTrue(report['passed'],report)
