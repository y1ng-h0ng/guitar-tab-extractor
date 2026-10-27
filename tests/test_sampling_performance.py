from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import cv2
import numpy as np
from tabextract.clean import neutral_channel
from tabextract.pages import Segment, load_segment_frames


class ExactSpeedupTests(unittest.TestCase):
    def test_channel_extrema_match_old_formula_including_strided_crops(self):
        rng = np.random.default_rng(24)
        image = rng.integers(0,256,(75,111,3),dtype=np.uint8)
        for crop in (image,image[::2,3:-2],image[::-1,::-1]):
            np.testing.assert_array_equal(neutral_channel(crop,'bright'),crop.min(axis=2))
            np.testing.assert_array_equal(neutral_channel(crop,'dark'),255-crop.max(axis=2))

    def test_sequential_sampling_returns_exact_original_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'测试.mp4'
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,96))
            self.assertTrue(writer.isOpened())
            for index in range(100):
                frame = np.full((96,160,3),index,np.uint8)
                cv2.putText(frame,str(index),(12,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                writer.write(frame)
            writer.release()
            segment = Segment(10,90,25); region=(7,5,140,80)
            guard = 3
            indices = np.unique(np.linspace(10+guard,90-guard,25).round().astype(int))
            reference = cv2.VideoCapture(str(path)); expected=[]
            try:
                for index in indices:
                    reference.set(cv2.CAP_PROP_POS_FRAMES,int(index))
                    ok,frame = reference.read(); self.assertTrue(ok)
                    expected.append(frame[5:85,7:147])
            finally: reference.release()
            capture = cv2.VideoCapture(str(path)); tracked=Mock(wraps=capture)
            with patch('tabextract.pages.cv2.VideoCapture',return_value=tracked):
                actual = load_segment_frames(path,region,segment)
            self.assertEqual(tracked.set.call_count,1)
            for old,new in zip(expected,actual): np.testing.assert_array_equal(old,new)
            self.assertEqual(len(actual),len(expected))
            self.assertTrue(all(frame.flags.owndata for frame in actual),
                            'Cropped samples must not retain full video buffers')
