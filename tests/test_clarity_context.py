"""Dark background texture must not receive a faint-score contrast boost."""
import unittest
import cv2
import numpy as np
from tabextract.clean import clean_frames, notation_response, stroke_response


class ForegroundContextTests(unittest.TestCase):
    def source(self, shadow=False):
        gray = np.full((180,480),15,np.uint8)
        for y in range(65,116,10):
            cv2.line(gray,(10,y),(470,y),220,1)
            if shadow: cv2.line(gray,(240,y),(340,y),35,1)
        # Same dark contrast as translucent notation, but over a bright staff.
        gray[20:43,90:92] = 38
        gray[88:91,370:378] = 35
        cv2.putText(gray,'7',(260,102),cv2.FONT_HERSHEY_SIMPLEX,.45,35 if shadow else 220,1)
        cv2.line(gray,(280,110),(280,150),35 if shadow else 220,1)
        gray[132:139,279:282] = 15  # An actual source gap must remain open.
        return gray

    def test_bright_staff_does_not_promote_dark_scenery(self):
        for polarity in ('bright','dark'):
            gray=self.source()
            frame=cv2.cvtColor(gray if polarity=='bright' else 255-gray,cv2.COLOR_GRAY2BGR)
            permissive=stroke_response(frame,polarity)
            self.assertGreater(int(permissive[25:40,90:92].max()),20)
            page=clean_frames([frame]*9,polarity)
            self.assertTrue((page[60:129,270:276] > 245).all())
            self.assertTrue((page[264:273,1110:1134] > 245).all())
            self.assertGreater(np.count_nonzero(page[270:309,780:825] < 180),30)
            self.assertLess(float(page[420:444,840:842].mean()),100)

    def test_faint_staff_columns_keep_notes_and_real_stem_gaps(self):
        for polarity in ('bright','dark'):
            gray=self.source(shadow=True)
            frame=cv2.cvtColor(gray if polarity=='bright' else 255-gray,cv2.COLOR_GRAY2BGR)
            response=notation_response(frame,polarity)
            self.assertGreater(int(response[90:103,260:275].max()),10)
            page=clean_frames([frame]*9,polarity)
            self.assertGreater(np.count_nonzero(page[270:309,780:825] < 180),30)
            self.assertLess(float(page[420:444,840:842].mean()),100)
            self.assertTrue((page[402:411,837:846] > 245).all())
            self.assertTrue((page[60:129,270:276] > 245).all())


if __name__ == '__main__': unittest.main()
