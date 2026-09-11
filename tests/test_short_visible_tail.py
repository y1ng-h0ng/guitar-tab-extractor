"""A note near the video edge is still real notation, even in a partial bar."""

import unittest

import cv2
import numpy as np

from tabextract.stitch import assemble_score


class ShortTailTests(unittest.TestCase):
    def test_partial_last_measure_retains_its_visible_note(self):
        score = np.full((240, 640), 255, np.uint8)
        for y in range(70, 171, 20):
            cv2.line(score, (0, y), (639, y), 0, 1)
        for x in (20, 300, 600):
            cv2.line(score, (x, 70), (x, 170), 0, 1)
        cv2.putText(score, "9", (614, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 2)
        cv2.line(score, (615, 152), (615, 212), 0, 2)
        result = assemble_score([score], log=lambda _: None)
        self.assertEqual(result.measures, 3)
        self.assertEqual(result.complete_measures, 2)
        # Search for the distinctive source note in the exported rows, so a
        # retained source canvas alone cannot conceal a later layout deletion.
        template = score[132:164, 611:630]
        scores = [float(cv2.matchTemplate(row, template, cv2.TM_CCOEFF_NORMED).max())
                  for row in result.rows]
        self.assertGreater(max(scores), 0.99)


if __name__ == "__main__":
    unittest.main()
