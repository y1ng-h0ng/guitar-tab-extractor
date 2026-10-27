import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import main
from tabextract.pdfout import export_rows, Canvas
from tabextract.pipeline import run_pipeline
from tabextract.saving import unused_output
import test_gui_workflow as gui_tests
import test_segment_review as review_tests


class OutputSafetyTests(unittest.TestCase):
    def test_existing_pdf_and_report_are_preserved_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory)/'旧谱.pdf'; pdf.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                run_pipeline('nonexistent.mp4', pdf)
            self.assertEqual(pdf.read_bytes(), b'original')
            pdf.unlink(); pdf.with_suffix('.report.json').write_bytes(b'old report')
            with self.assertRaises(FileExistsError):
                run_pipeline('nonexistent.mp4', pdf, generate_report=True)
            self.assertEqual(unused_output(pdf, True), str(pdf.with_name('旧谱 (2).pdf')))

    def test_file_appearing_during_render_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory)/'谱.pdf'
            original_save = Canvas.save
            def race(canvas):
                original_save(canvas)
                pdf.write_bytes(b'another process')
            with patch.object(Canvas, 'save', race), self.assertRaises(FileExistsError):
                export_rows([np.full((120,480),255,np.uint8)],pdf)
            self.assertEqual(pdf.read_bytes(),b'another process')
            self.assertFalse(list(pdf.parent.glob('.tabextract-*')))
            export_rows([np.full((120,480),255,np.uint8)],pdf,overwrite=True)
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF'))

    def test_cli_refuses_explicit_existing_target_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory)/'谱.pdf'; pdf.write_bytes(b'original')
            with patch('main.VideoSource') as source, patch('main.run_pipeline') as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(['https://example.com/video','-o',str(pdf),'--no-cache']),1)
                source.return_value.resolve.assert_not_called(); run.assert_not_called()
            self.assertEqual(pdf.read_bytes(),b'original')


class GuiOutputSafetyTests(unittest.TestCase):
    setUp = gui_tests.GuiWorkflowTests.setUp
    close_root = gui_tests.GuiWorkflowTests.close_root
    drop = gui_tests.GuiWorkflowTests.drop

    def test_auto_name_and_repeated_runs_never_overwrite(self):
        original = self.video.with_name(self.video.stem+'_吉他谱.pdf')
        original.write_bytes(b'original')
        self.drop(self.video)
        self.assertEqual(Path(self.app.output.get()), original.with_name(original.stem+' (2).pdf'))
        Path(self.app.output.get()).write_bytes(b'other')
        with patch.object(self.app,'launch',side_effect=lambda fn:fn()), patch('gui.run_pipeline') as run, patch('gui.messagebox.askyesno') as ask:
            self.app.start()
            self.assertEqual(Path(run.call_args.args[1]),original.with_name(original.stem+' (3).pdf'))
            self.assertFalse(run.call_args.kwargs['overwrite']); ask.assert_not_called()
        self.assertEqual(original.read_bytes(),b'original')

    def test_custom_name_requires_yes_and_no_does_not_start(self):
        self.drop(self.video)
        pdf = self.video.with_suffix('.pdf'); pdf.write_bytes(b'original')
        self.app.output.set(str(pdf))
        with patch.object(self.app,'launch',side_effect=lambda fn:fn()) as launch, patch('gui.run_pipeline') as run, patch('gui.messagebox.askyesno',return_value=False) as ask:
            self.app.start(); ask.assert_called_once(); launch.assert_not_called()
            self.assertEqual(ask.call_args.kwargs['default'],'no')
            ask.return_value = True
            self.app.start()
            self.assertTrue(run.call_args.kwargs['overwrite'])
        self.assertEqual(pdf.read_bytes(),b'original')


class ReviewOutputSafetyTests(unittest.TestCase):
    setUp = review_tests.LocalReviewTests.setUp
    start_patch = review_tests.LocalReviewTests.start_patch

    def test_external_pdf_replacement_stops_local_edits(self):
        self.pdf.write_bytes(b'externally replaced')
        with self.assertRaisesRegex(FileExistsError,'外部修改'):
            self.session.reprocess(2,2.1,2.9,'bright',1,log=lambda _:None)
        self.loader.assert_not_called()
        self.assertEqual(self.pdf.read_bytes(),b'externally replaced')
