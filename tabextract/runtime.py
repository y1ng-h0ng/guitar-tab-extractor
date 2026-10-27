"""Writable locations and worker commands for source and frozen editions."""
import os
import sys
from pathlib import Path

APP_VERSION = '1.1.1'


def frozen():
    return bool(getattr(sys, 'frozen', False))


def data_dir():
    return Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData/Local')/'GuitarTabExtractor'


def default_cache():
    return data_dir()/'cache' if frozen() else Path(__file__).resolve().parents[1]/'.tabextract-cache'


def linked_output_dir():
    return data_dir()/'output' if frozen() else Path.cwd()/'output'


def downloader_command():
    if not frozen():
        return [sys.executable, '-m', 'yt_dlp']
    worker = Path(sys.executable).with_name('tabextract-worker.exe')
    if not worker.is_file():
        raise RuntimeError('下载组件不完整，请重新安装，或完整解压便携版，勿单独移动主程序。')
    return [str(worker), '--download-worker']
