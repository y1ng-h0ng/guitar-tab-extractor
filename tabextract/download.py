"""Acquire one linked video locally, with cancellable subprocess downloads."""

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit

import cv2

from .support import check_cancel


def video_url(value):
    """Accept a single HTTP(S) URL, including copied app share text."""
    value = str(value).strip()
    urls = re.findall(r"https?://[^\s<>\"'\u201c\u201d\u4e00-\u9fff]+", value, re.I)
    if re.fullmatch(r"https?://[^\s<>\"']+", value, re.I):
        urls = [value]
    if not urls:
        if '://' in value:
            raise ValueError('视频链接需要以 http:// 或 https:// 开头')
        return None
    if len(urls) != 1:
        raise ValueError('请一次输入一个视频链接')
    url = urls[0].rstrip('.,;!，。；！？、）)]}')
    parts = urlsplit(url)
    if not parts.hostname or parts.username or parts.password:
        raise ValueError('视频链接无效，请复制视频的分享链接')
    return url


def default_output(source):
    url = video_url(source)
    if url:
        key = hashlib.sha256(url.encode()).hexdigest()[:8]
        return str(Path.cwd() / 'output' / f'链接视频_{key}_吉他谱.pdf')
    return os.path.splitext(str(source))[0] + '_吉他谱.pdf'


def safe_title(title):
    title = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', str(title)).strip(' .')[:80].rstrip(' .')
    if not title or title.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        title = '链接视频'
    return title


def download_video(url, directory, cancel_event=None, log=print, progress_cb=None):
    """Download to an owned directory; callers manage its lifetime.

    Only a video stream is needed for score images, so no audio merge is
    requested. A child process makes parsing and stalled network I/O cancellable.
    Machine-readable events, not localized terminal messages, select the file.
    """
    url = video_url(url)
    if not url:
        raise ValueError('请输入有效的视频链接')
    check_cancel(cancel_event)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # noqa: F401 - fail before launching a missing dependency
    except ImportError as error:
        raise RuntimeError('缺少链接下载组件，请重新运行 install_windows.bat') from error
    # Prefer decodable AVC/MP4, then any single video-containing format.
    fmt = 'bv[ext=mp4][vcodec^=avc]/b[ext=mp4][vcodec^=avc]/bv[ext=mp4]/b[ext=mp4]/bv/b'
    args = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist',
            '--playlist-items', '1', '--no-cache-dir', '--no-warnings', '--no-colors',
            '--newline', '--progress', '--no-simulate', '--socket-timeout', '15',
            '--retries', '2', '--fragment-retries', '2', '--fixup', 'never',
            '--hls-prefer-native', '--match-filter', '!is_live',
            '--progress-template', 'download:__TAB_PROGRESS__%(progress._percent_str)s',
            '--print', 'before_dl:__TAB_TITLE__%(title)j',
            '--print', 'after_move:__TAB_FILE__%(filepath)j',
            '-f', fmt, '-o', str(directory / 'video.%(ext)s'), '--', url]
    log('正在解析视频链接 ...')
    if progress_cb:
        progress_cb(0, '正在解析视频链接 ...')
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace',
                            env={**os.environ, 'PYTHONUTF8': '1'},
                            start_new_session=os.name != 'nt',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    events = queue.Queue()
    def read_output():
        try:
            for line in proc.stdout:
                events.put(line.strip())
        finally:
            events.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    path, title, error_text = None, '链接视频', ''
    last_pct = -1
    try:
        while True:
            check_cancel(cancel_event)
            try:
                line = events.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith('__TAB_FILE__'):
                path = Path(json.loads(line[len('__TAB_FILE__'):])).resolve()
            elif line.startswith('__TAB_TITLE__'):
                title = safe_title(json.loads(line[len('__TAB_TITLE__'):]))
                log(f'获取视频：{title}')
            elif line.startswith('__TAB_PROGRESS__'):
                match = re.search(r'([\d.]+)%', line)
                if match:
                    pct = min(100, max(0, float(match[1])))
                    if int(pct) != last_pct:
                        last_pct = int(pct)
                        if progress_cb:
                            progress_cb(pct, f'正在获取视频：{pct:.0f}%')
            elif line.startswith('ERROR:'):
                error_text = re.sub(r'https?://\S+', '[链接]', line)[:400]
        while proc.poll() is None:
            check_cancel(cancel_event)
            try:
                proc.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
        code = proc.returncode
        check_cancel(cancel_event)
        if code or path is None:
            detail = '\n' + error_text if error_text else ''
            raise RuntimeError('视频获取失败：请检查链接和网络；该视频可能需要登录、暂不受支持或是直播。'
                               '可更新 yt-dlp 后重试，或改用本地视频。' + detail)
        if not path.is_relative_to(directory) or not path.is_file():
            raise RuntimeError('下载结果无效，未找到视频文件')
        cap = cv2.VideoCapture(str(path))
        try:
            ok, _ = cap.read()
        finally:
            cap.release()
        if not ok:
            raise RuntimeError('已获取文件，但无法读取视频画面；请尝试其他清晰度或本地视频')
        destination = directory / (title + path.suffix)
        if destination != path:
            path.rename(destination)
        log('视频已获取，开始整理谱面。')
        return str(destination)
    finally:
        if proc.poll() is None:
            # Stop any protocol helper too, before deleting temporary files.
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    proc.terminate()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        reader.join(timeout=3)
        proc.stdout.close()


class VideoSource:
    """Reuse a downloaded video for preview/run; remove it on change or close."""
    def __init__(self):
        self.url = None
        self.path = None
        self.temp = None

    def resolve(self, source, **options):
        url = video_url(source)
        if not url:
            self.close()
            if not Path(source).is_file():
                raise ValueError('请先选择有效的视频文件或输入视频链接')
            return str(source)
        check_cancel(options.get('cancel_event'))
        if url == self.url and self.path and Path(self.path).is_file():
            return self.path
        self.close()
        self.temp = tempfile.TemporaryDirectory(prefix='guitar-tab-video-')
        try:
            self.path = download_video(url, self.temp.name, **options)
            self.url = url
            return self.path
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.temp:
            self.temp.cleanup()
        self.temp = self.url = self.path = None
