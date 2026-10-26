"""Acquire one linked video locally, with cancellable subprocess downloads."""

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
from .runtime import downloader_command, linked_output_dir
import tempfile
import threading
from urllib.parse import urlsplit, parse_qs, urlencode, urlunsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

from .support import check_cancel
from .cache import fingerprint


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
    if parts.hostname.lower() in ('www.bilibili.com', 'bilibili.com'):
        bvid = parse_qs(parts.query).get('bvid', [''])[0]
        if re.fullmatch(r'BV[0-9A-Za-z]+', bvid) and parts.path.startswith('/list/'):
            page = parse_qs(parts.query).get('p', ['1'])[0]
            return f'https://www.bilibili.com/video/{bvid}/?p={page}' if page.isdigit() else f'https://www.bilibili.com/video/{bvid}/'
    return url


class DownloadError(RuntimeError):
    def __init__(self, kind, message, detail=''):
        self.kind = kind
        detail = re.sub(r'https?://\S+', '[链接]', detail)[-500:]
        super().__init__('视频获取失败：' + message + ('\n' + detail if detail else ''))


def download_error(detail):
    lower = detail.lower()
    if any(s in lower for s in ('sign in', 'login', 'log in', 'cookies', 'private video', 'members-only')):
        return DownloadError('login', '此视频要求登录或访问权限；请选择公开可访问的视频，或使用已有的本地文件。', detail)
    if any(s in lower for s in ('404', '410 gone', 'deleted', 'does not exist', 'not found')):
        return DownloadError('unavailable', '链接失效、视频已删除或不可用，请检查链接。', detail)
    if any(s in lower for s in ('timed out', 'timeout', 'connection', 'resolve', 'network', 'ssl', 'certificate')):
        return DownloadError('network', '网络连接失败，请检查网络后重试；已下载的部分会保留。', detail)
    if any(s in lower for s in ('unsupported url', 'not supported', 'drm', 'is_live', 'live stream')):
        return DownloadError('unsupported', '此平台或视频类型暂不支持；可更新 yt-dlp 或使用本地视频。', detail)
    if 'requested format' in lower:
        return DownloadError('quality', '所选清晰度已不可用，请重新解析链接或改选自动清晰度。', detail)
    if any(s in lower for s in ('403', '412', '429', 'rate limit', 'initial state')):
        return DownloadError('platform', '平台暂时拒绝访问或限制请求，请稍后重试；这不一定是登录问题。', detail)
    return DownloadError('unknown', '解析或下载未完成，请根据下方原因重试，或改用本地视频。', detail)


def _selection(url, part=None):
    url = video_url(url)
    if not url:
        raise ValueError('请输入有效的视频链接')
    if part is not None and (isinstance(part, bool) or not isinstance(part, int) or part < 1):
        raise ValueError('分 P 序号必须是正整数')
    parsed = urlsplit(url)
    if parsed.hostname in ('www.bilibili.com', 'bilibili.com') and '/video/' in parsed.path:
        query = parse_qs(parsed.query)
        if part is not None:
            query['p'] = [str(part)]
        return urlunsplit(parsed._replace(query=urlencode(query, doseq=True))), ['--no-playlist']
    return url, ['--yes-playlist', '--playlist-items', str(part or 1)]


def _stop_process(proc):
    if proc.poll() is not None:
        return
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
        proc.kill(); proc.wait()


def _probe_json(url, flags, cancel_event):
    check_cancel(cancel_event)
    args = [*downloader_command(), '--ignore-config', '--no-cache-dir',
            '--no-warnings', '--no-colors', '--skip-download', '--dump-single-json',
            '--socket-timeout', '15', '--retries', '1', *flags, '--', url]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONUTF8': '1'},
        start_new_session=os.name != 'nt', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        while True:
            check_cancel(cancel_event)
            try:
                output, errors = proc.communicate(timeout=.15)
                break
            except subprocess.TimeoutExpired:
                continue
        check_cancel(cancel_event)
        if proc.returncode:
            raise download_error(errors)
        try:
            result = json.loads(output)
        except ValueError:
            raise download_error('平台返回了无法识别的视频信息') from None
        return result
    finally:
        _stop_process(proc)
        proc.stdout.close(); proc.stderr.close()


def probe_video(url, part=None, cancel_event=None, previous=None):
    """List at most 200 entries; resolve formats only for the selected part."""
    url = video_url(url)
    if not url:
        raise ValueError('请先输入视频链接')
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    default_part = int(query.get('p', ['1'])[0]) if query.get('p', ['1'])[0].isdigit() else 1
    part = part if part is not None else default_part
    _selection(url, part)  # validate before starting a subprocess
    listing_url = url
    if parsed.hostname in ('www.bilibili.com', 'bilibili.com') and '/video/' in parsed.path:
        query.pop('p', None)
        listing_url = urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))
    listing = previous.get('_listing') if previous and previous.get('url') == url else None
    if listing is None:
        listing = _probe_json(listing_url, ['--flat-playlist', '--yes-playlist', '--playlist-items', '1:200'], cancel_event)
    entries = listing.get('entries')
    parts = [{'index': i+1, 'title': (e or {}).get('title') or f'分 P {i+1}'}
             for i, e in enumerate(entries)] if entries is not None else [{'index': 1, 'title': listing.get('title', '视频')}]
    if not 1 <= part <= len(parts):
        raise ValueError(f'分 P 序号超出范围（当前可选 1–{len(parts)}）')
    if entries is None and listing.get('formats'):
        info = listing
    else:
        selected_url, flags = _selection(url, part)
        info = _probe_json(selected_url, flags, cancel_event)
        if info.get('entries') is not None:
            info = next((e for e in info['entries'] if e), None)
        if not info:
            raise download_error('所选分 P 没有可用视频')
    if info.get('is_live'):
        raise download_error('live stream is not supported')
    formats = []
    for f in info.get('formats', []):
        if f.get('vcodec') == 'none' or not f.get('format_id'):
            continue
        if f.get('has_drm') or not (f.get('width') or f.get('height') or f.get('vcodec')):
            continue
        identifier = str(f['format_id'])
        if not re.fullmatch(r'[\w.-]+', identifier):
            continue
        formats.append({'id': identifier, 'width': f.get('width'), 'height': f.get('height'),
            'label': f"{f.get('width') or '?'} × {f.get('height') or '?'} · {f.get('fps') or '?'} fps · {f.get('ext') or '?'} [{identifier}]"})
    formats.reverse()
    return dict(url=url, part=part, parts=parts, title=info.get('title', '视频'),
        duration=info.get('duration'), width=info.get('width'), height=info.get('height'),
        thumbnail=info.get('thumbnail'), formats=formats, _listing=listing,
        limited=len(parts) >= 200)


def thumbnail_image(url, cancel_event=None):
    """Optional bounded thumbnail; preview failures must not block extraction."""
    if not url or urlsplit(url).scheme not in ('http', 'https'):
        return None
    check_cancel(cancel_event)
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=5) as response:
            content = response.read(2 * 1024**2 + 1)
        check_cancel(cancel_event)
        if len(content) <= 2 * 1024**2:
            return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        pass
    return None


def default_output(source):
    url = video_url(source)
    if url:
        key = hashlib.sha256(url.encode()).hexdigest()[:8]
        return str(linked_output_dir() / f'链接视频_{key}_吉他谱.pdf')
    return os.path.splitext(str(source))[0] + '_吉他谱.pdf'


def safe_title(title):
    title = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', str(title)).strip(' .')[:80].rstrip(' .')
    if not title or title.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        title = '链接视频'
    return title


def download_video(url, directory, cancel_event=None, log=print, progress_cb=None,
                   part=None, format_id=None):
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
    if format_id is not None:
        if not re.fullmatch(r'[\w.-]+', str(format_id)):
            raise ValueError('清晰度编号无效，请重新解析链接后选择')
        fmt = str(format_id)
    url, playlist_flags = _selection(url, part)
    args = [*downloader_command(), '--ignore-config', *playlist_flags,
            '--no-cache-dir', '--no-warnings', '--no-colors',
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
            raise download_error(error_text)
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
            path.replace(destination)
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
    """Temporary by default; GUI/CLI opt into an owned persistent download cache."""
    def __init__(self, cache_store=None):
        self.url = None
        self.path = None
        self.temp = None
        self.cache_store = cache_store
        self.lease = None
        self.selection = None

    def resolve(self, source, **options):
        url = video_url(source)
        if not url:
            self.close()
            if not Path(source).is_file():
                raise ValueError('请先选择有效的视频文件或输入视频链接')
            return str(source)
        check_cancel(options.get('cancel_event'))
        selection = (url, options.get('part'), options.get('format_id'))
        if selection == self.selection and self.path and Path(self.path).is_file():
            return self.path
        self.close()
        try:
            if self.cache_store:
                identity = dict(kind='download', version=1, url=url,
                                part=selection[1], format_id=selection[2])
                self.cache_store.prune()
                lease = self.cache_store.job(identity)
                checkpoint = lease.__enter__()
                self.lease = lease
                saved = checkpoint.load('video')
                directory = checkpoint.path
                if saved:
                    candidate = (directory / saved['name']).resolve()
                    if (candidate.parent == directory and candidate.is_file() and
                            fingerprint(candidate, options.get('cancel_event')) == saved.get('sha256')):
                        self.path = str(candidate)
                        options.get('log', print)('复用已下载的视频。')
                if not self.path:
                    self.path = download_video(url, directory, **options)
                    checkpoint.save('video', dict(name=Path(self.path).name,
                        sha256=fingerprint(self.path, options.get('cancel_event'))))
            else:
                self.temp = tempfile.TemporaryDirectory(prefix='guitar-tab-video-')
                self.path = download_video(url, self.temp.name, **options)
            self.url = url
            self.selection = selection
            return self.path
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.temp:
            self.temp.cleanup()
        if self.lease:
            self.lease.__exit__(None, None, None)
        self.lease = self.selection = None
        self.temp = self.url = self.path = None
