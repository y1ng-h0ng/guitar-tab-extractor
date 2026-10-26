"""Owned, versioned checkpoints. No pickle or files outside the cache are read."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np

from .pages import Segment
from .support import check_cancel
from .runtime import default_cache

SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def assembly_name(records):
    # A 128-bit revision key avoids exceeding Windows' path limit inside an
    # already fingerprinted job directory, without changing older job identities.
    return 'assembly-v2-' + digest([(r['id'], r['revision']) for r in records])[:32]


def fingerprint(path, cancel_event=None):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            check_cancel(cancel_event)
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _linked(path):
    # Windows junctions must be rejected too (is_symlink alone is insufficient).
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


class Checkpoint:
    def __init__(self, path):
        self.path = path

    def file(self, name):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('无效的缓存名称')
        path = self.path / (name + '.npz')
        if path.exists() and _linked(path):
            raise ValueError('缓存文件不能是链接')
        return path

    def load(self, name, default=None):
        try:
            with np.load(self.file(name), allow_pickle=False) as data:
                meta = json.loads(data['metadata'].tobytes().decode('utf-8'))
                if meta['schema'] != SCHEMA:
                    return default

                def decode(value):
                    if isinstance(value, list):
                        return [decode(v) for v in value]
                    if not isinstance(value, dict):
                        return value
                    if 'array' in value:
                        return data[value['array']].copy()
                    if 'segment' in value:
                        return Segment(*value['segment'])
                    if 'tuple' in value:
                        return tuple(decode(v) for v in value['tuple'])
                    return {decode(k): decode(v) for k, v in value['mapping']}
                return decode(meta['value'])
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            return default
        except Exception as error:
            # Bad ZIP checksums and truncated archives are cache misses.
            import zipfile
            if isinstance(error, zipfile.BadZipFile):
                return default
            raise

    def save(self, name, value):
        arrays = {}

        def encode(item):
            if isinstance(item, np.ndarray):
                key = f'array{len(arrays)}'
                arrays[key] = item
                return {'array': key}
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, Segment):
                return {'segment': [item.first, item.last, item.fps]}
            if isinstance(item, tuple):
                return {'tuple': [encode(v) for v in item]}
            if isinstance(item, list):
                return [encode(v) for v in item]
            if isinstance(item, dict):
                return {'mapping': [[encode(k), encode(v)] for k, v in item.items()]}
            return item
        meta = json.dumps({'schema': SCHEMA, 'value': encode(value)}, ensure_ascii=False)
        arrays['metadata'] = np.frombuffer(meta.encode('utf-8'), np.uint8)
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.path)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, self.file(name))
        finally:
            Path(temporary).unlink(missing_ok=True)


class CacheStore:
    """Entry locks also protect live downloads from another window's cleanup."""
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('GUITAR_TAB_CACHE') or
                         default_cache()).resolve()

    @contextmanager
    def job(self, identity):
        key = digest(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock(key):
            path = self.root / key
            if path.exists() and _linked(path):
                raise ValueError('缓存目录不能是链接')
            path.mkdir(exist_ok=True)
            checkpoint = Checkpoint(path)
            checkpoint.save('entry', {'owner': 'guitar-tab-extractor', 'identity': identity,
                                      'last_used': time.time()})
            try:
                yield checkpoint
            finally:
                checkpoint.save('entry', {'owner': 'guitar-tab-extractor', 'identity': identity,
                                          'last_used': time.time()})

    @contextmanager
    def _lock(self, key):
        locks = self.root / '_locks'
        locks.mkdir(exist_ok=True)
        if _linked(locks):
            raise ValueError('缓存锁目录不能是链接')
        lock = locks / key
        if lock.exists() and _linked(lock):
            raise ValueError('缓存锁不能是链接')
        with open(lock, 'a+b') as stream:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('此缓存正在另一个窗口中使用，请稍后重试') from error
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def entries(self):
        entries = []
        if not self.root.exists():
            return entries
        for path in self.root.iterdir():
            if not re.fullmatch('[0-9a-f]{64}', path.name) or not path.is_dir() or _linked(path):
                continue
            meta = Checkpoint(path).load('entry')
            if not meta or meta.get('owner') != 'guitar-tab-extractor':
                continue
            # Never walk into links while counting or deleting cached files.
            size, unsafe = 0, False
            for directory, dirs, names in os.walk(path, followlinks=False):
                for name in list(dirs) + names:
                    child = Path(directory) / name
                    try:
                        if _linked(child):
                            unsafe = True
                            if name in dirs:
                                dirs.remove(name)
                        elif child.is_file():
                            size += child.stat().st_size
                    except FileNotFoundError:
                        # An atomic writer or another cleanup may have moved it.
                        continue
            if not unsafe:
                entries.append(dict(meta, key=path.name, size=size))
        return entries

    def prune(self, clear=False, days=7, max_bytes=2 * 1024**3):
        entries = sorted(self.entries(), key=lambda e: e['last_used'])
        total = sum(e['size'] for e in entries)
        removed = 0
        for entry in entries:
            if not (clear or entry['last_used'] < time.time() - days * 86400 or total > max_bytes):
                continue
            try:
                with self._lock(entry['key']):
                    target = self.root / entry['key']
                    if target.resolve().parent != self.root or _linked(target):
                        continue
                    # Recheck under the lock; the entry may have been refreshed.
                    current = Checkpoint(target).load('entry')
                    if not clear and current and current['last_used'] != entry['last_used']:
                        continue
                    shutil.rmtree(target)
                    total -= entry['size']; removed += entry['size']
            except (RuntimeError, OSError):
                continue
        return removed
