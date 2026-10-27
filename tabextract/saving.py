"""Safe output naming and publication; existing files require explicit consent."""
import hashlib
import os
from pathlib import Path


def output_paths(path, report=False):
    path = Path(path)
    return [path, path.with_suffix('.report.json')] if report else [path]


def conflicts(path, report=False):
    return [p for p in output_paths(path, report) if os.path.lexists(p)]


def unused_output(path, report=False):
    path = Path(path)
    candidate = path
    number = 2
    while conflicts(candidate, report):
        candidate = path.with_name(f'{path.stem} ({number}){path.suffix}')
        number += 1
    return str(candidate)


def check_output(path, report=False, overwrite=False):
    existing = conflicts(path, report)
    if any(p.is_dir() for p in existing):
        raise ValueError('输出位置是文件夹，请另选 PDF 文件名')
    if existing and not overwrite:
        raise FileExistsError(f'文件已存在，未覆盖：{existing[0]}。请另选文件名，或明确确认覆盖。')


def publish_file(staged, target, overwrite=False):
    if overwrite:
        os.replace(staged, target)
    elif os.name == 'nt':
        # Unlike replace(), Windows rename fails if a destination appeared
        # after the preflight check. Never truncate an existing user's file.
        os.rename(staged, target)
    else:
        os.link(staged, target)
        os.unlink(staged)


def signature(path):
    try:
        with Path(path).open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
            return digest.hexdigest()
    except FileNotFoundError:
        return None
