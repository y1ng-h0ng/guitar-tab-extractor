"""Build the Windows app, portable archive and optional Inno Setup installer."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ('numpy', 'opencv-python-headless', 'Pillow', 'reportlab',
           'tkinterdnd2', 'yt-dlp', 'charset-normalizer')


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def notices(bundle):
    target = bundle/'licenses'
    target.mkdir(exist_ok=True)
    for name in RUNTIME:
        dist = metadata.distribution(name)
        for rel in dist.files or []:
            path = Path(str(rel))
            if '..' in path.parts:
                continue
            if any(word in str(path).lower() for word in ('license', 'copying', 'notice')):
                source = Path(dist.locate_file(rel))
                if source.is_file():
                    dest = target/name/path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
    base = Path(sys.base_prefix)
    shutil.copy2(base/'LICENSE.txt', target/'Python-LICENSE.txt')
    for source in (base/'tcl').rglob('*'):
        if source.is_file() and source.name.lower() in ('license.terms', 'license', 'license.txt'):
            dest = target/'Tcl-Tk'/source.relative_to(base/'tcl')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    shutil.copy2(ROOT/'packaging/ChineseSimplified.LICENSE', target/'ChineseSimplified-LICENSE.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iscc', type=Path, help='Inno Setup 6 ISCC.exe')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/releases')
    parser.add_argument('--dist-dir', type=Path, default=ROOT/'dist')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/windows')
    parser.add_argument('--reuse-bundle', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('Windows x64 is required')
    if args.iscc and not args.iscc.is_file():
        parser.error('ISCC.exe does not exist')
    version = re.search(r"APP_VERSION = '([^']+)'", (ROOT/'tabextract/runtime.py').read_text(encoding='utf-8')).group(1)
    output = args.output.resolve()
    dist = args.dist_dir.resolve()
    bundle = dist/'GuitarTabExtractor'
    if output == bundle or bundle in output.parents:
        parser.error('Output must be outside the application bundle')
    output.mkdir(parents=True, exist_ok=True)
    if not args.reuse_bundle:
        run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
            '--workpath', args.build_dir.resolve(), '--distpath', dist, ROOT/'packaging/windows.spec')
    for required in ('GuitarTabExtractor.exe', 'tabextract-worker.exe', '_internal/python312.dll'):
        if not (bundle/required).is_file():
            raise RuntimeError(f'Incomplete bundle: {required}')
    for source in ('LICENSE', 'packaging/使用说明.txt'):
        shutil.copy2(ROOT/source, bundle/Path(source).name)
    notices(bundle)
    manifest = dict(version=version, source_commit=git('rev-parse', 'HEAD'),
        source_dirty=bool(git('status', '--porcelain', '--untracked-files=normal')),
        python=sys.version, architecture=platform.machine(),
        dependencies={name: metadata.version(name) for name in (*RUNTIME, 'pyinstaller', 'pyinstaller-hooks-contrib')})
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2)
    (bundle/'build-manifest.json').write_text(encoded, encoding='utf-8')
    (output/'build-manifest.json').write_text(encoded, encoding='utf-8')
    shutil.copy2(ROOT/'packaging/使用说明.txt', output/'使用说明.txt')
    archive = output/f'GuitarTabExtractor-{version}-Portable-x64'
    shutil.make_archive(str(archive), 'zip', root_dir=dist, base_dir=bundle.name)
    artifacts = [Path(str(archive)+'.zip')]
    if args.iscc:
        run(args.iscc.resolve(), f'/DBundleDir={bundle}', f'/DOutputDir={output}',
            f'/DAppVersion={version}', ROOT/'packaging/windows.iss')
        artifacts.append(output/f'GuitarTabExtractor-{version}-Setup-x64.exe')
    checksums = []
    for path in artifacts:
        with path.open('rb') as stream:
            checksums.append(f'{hashlib.file_digest(stream, "sha256").hexdigest()}  {path.name}')
    (output/'SHA256SUMS.txt').write_text('\n'.join(checksums)+'\n', encoding='ascii')
    print(f'Created {len(artifacts)} package(s) in {output}')


if __name__ == '__main__':
    main()
