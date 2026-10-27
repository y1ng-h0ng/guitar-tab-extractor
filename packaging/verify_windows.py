"""Exercise an unpacked/installed app with Python removed from its environment."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path, help='Optional real video; analyze first 8 seconds')
    args = parser.parse_args()
    bundle = args.bundle.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith('PYTHON')}
    env['PATH'] = str(Path(env['SYSTEMROOT'])/'System32')+os.pathsep+env['SYSTEMROOT']
    env['LOCALAPPDATA'] = str(out/'profile')

    def run(exe, *argv, expected_code=0):
        result = subprocess.run([str(bundle/exe), *map(str, argv)], env=env, cwd=out,
            capture_output=True, encoding='utf-8', errors='strict', timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode != expected_code:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout

    run('GuitarTabExtractor.exe', '--smoke-test', out/'gui')
    result = json.loads((out/'gui/result.json').read_text(encoding='utf-8'))
    assert result['frozen'] and result['drag_drop'] and result['pdf_pages'] > 0
    video = out/'gui/测试视频.mp4'
    existing = out/'gui/测试视频_吉他谱.pdf'
    existing.write_bytes(b'User-owned PDF sentinel')
    run('tabextract-worker.exe', '--cli', video, '--no-cache', '--region', '0','0','480','180')
    assert existing.read_bytes() == b'User-owned PDF sentinel'
    assert (existing.parent/'测试视频_吉他谱 (2).pdf').read_bytes().startswith(b'%PDF')
    run('tabextract-worker.exe', '--cli', video, '--no-cache', '-o', existing, expected_code=1)
    assert existing.read_bytes() == b'User-owned PDF sentinel'
    run('tabextract-worker.exe', '--cli', video, '--no-cache', '--region', '0','0','480','180',
        '-o', existing, '--overwrite')
    assert existing.read_bytes().startswith(b'%PDF')
    result['automatic_name_collision'] = True
    result['explicit_overwrite_required'] = True
    (out/'gui/watch.html').write_text(
        '<html><head><meta charset="utf-8"><title>中文吉他谱测试</title></head>'
        '<body><video src="测试视频.mp4"></video></body></html>', encoding='utf-8')

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(out/'gui')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/watch.html'
        info = json.loads(run('tabextract-worker.exe', '--cli', url, '--info', '--no-cache'))
        assert '中文吉他谱测试' in info['title'], info
        log = run('tabextract-worker.exe', '--cli', url, '--no-cache', '--region',
            '0', '0', '480', '180', '-o', out/'链接测试.pdf')
        assert (out/'链接测试.pdf').read_bytes().startswith(b'%PDF')
        (out/'http.log').write_text(log, encoding='utf-8')
        result['http_download_to_pdf'] = True
        result['chinese_link_title'] = info['title']
    finally:
        server.shutdown(); server.server_close(); thread.join()
    if args.video:
        log = run('tabextract-worker.exe', '--cli', args.video.resolve(), '--no-cache',
            '--end', '8', '-o', out/'原视频测试.pdf')
        assert (out/'原视频测试.pdf').read_bytes().startswith(b'%PDF')
        (out/'real-video.log').write_text(log, encoding='utf-8')
        result['real_video'] = str(args.video.resolve())
    result['python_removed_from_path'] = True
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen GUI, Unicode metadata, HTTP download and PDF checks passed.')


if __name__ == '__main__':
    main()
