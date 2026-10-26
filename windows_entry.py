"""Shared frozen entry: windowed GUI and a console worker with real pipes."""
import os
import sys


def main():
    # Frozen Python ignores PYTHONUTF8. Downloader pipes must still be UTF-8,
    # including Chinese titles on machines whose Windows code page is GBK.
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if stream is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))
        elif hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    if sys.argv[1:2] == ['--download-worker']:
        import yt_dlp
        yt_dlp.main(sys.argv[2:])
        return 0
    if sys.argv[1:2] == ['--cli']:
        import main as command_line
        return command_line.main(sys.argv[2:])
    if sys.argv[1:2] == ['--smoke-test']:
        from tabextract.bundle_check import check
        check(sys.argv[2])
        return 0
    # No-console executables have None streams. Keep library diagnostics from
    # crashing the GUI; fatal startup errors get a persistent log and a dialog.
    try:
        from gui import main as gui_main
        gui_main()
    except Exception:
        import traceback
        from tabextract.runtime import data_dir
        folder = data_dir(); folder.mkdir(parents=True, exist_ok=True)
        log = folder/'startup-error.log'
        log.write_text(traceback.format_exc(), encoding='utf-8')
        import tkinter.messagebox
        tkinter.messagebox.showerror('启动失败', f'错误记录已保存到：\n{log}\n请完整解压或重新安装。')
        return 1
    return 0


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
