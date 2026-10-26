"""Offline smoke check executed by the actual packaged executable."""
import json
import subprocess
from pathlib import Path


def check(destination):
    import tkinter as tk
    from PIL import Image, ImageTk
    from tkinterdnd2 import TkinterDnD
    import cv2
    import numpy as np
    from gui import App
    from .runtime import frozen, default_cache, linked_output_dir, downloader_command
    from .pipeline import run_pipeline
    import ssl
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    root = tk.Tk(); root.withdraw()
    app = App(root)
    try:
        root.update_idletasks()
        assert app.drop_enabled, '拖放组件未加载'
        image = ImageTk.PhotoImage(Image.new('RGB',(16,16),'white'), master=root)
        assert image.width() == 16
        tk_version = root.tk.call('info','patchlevel')
    finally:
        app.close()
    video = out/'测试视频.mp4'
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),12,(480,180))
    assert writer.isOpened(), 'MP4 编码组件不可用'
    frame = np.full((180,480,3),20,np.uint8)
    for y in range(65,116,10): cv2.line(frame,(10,y),(470,y),(230,230,230),1)
    for x in (30,170,310,455): cv2.line(frame,(x,65),(x,115),(230,230,230),1)
    cv2.putText(frame,'7',(90,103),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
    cv2.line(frame,(130,110),(130,146),(230,230,230),1)
    for _ in range(24): writer.write(frame)
    writer.release()
    summary = run_pipeline(video,out/'测试谱面.pdf',region=(0,0,480,180),log=lambda _:None)
    assert (out/'测试谱面.pdf').read_bytes().startswith(b'%PDF')
    proc = subprocess.run([*downloader_command(),'--version'],capture_output=True,text=True,
        encoding='utf-8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert proc.returncode == 0, proc.stderr
    result = dict(frozen=frozen(),tk=tk_version,drag_drop=True,images=True,
        mp4=True,tls_validation=True,pdf_pages=summary['pdf_pages'],downloader=proc.stdout.strip(),
        cache=str(default_cache()),linked_output=str(linked_output_dir()))
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
