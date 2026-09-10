"""Tkinter desktop application. All widgets are updated on the main thread."""

import os
import queue
import threading
import traceback
from pathlib import Path
import cv2

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
except ImportError as error:
    raise SystemExit("缺少图形界面依赖，请安装官方 Python（包含 Tcl/Tk）和 requirements.txt。") from error
from tabextract import run_pipeline
from tabextract.pipeline import inspect_video
from tabextract.support import CancelledError, open_file, validate_options

VIDEO_EXTS = "*.mp4 *.mov *.avi *.mkv *.flv *.wmv *.m4v"
COLORS = {"自动识别": "auto", "白字 / 暗底": "bright", "黑字 / 白底": "dark"}


class RegionDialog(tk.Toplevel):
    def __init__(self, parent, frame, region, on_apply):
        super().__init__(parent)
        self.title("预览 / 框选谱面")
        self.transient(parent); self.grab_set()
        self.frame = frame; self.on_apply = on_apply; self.region = region; self.drag = None
        h, w = frame.shape[:2]
        self.scale = min(960 / w, 560 / h, 1.0)
        self.w, self.h = round(w * self.scale), round(h * self.scale)
        ttk.Label(self, text="拖动鼠标框选谱面。请包含上方字母、推弦标记，以及下方符干和符梁。", padding=10).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=self.w, height=self.h, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=10)
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((self.w, self.h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ffb23d", width=2)
        self.label = ttk.Label(self, padding=10); self.label.pack(anchor="w")
        buttons = ttk.Frame(self, padding=10); buttons.pack(fill="x")
        ttk.Button(buttons, text="使用这个范围", command=self.apply).pack(side="right")
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=8)
        self.canvas.bind("<ButtonPress-1>", self.press); self.canvas.bind("<B1-Motion>", self.move); self.canvas.bind("<ButtonRelease-1>", self.release)
        self.draw()

    def draw(self):
        if self.region:
            x, y, w, h = self.region; s = self.scale
            self.canvas.coords(self.rect, x * s, y * s, (x + w) * s, (y + h) * s)
            self.label.config(text=f"范围：x={x}, y={y}, 宽={w}, 高={h}")
        else:
            self.label.config(text="请在图中拖动，框出完整谱面。")

    def point(self, event):
        return max(0, min(self.w, event.x)), max(0, min(self.h, event.y))

    def press(self, event): self.drag = self.point(event)

    def move(self, event):
        if self.drag:
            x, y = self.point(event); self.canvas.coords(self.rect, *self.drag, x, y)

    def release(self, event):
        if not self.drag: return
        x, y = self.point(event); a, b = self.drag; self.drag = None
        left, top = round(min(a, x) / self.scale), round(min(b, y) / self.scale)
        right, bottom = round(max(a, x) / self.scale), round(max(b, y) / self.scale)
        fh, fw = self.frame.shape[:2]; right = min(right, fw); bottom = min(bottom, fh)
        self.region = (left, top, right - left, bottom - top); self.draw()

    def apply(self):
        if not self.region or self.region[2] < 80 or self.region[3] < 25:
            messagebox.showerror("范围太小", "请框选完整谱面。", parent=self); return
        self.on_apply(self.region); self.destroy()


class App:
    def __init__(self, root):
        self.root = root; root.title("吉他谱视频提取工具"); root.geometry("860x660"); root.minsize(740, 580)
        self.events = queue.Queue(); self.worker = None; self.cancel_event = threading.Event(); self.closing = False
        self.region = None; self.region_source = None; self.summary = None; self.preview_path = None
        self.video = tk.StringVar(); self.output = tk.StringVar(); self.fps = tk.StringVar(value="6"); self.dpi = tk.StringVar(value="300"); self.bars = tk.StringVar(value="4")
        self.color = tk.StringVar(value="自动识别"); self.debug = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="选择视频即可开始；也可以先预览谱面范围。"); self.region_text = tk.StringVar(value="自动定位谱面")
        frame = ttk.Frame(root, padding=18); frame.pack(fill="both", expand=True); frame.columnconfigure(1, weight=1); frame.rowconfigure(8, weight=1)
        ttk.Label(frame, text="从演示视频整理吉他谱", font=("Microsoft YaHei UI", 17, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))
        ttk.Label(frame, text="视频文件").grid(row=1, column=0, sticky="w", padx=(0, 10)); ttk.Entry(frame, textvariable=self.video).grid(row=1, column=1, sticky="ew", pady=5); ttk.Button(frame, text="选择视频", command=self.pick_video).grid(row=1, column=2, padx=(10, 0))
        ttk.Label(frame, text="输出 PDF").grid(row=2, column=0, sticky="w"); ttk.Entry(frame, textvariable=self.output).grid(row=2, column=1, sticky="ew", pady=5); ttk.Button(frame, text="保存位置", command=self.pick_output).grid(row=2, column=2, padx=(10, 0))
        roi = ttk.Frame(frame); roi.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        self.preview_btn = ttk.Button(roi, text="预览 / 框选谱面", command=self.preview); self.preview_btn.pack(side="left")
        self.auto_btn = ttk.Button(roi, text="恢复自动定位", command=self.reset_region); self.auto_btn.pack(side="left", padx=8); ttk.Label(roi, textvariable=self.region_text).pack(side="left", padx=8)
        opts = ttk.Frame(frame); opts.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        for label, var, width in [("每行目标小节", self.bars, 3), ("采样 fps", self.fps, 4), ("DPI", self.dpi, 5)]:
            ttk.Label(opts, text=label).pack(side="left", padx=(0, 5)); ttk.Entry(opts, textvariable=var, width=width).pack(side="left", padx=(0, 14))
        ttk.Combobox(opts, textvariable=self.color, values=list(COLORS), state="readonly", width=14).pack(side="left")
        ttk.Checkbutton(frame, text="保存过程截图，便于检查符干、连音线和小节边界", variable=self.debug).grid(row=5, column=0, columnspan=3, sticky="w", pady=7)
        actions = ttk.Frame(frame); actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)
        self.run_btn = ttk.Button(actions, text="开始提取", command=self.start); self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(actions, text="取消", command=self.cancel, state="disabled"); self.cancel_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(actions, text="打开 PDF", command=self.open_pdf, state="disabled"); self.open_btn.pack(side="left")
        self.report_btn = ttk.Button(actions, text="查看检查结果", command=self.show_report, state="disabled"); self.report_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(frame, maximum=100); self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        logs = ttk.Frame(frame); logs.grid(row=8, column=0, columnspan=3, sticky="nsew")
        self.log_text = tk.Text(logs, state="disabled", height=13, wrap="word", font=("Consolas", 10)); self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(logs, command=self.log_text.yview); scroll.pack(side="right", fill="y"); self.log_text.config(yscrollcommand=scroll.set)
        ttk.Label(frame, textvariable=self.status, wraplength=780).grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))
        root.protocol("WM_DELETE_WINDOW", self.close); root.after(80, self.poll)

    def pick_video(self):
        path = filedialog.askopenfilename(title="选择演示视频", filetypes=[("视频", VIDEO_EXTS), ("所有文件", "*.*")])
        if path:
            old = self.video.get(); self.video.set(path); self.reset_region()
            if not self.output.get() or self.output.get() == os.path.splitext(old)[0] + "_吉他谱.pdf": self.output.set(os.path.splitext(path)[0] + "_吉他谱.pdf")

    def pick_output(self):
        path = filedialog.asksaveasfilename(title="保存 PDF", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path: self.output.set(path)

    def reset_region(self): self.region = None; self.region_source = None; self.region_text.set("自动定位谱面")

    def valid_video(self):
        path = self.video.get().strip()
        if not Path(path).is_file(): raise ValueError("请先选择有效的视频文件")
        return path

    def busy(self, value):
        for b in (self.run_btn, self.preview_btn, self.auto_btn): b.config(state="disabled" if value else "normal")
        self.cancel_btn.config(state="normal" if value else "disabled")

    def launch(self, fn):
        if self.worker and self.worker.is_alive(): return
        self.cancel_event.clear(); self.busy(True)
        def task():
            try: fn()
            except CancelledError: self.events.put(("cancelled", None))
            except Exception: self.events.put(("error", traceback.format_exc()))
            finally: self.events.put(("finished", None))
        self.worker = threading.Thread(target=task, daemon=True); self.worker.start()

    def preview(self):
        try: path = self.valid_video()
        except ValueError as error: messagebox.showerror("视频文件", str(error)); return
        self.status.set("正在读取预览并定位谱面 ..."); self.preview_path = path; selected = self.region if self.region_source == path else None
        def task():
            try: region, _, frame = inspect_video(path, selected, self.cancel_event)
            except ValueError:
                cap = cv2.VideoCapture(path)
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 5)); ok, frame = cap.read()
                    if not ok: raise OSError("无法读取预览")
                finally: cap.release()
                region = None
            self.events.put(("preview", (frame, region, path)))
        self.launch(task)

    def apply_region(self, region, path):
        self.region = region; self.region_source = path; self.region_text.set("已框选：" + str(region)); self.status.set("谱面范围已设置，开始提取即可。")

    def start(self):
        try:
            video = self.valid_video(); fps = float(self.fps.get()); dpi = int(self.dpi.get()); bars = int(self.bars.get()); validate_options(fps, dpi, bars)
            out = self.output.get().strip() or os.path.splitext(video)[0] + "_吉他谱.pdf"; self.output.set(out)
            region = self.region if self.region_source == video else None; polarity = COLORS[self.color.get()]
            debug = str(Path(out).parent / ("debug_" + Path(video).stem)) if self.debug.get() else None
        except (ValueError, KeyError) as error: messagebox.showerror("检查设置", str(error)); return
        self.summary = None; self.open_btn.config(state="disabled"); self.report_btn.config(state="disabled"); self.progress["value"] = 0
        self.log_text.config(state="normal"); self.log_text.delete("1.0", "end"); self.log_text.config(state="disabled")
        def task():
            summary = run_pipeline(video, out, sample_fps=fps, dpi=dpi, bars_per_row=bars, region=region, polarity=polarity, debug_dir=debug, cancel_event=self.cancel_event, log=lambda msg: self.events.put(("log", msg)), progress_cb=lambda pct, msg: self.events.put(("progress", (pct, msg))))
            self.events.put(("done", summary))
        self.launch(task)

    def cancel(self): self.cancel_event.set(); self.cancel_btn.config(state="disabled"); self.status.set("正在取消，请稍候 ...")

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log": self.append_log(payload)
                elif kind == "progress": self.progress["value"] = payload[0]; self.status.set(payload[1])
                elif kind == "preview":
                    frame, region, path = payload
                    if not self.closing: RegionDialog(self.root, frame, region, lambda value: self.apply_region(value, path))
                elif kind == "done":
                    self.summary = payload; self.open_btn.config(state="normal"); self.report_btn.config(state="normal")
                    text = f"完成：{payload['pdf_pages']} 页 PDF，{payload['score_rows']} 行谱表。"
                    if payload["warnings"]: text += f" 有 {len(payload['warnings'])} 项待核对，请点“查看检查结果”。"
                    self.status.set(text)
                elif kind == "cancelled": self.status.set("已取消。")
                elif kind == "error":
                    self.append_log(payload); self.status.set("提取失败，详见日志。")
                    if not self.closing: messagebox.showerror("处理失败", payload.strip().splitlines()[-1])
                elif kind == "finished":
                    self.busy(False)
                    if self.closing: self.root.destroy(); return
        except queue.Empty: pass
        self.root.after(80, self.poll)

    def append_log(self, msg):
        self.log_text.config(state="normal"); self.log_text.insert("end", str(msg) + "\n"); self.log_text.see("end"); self.log_text.config(state="disabled")

    def open_pdf(self):
        if self.summary:
            try: open_file(self.summary["output_pdf"])
            except OSError as error: messagebox.showerror("无法打开", str(error))

    def show_report(self):
        if not self.summary: return
        r = self.summary; win = tk.Toplevel(self.root); win.title("提取检查结果"); win.geometry("740x420")
        box = tk.Text(win, wrap="word", padx=14, pady=14); box.pack(fill="both", expand=True)
        text = f"源页面：{r['video_pages']}\n谱表行数：{r['score_rows']}\nPDF 页数：{r['pdf_pages']}\n"
        text += f"图像边界推定的小节单元：{r['measures']}（完整边界：{r['complete_measure_units']}）\n\n"
        if r.get("rows"): text += "各行小节数：" + " / ".join(str(row["units"]) for row in r["rows"]) + "\n"
        if "crossing_marks" in r:
            marks = r["crossing_marks"]; kept = sum(mark["kept_on_same_row"] for mark in marks); text += f"跨小节连线：检测到 {len(marks)} 处，其中 {kept} 处保持同行。\n\n"
        text += "待核对内容：\n" + ("\n".join(r["warnings"]) if r["warnings"] else "未发现接缝或边界异常。建议对照过程截图核对细小技巧标记。")
        text += f"\n\nPDF：{r['output_pdf']}\n详细报告：{r['report_path']}"
        box.insert("1.0", text); box.config(state="disabled")

    def close(self):
        if self.worker and self.worker.is_alive(): self.closing = True; self.cancel()
        else: self.root.destroy()


def main():
    root = tk.Tk()
    try: ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError: pass
    App(root); root.mainloop()


if __name__ == "__main__":
    main()
