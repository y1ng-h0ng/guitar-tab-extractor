"""One-image-at-a-time review navigator. Widgets stay on Tk's main thread."""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

COLORS = {"自动识别": "auto", "白字 / 暗底": "bright", "黑字 / 白底": "dark"}


class ReviewDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("检查片段 / 局部调整")
        self.geometry("1180x800")
        self.minsize(900, 650)
        self.selected_id = None
        self.issue = None
        self.mode = tk.StringVar(value="提取片段")
        self.zoom = tk.StringVar(value="适应窗口")
        self.start = tk.StringVar(); self.end = tk.StringVar()
        self.polarity = tk.StringVar(); self.gain = tk.StringVar()
        self.time = tk.StringVar(); self.row = tk.StringVar()
        self.description = tk.StringVar(); self.bounds = tk.StringVar()
        self.overview = tk.StringVar()
        self.drawing = tk.StringVar(value='浏览')
        self.drag = None; self.image_scale = 1.
        ttk.Label(self, textvariable=self.overview, padding=10, wraplength=1100).pack(fill="x")
        body = ttk.Panedwindow(self, orient="horizontal"); body.pack(fill="both", expand=True, padx=10)
        left = ttk.Frame(body); right = ttk.Frame(body); body.add(left, weight=1); body.add(right, weight=4)
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        self.tree.column("#0", width=285, stretch=True)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(left, command=self.tree.yview); scroll.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.select)
        toolbar = ttk.Frame(right); toolbar.pack(fill="x")
        modes = ttk.Combobox(toolbar, textvariable=self.mode, state="readonly", width=12,
                             values=("提取片段", "原视频画面", "生成谱行"))
        modes.pack(side="left"); modes.bind("<<ComboboxSelected>>", lambda _: self.render())
        zoom = ttk.Combobox(toolbar, textvariable=self.zoom, state="readonly", width=10,
                            values=("适应窗口", "100%", "200%"))
        zoom.pack(side="left", padx=6); zoom.bind("<<ComboboxSelected>>", lambda _: self.render())
        ttk.Label(toolbar, text="相关谱行").pack(side="left")
        self.row_box = ttk.Combobox(toolbar, textvariable=self.row, state="readonly", width=6)
        self.row_box.pack(side="left", padx=4)
        self.row_box.bind("<<ComboboxSelected>>", lambda _: self.show_row())
        ttk.Button(toolbar, text="全部提示", command=self.show_warnings).pack(side="right")
        display = ttk.Frame(right); display.pack(fill="both", expand=True, pady=8)
        display.rowconfigure(0, weight=1); display.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(display, background="#ddd", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sx = ttk.Scrollbar(display, orient="horizontal", command=self.canvas.xview)
        sy = ttk.Scrollbar(display, orient="vertical", command=self.canvas.yview)
        sx.grid(row=1, column=0, sticky="ew"); sy.grid(row=0, column=1, sticky="ns")
        self.canvas.config(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind('<ButtonPress-1>', self.begin_region)
        self.canvas.bind('<B1-Motion>', self.move_region)
        self.canvas.bind('<ButtonRelease-1>', self.end_region)
        ttk.Label(right, textvariable=self.description, wraplength=800).pack(fill="x")
        locate = ttk.Frame(right); locate.pack(fill="x", pady=6)
        ttk.Label(locate, text="视频时间（秒）").pack(side="left")
        ttk.Entry(locate, textvariable=self.time, width=10).pack(side="left", padx=5)
        ttk.Button(locate, text="定位原画面", command=self.locate).pack(side="left")
        settings = ttk.LabelFrame(self, text="仅调整选中片段", padding=10)
        settings.pack(fill="x", padx=10, pady=10)
        ttk.Label(settings, textvariable=self.bounds).grid(row=0, column=0, columnspan=10, sticky="w", pady=(0, 8))
        for col, label, var in ((0, "采样开始（秒）", self.start), (2, "结束（秒）", self.end)):
            ttk.Label(settings, text=label).grid(row=1, column=col, sticky="w")
            ttk.Entry(settings, textvariable=var, width=9).grid(row=1, column=col+1, padx=5)
        ttk.Combobox(settings, textvariable=self.polarity, values=list(COLORS), state="readonly", width=15).grid(row=1, column=4, padx=5)
        ttk.Label(settings, text="细节保留").grid(row=1, column=5)
        ttk.Spinbox(settings, textvariable=self.gain, from_=.5, to=2, increment=.1, width=5).grid(row=1, column=6, padx=5)
        actions = ttk.Frame(settings)
        actions.grid(row=3, column=0, columnspan=10, sticky="w", pady=(8, 0))
        self.apply_btn = ttk.Button(actions, text="只重处理此片段", command=self.apply)
        self.apply_btn.pack(side="left")
        self.reset_btn = ttk.Button(actions, text="恢复默认并重处理", command=self.reset)
        self.reset_btn.pack(side="left", padx=8)
        self.draw_box = ttk.Combobox(actions, textvariable=self.drawing, state='readonly',
                                    width=12, values=('浏览', '框选擦除', '框选保护'))
        self.draw_box.pack(side='left', padx=8)
        self.draw_box.bind('<<ComboboxSelected>>', self.choose_drawing)
        self.undo_btn = ttk.Button(actions, text='撤销上次框选', command=self.undo_region)
        self.undo_btn.pack(side='left')
        ttk.Label(settings, text="1.0 为默认；增大可保留更淡的笔画，也可能带回杂线。其他片段的参数不变；接缝与最终分页会重新计算。",
                  wraplength=1080).grid(row=2, column=0, columnspan=10, sticky="w", pady=(8, 0))
        ttk.Label(settings, text='在“提取片段”上拖动框选，松开后更新 PDF；红色擦除，绿色保护，重叠时保护优先。保护仅恢复源画面有证据的笔画。',
                  wraplength=1080).grid(row=4, column=0, columnspan=10, sticky='w', pady=(6, 0))
        self.refresh()

    def record(self):
        return next((r for r in self.app.review.records if r["id"] == self.selected_id), None)

    def refresh(self):
        session = self.app.review
        if not session.summary:
            return
        report = session.summary
        self.overview.set(f"{report['pdf_pages']} 页 PDF，{report['score_rows']} 行；{len(report.get('issues', []))} 处图像疑点。"
                          "点击问题定位。红框表示核对范围，不代表已识别出缺失音符。\n"
                          "相关谱行可能包含同一连续片段的多行，可逐行查看。局部修改随处理缓存保留；清理缓存后不再保留。")
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        for record in session.records:
            sid = record["id"]
            seg = record["segment"]
            parent = f"segment-{sid}"
            issues = [i for i in report.get("issues", []) if i["segment_id"] == sid]
            self.tree.insert("", "end", iid=parent, text=f"片段 {sid} · {seg.start:.2f}–{seg.end:.2f}s" + (f" · {len(issues)} 项" if issues else ""), open=bool(issues))
            self.items[parent] = (sid, None)
            for n, issue in enumerate(issues):
                item = f"{parent}-{n}"
                self.tree.insert(parent, "end", iid=item, text=issue["message"])
                self.items[item] = (sid, issue)
        if session.records:
            target = f"segment-{self.selected_id or session.records[0]['id']}"
            self.tree.selection_set(target); self.tree.see(target); self.select()

    def select(self, *_):
        if self.app.worker and self.app.worker.is_alive():
            return
        selection = self.tree.selection()
        if not selection or selection[0] not in self.items:
            return
        self.selected_id, self.issue = self.items[selection[0]]
        record = self.record(); segment = record["segment"]
        original = record["original"]
        self.start.set(f"{segment.start:.3f}"); self.end.set(f"{segment.end:.3f}")
        self.gain.set(str(record["detail_gain"]))
        self.polarity.set(next(k for k, v in COLORS.items() if v == record["polarity"]))
        self.time.set(f"{self.issue['time_seconds'] if self.issue else (segment.start+segment.end)/2:.3f}")
        self.bounds.set(f"片段 {self.selected_id}；允许采样范围 {original.start:.3f}–{original.end:.3f} 秒")
        self.description.set(self.issue["message"] if self.issue else "选择左侧疑点可显示核对范围；也可以直接调整此片段。")
        self.drag = None
        self.undo_btn.config(state='normal' if record.get('regions') else 'disabled')
        page = next((v["page"] for v in self.app.review.summary["source_views"] if v["segment_id"] == self.selected_id), None)
        rows = [str(i+1) for i, row in enumerate(self.app.review.summary["rows"]) if page in row.get("source_pages", [])]
        self.row_box.config(values=rows); self.row.set(rows[0] if rows else "")
        self.render()

    def show_row(self):
        self.mode.set("生成谱行"); self.render()

    def show_warnings(self):
        report = self.app.review.summary
        win = tk.Toplevel(self); win.title("全部提取提示"); win.geometry("700x430")
        box = tk.Text(win, wrap="word", padx=12, pady=12)
        box.pack(fill="both", expand=True)
        box.insert("end", "\n".join(report["warnings"]) or "没有接缝或边界提示。")
        box.insert("end", "\n\n图像疑点：\n" + "\n".join(
            f"片段 {item['segment_id']} · {item['time_seconds']:.2f}s：{item['message']}"
            for item in report.get("issues", [])))
        box.insert("end", f"\n\nPDF：{report['output_pdf']}")
        box.config(state="disabled")

    def locate(self):
        self.mode.set("原视频画面"); self.render()

    def render(self):
        if self.app.worker and self.app.worker.is_alive():
            return
        record = self.record()
        if record is None or not self.winfo_exists():
            return
        box = self.issue["bbox"] if self.issue else None
        try:
            if self.mode.get() == "原视频画面":
                time = float(self.time.get())
                original = record["original"]
                if not original.start-.001 <= time <= original.end+.001:
                    raise ValueError("定位时间需位于当前片段范围内")
                cap = cv2.VideoCapture(self.app.review.config["video_path"])
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, time*1000)
                    ok, image = cap.read()
                finally:
                    cap.release()
                if not ok:
                    raise ValueError("无法读取此时间的原画面")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                x, y, w, h = self.app.review.summary["region"]
                if box:
                    px, py, pw, ph = box
                    sh, sw = record["page"].shape
                    box = [x+px*w/sw, y+py*h/sh, pw*w/sw, ph*h/sh]
                else:
                    box = [x, y, w, h]
            elif self.mode.get() == "生成谱行":
                if not self.row.get():
                    raise ValueError("此片段未进入最终谱面，可查看提取片段并尝试局部重处理")
                image = self.app.review.rows[int(self.row.get())-1]
                box = None
            else:
                image = record["page"]
            photo = Image.fromarray(image)
            scale = (min(max(100, self.canvas.winfo_width()-10)/photo.width,
                         max(100, self.canvas.winfo_height()-10)/photo.height, 1)
                     if self.zoom.get() == "适应窗口" else float(self.zoom.get()[:-1])/100)
            photo = photo.resize((max(1, round(photo.width*scale)), max(1, round(photo.height*scale))), Image.Resampling.LANCZOS)
            self.image_scale = scale
            self.photo = ImageTk.PhotoImage(photo)
            self.canvas.delete("all"); self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            if box:
                x, y, w, h = box
                self.canvas.create_rectangle(x*scale, y*scale, (x+w)*scale, (y+h)*scale, outline="#e14631", width=2)
            if self.mode.get() == '提取片段':
                for index, operation in enumerate(record.get('regions', []), 1):
                    x, y, w, h = operation['bbox']
                    color = '#c93636' if operation['kind'] == 'erase' else '#16804a'
                    self.canvas.create_rectangle(x*scale, y*scale, (x+w)*scale, (y+h)*scale, outline=color, width=2)
                    self.canvas.create_text(x*scale+3, y*scale+3, text=str(index), fill=color, anchor='nw')
            self.canvas.config(scrollregion=(0, 0, photo.width, photo.height))
        except (ValueError, OSError, cv2.error) as error:
            self.canvas.delete("all")
            self.canvas.create_text(20, 30, text=str(error), anchor="nw", width=600)

    def apply(self):
        if self.record() is None or (self.app.worker and self.app.worker.is_alive()):
            return
        try:
            values = (self.selected_id, float(self.start.get()), float(self.end.get()),
                      COLORS[self.polarity.get()], float(self.gain.get()))
        except (ValueError, KeyError):
            messagebox.showerror("检查局部设置", "请填写有效的时间、颜色模式和细节系数", parent=self)
            return
        self.app.reprocess_segment(*values)

    def choose_drawing(self, _=None):
        self.drag = None
        if self.drawing.get() != '浏览':
            self.mode.set('提取片段')
        self.canvas.config(cursor='crosshair' if self.drawing.get() != '浏览' else '')
        self.render()

    def region_point(self, event):
        h, w = self.record()['page'].shape
        return (max(0, min(w, round(self.canvas.canvasx(event.x)/self.image_scale))),
                max(0, min(h, round(self.canvas.canvasy(event.y)/self.image_scale))))

    def begin_region(self, event):
        if (self.app.worker and self.app.worker.is_alive()) or self.record() is None:
            return
        if self.mode.get() != '提取片段' or self.drawing.get() == '浏览': return
        self.drag = self.region_point(event)

    def move_region(self, event):
        if self.drag is None: return
        x, y = self.region_point(event); a, b = self.drag; s = self.image_scale
        self.canvas.delete('pending-region')
        self.canvas.create_rectangle(a*s, b*s, x*s, y*s, outline='#1467b0', width=2, tags='pending-region')

    def end_region(self, event):
        if self.drag is None: return
        if self.mode.get() != '提取片段':
            self.drag = None; self.canvas.delete('pending-region'); return
        x, y = self.region_point(event); a, b = self.drag; self.drag = None
        self.canvas.delete('pending-region')
        if abs(x-a) < 2 or abs(y-b) < 2: return
        kind = 'erase' if self.drawing.get() == '框选擦除' else 'protect'
        self.app.edit_regions(self.selected_id, kind, [min(a,x), min(b,y), abs(x-a), abs(y-b)])

    def undo_region(self):
        if self.record() is not None:
            self.app.edit_regions(self.selected_id, undo=True)

    def reset(self):
        record = self.record()
        if record is None:
            return
        original = record["original"]
        self.start.set(str(original.start)); self.end.set(str(original.end)); self.gain.set("1.0")
        polarity = self.app.review.config["polarity"]
        self.polarity.set(next(k for k, v in COLORS.items() if v == polarity))
        self.apply()

    def busy(self, value):
        self.drag = None
        self.draw_box.config(state='disabled' if value else 'readonly')
        for button in (self.apply_btn, self.reset_btn, self.undo_btn):
            button.config(state="disabled" if value else "normal")
        if not value and self.record() is not None:
            self.undo_btn.config(state='normal' if self.record().get('regions') else 'disabled')

    def destroy(self):
        super().destroy()
        # A later extraction worker may trigger cyclic GC. Release Tcl-owned
        # objects here on the UI thread rather than during that worker's GC.
        self.photo = None
        for name in ("mode", "zoom", "start", "end", "polarity", "gain", "time",
                     "row", "description", "bounds", "overview", "drawing"):
            setattr(self, name, None)
