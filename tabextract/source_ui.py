"""Optional source details and cache management windows (Tk main thread only)."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2

from .cache import Checkpoint


def describe(metadata):
    duration = metadata.get('duration')
    length = f'{duration:.1f} 秒' if duration is not None else '时长未知'
    resolution = f"{metadata.get('width') or '?'} × {metadata.get('height') or '?'}"
    return f"{metadata['title']}\n{length} · {resolution} · 分 P {metadata['part']} / {len(metadata['parts'])}"


class LinkDialog(tk.Toplevel):
    def __init__(self, app, metadata, preview):
        super().__init__(app.root)
        self.app = app
        self.title('链接信息与视频选择'); self.geometry('760x620')
        self.transient(app.root)
        self.description = tk.StringVar()
        ttk.Label(self, textvariable=self.description, wraplength=710, padding=12).pack(fill='x')
        self.preview = ttk.Label(self, text='此视频没有可用的预览图', anchor='center')
        self.preview.pack(fill='both', expand=True, padx=12)
        opts = ttk.Frame(self, padding=12); opts.pack(fill='x')
        ttk.Label(opts, text='分 P').grid(row=0, column=0, sticky='w')
        self.part = ttk.Combobox(opts, state='readonly', width=66)
        self.part.grid(row=0, column=1, sticky='ew', pady=6)
        self.part.bind('<<ComboboxSelected>>', self.choose_part)
        ttk.Label(opts, text='清晰度').grid(row=1, column=0, sticky='w')
        self.quality = ttk.Combobox(opts, state='readonly', width=66)
        self.quality.grid(row=1, column=1, sticky='ew', pady=6)
        ttk.Label(self, text='仅列出平台当前允许访问的清晰度；自动优先选择可解码的高清画面。\n预览为平台封面，生成时使用实际视频帧。处理时间可在主窗口填写。',
                  wraplength=710, padding=12).pack(fill='x')
        buttons = ttk.Frame(self, padding=12); buttons.pack(fill='x')
        self.apply_btn = ttk.Button(buttons, text='使用这些设置', command=self.apply)
        self.apply_btn.pack(side='right')
        ttk.Button(buttons, text='取消解析', command=app.cancel).pack(side='left')
        self.update_details(metadata, preview)

    def update_details(self, metadata, preview):
        self.metadata = metadata
        self.description.set(describe(metadata) + ('\n仅列出前 200 个分 P。' if metadata.get('limited') else ''))
        self.part.configure(values=[f"{p['index']}. {p['title']}" for p in metadata['parts']])
        self.part.current(metadata['part']-1)
        self.quality.configure(values=['自动（推荐）'] + [f['label'] for f in metadata['formats']])
        selected = self.app.link_options.get('format_id')
        ids = [None] + [f['id'] for f in metadata['formats']]
        self.quality.current(ids.index(selected) if selected in ids else 0)
        self.photo = None
        if preview is not None:
            image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            image.thumbnail((700, 310), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo or '', text='' if self.photo else '平台未提供可读取的预览图')

    def choose_part(self, _=None):
        self.app.fetch_link_details(self.part.current()+1, self.metadata)

    def busy(self, value):
        for box in (self.part, self.quality):
            box.configure(state='disabled' if value else 'readonly')
        self.apply_btn.configure(state='disabled' if value else 'normal')
        if not value:
            # A failed/cancelled fetch must not leave a new part paired with old formats.
            self.part.current(self.metadata['part']-1)

    def apply(self):
        index = self.quality.current()
        fmt = None if index <= 0 else self.metadata['formats'][index-1]['id']
        self.app.apply_link_options(self.metadata, fmt)
        self.destroy()


class CacheDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title('缓存与断点续做'); self.geometry('790x420')
        self.transient(app.root)
        self.info = tk.StringVar()
        ttk.Label(self, textvariable=self.info, padding=12).pack(anchor='w')
        self.tree = ttk.Treeview(self, columns=('source', 'size'), show='headings', selectmode='browse')
        self.tree.heading('source', text='可恢复的提取任务'); self.tree.heading('size', text='缓存大小')
        self.tree.column('source', width=600); self.tree.column('size', width=120)
        self.tree.pack(fill='both', expand=True, padx=12)
        ttk.Label(self, text='默认保留 7 天，超过 2 GiB 时优先清理较旧缓存。正在使用的任务会跳过。\n清理只删除下载副本和处理缓存，保留原视频、PDF 和报告。', padding=12).pack(anchor='w')
        actions = ttk.Frame(self, padding=12); actions.pack(fill='x')
        ttk.Button(actions, text='载入所选任务', command=self.restore).pack(side='left')
        ttk.Button(actions, text='清理过期缓存', command=lambda: self.clean(False)).pack(side='left', padx=8)
        ttk.Button(actions, text='一键清理全部缓存', command=lambda: self.clean(True)).pack(side='right')
        self.refresh()

    def refresh(self):
        entries = self.app.cache.entries()
        self.info.set(f"缓存占用：{sum(e['size'] for e in entries)/1024**2:.1f} MiB　共 {len(entries)} 项")
        self.tree.delete(*self.tree.get_children())
        self.requests = {}
        for entry in sorted(entries, key=lambda e: e['last_used'], reverse=True):
            if entry['identity'].get('kind') != 'extraction':
                continue
            request = Checkpoint(self.app.cache.root / entry['key']).load('request')
            if request:
                self.requests[entry['key']] = request
                options = request.get('source_options') or {}
                span = request.get('time_range')
                label = request['source']
                if options.get('part'):
                    label += f" · P{options['part']}"
                if span:
                    label += f" · {span[0]}–{span[1] if span[1] is not None else '结尾'} 秒"
                label += f" · {request['sample_fps']} fps"
                self.tree.insert('', 'end', iid=entry['key'], values=(label, f"{entry['size']/1024**2:.1f} MiB"))

    def restore(self):
        selection = self.tree.selection()
        if not selection:
            return
        if self.app.worker and self.app.worker.is_alive():
            return
        self.app.restore_request(self.requests[selection[0]])
        self.destroy()

    def clean(self, clear):
        if self.app.worker and self.app.worker.is_alive():
            messagebox.showinfo('正在处理', '请完成或取消当前任务后再清理。', parent=self)
            return
        if clear:
            self.app.close_review(); self.app.source.close()
        removed = self.app.cache.prune(clear=clear)
        self.refresh()
        self.app.status.set(f'已清理 {removed/1024**2:.1f} MiB 缓存；原视频和 PDF 保留。')
