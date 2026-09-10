# Guitar Tab Video Extractor / 吉他谱视频提取工具

**Turn guitar-tab tutorial videos into clean, printable PDF scores.**  
**把视频里不断出现、翻页的六线谱自动提取、拼接并整理成可打印的 A4 PDF。**

`Video → Detect tablature → Capture stable pages → Stitch overlaps → Reflow measures → PDF`

This project works directly with tablature that is already visible in video frames. It does **not** listen to the audio, guess notes, or regenerate the score with OCR.

本项目直接处理视频画面中**已经存在的六线谱**：定位谱面、识别翻页、提取稳定画面、去除重复区域，再按照小节重新排版输出 PDF。它**不是听音扒谱工具**，也不会用 OCR 重新生成乐谱。

### Why this project? / 它解决什么问题？

很多吉他教学视频会把六线谱固定显示在画面下方，并随着演奏不断翻页。直接截图通常会遇到重复页面、动态背景、翻页重叠、小节断行和排版混乱等问题。

Guitar Tab Video Extractor 尝试把这套过程自动化，将视频中的已有谱面整理成更适合阅读和打印的连续 PDF。

### Highlights / 核心特点

- 🎸 **Made for guitar tablature** — 针对六线谱，而不是通用视频截图工具
- 🎬 **Video-aware page detection** — 根据谱面变化识别稳定页面和翻页
- 🧩 **Overlap stitching** — 自动寻找相邻页面的重叠区域，减少重复内容
- 📏 **Measure-aware layout** — 根据小节线重新分行，而不是机械裁图
- 🔗 **Tie/slur protection** — 尽量避免把跨小节连线拆到两行
- 🖼️ **Preserves the original score image** — 保留原谱面的数字、符干、技巧标记和连线，不重新 OCR 排字
- 🏁 **Final barline** — 在整首谱最后一个小节补上“细线 + 粗线”终止线
- 📄 **Printable A4 output** — 自动缩放、分页并生成 A4 PDF
- 🔒 **Local processing** — 视频和谱面均在本地处理，不需要大模型 API

> **No audio transcription · No OCR reconstruction · No LLM API required at runtime**

---

## Quick Start / 快速开始

### Windows

1. 下载并完整解压项目，不要直接在压缩软件中运行文件。
2. 安装 **Python 3.10+**，建议 Python 3.12，并确保安装 Tcl/Tk 和 pip。
3. 第一次运行时双击 `install_windows.bat` 安装依赖。
4. 双击 `launch_windows.bat` 启动图形界面。
5. 选择视频和 PDF 输出位置，点击“开始提取”。“保存过程截图”和“生成报告文件”默认均关闭，需要时再勾选。

如果已经配置好 Python 环境：

```bash
python -m pip install -r requirements.txt
python gui.py
```

---

## How It Works / 工作流程

```text
视频输入
  ↓
自动 / 手动定位六线谱区域
  ↓
检测稳定页面与翻页
  ↓
多帧采样并清理动态背景干扰
  ↓
匹配相邻页面的重叠谱面
  ↓
拼接为连续长谱
  ↓
检测小节线与跨小节连线
  ↓
重新分行、补终止线并排版为 A4 PDF
```

项目直接从视频画面中提取和拼接已有谱面，不使用大模型 API，也不通过音频重新转谱。

---

## Score Region Selection / 谱面框选

程序会尝试自动定位视频中的六线谱区域。

如果自动定位不准确，可以点击“预览 / 框选谱面”，用鼠标重新框选谱面范围，再点击“使用这个范围”。“恢复自动定位”可以取消手动设置。

框选时应尽量完整包含：

- 六条谱线
- 音符和数字
- H / P 等技巧标记
- 推弦箭头、`full` 等文字
- 连音线、符干和符梁

对于不同颜色的谱面：

- 白色谱字叠在暗色画面上：通常保持“自动识别”，必要时选择“白字 / 暗底”。
- 白纸黑字谱面：必要时选择“黑字 / 白底”。

原视频分辨率越高，细小符号通常越容易保留。提高 DPI 可以改变 PDF 中图像的输出分辨率，但不能恢复源视频中已经丢失的细节。

---

## Output / 输出内容

默认设置下，程序只生成：

- `视频名_吉他谱.pdf`：整理后的 A4 吉他谱 PDF；最后一个小节会补上标准终止线

下面两类输出均为可选项，GUI 默认不勾选：

- `视频名_吉他谱.report.json`：勾选“生成报告文件”后生成，记录页面、接缝、小节边界、排版和待核对项等处理信息
- `debug_视频名/`：勾选“保存过程截图”后生成，保存谱面区域、清洗结果、分行图和拼接长图等中间结果

即使不生成 `.report.json`，本次运行的检查结果仍可在 GUI 的“查看检查结果”中查看。

如果同名 PDF 已存在，只有在新文件成功生成后才会替换。处理失败或取消时不会留下被截断的半成品 PDF。

---

## Features / 功能说明

- 自动检测视频中的六线谱区域，也支持手动框选
- 根据谱面变化检测翻页，而不是单纯依赖整个视频画面的变化
- 对稳定页面进行多帧采样，尽量抑制动态背景和播放指示线等干扰
- 直接保留谱面中的数字、字母、附点、括号、连音线、符干和符梁，不重新 OCR 排版乐谱
- 对相邻页面寻找重叠区域并进行拼接，减少翻页造成的重复内容
- 根据小节线切分连续谱面，并尽量在合适的小节边界换行
- 检测跨小节的连线，排版时尽量避免把连音关系拆到两行
- 在安全空白区域调整行宽，不直接横向拉伸音符或技巧标记
- 在最终谱面的最后一个小节补上细线 + 粗线组成的终止线
- 将完整谱表行排版到 A4 页面中，并自动处理缩放、页边距和分页
- 支持中文路径、处理取消、参数检查和跨平台文件打开

---

## Scope & Limitations / 适用范围和限制

该工具更适合以下类型的视频：

- 谱面位置相对固定
- 画面缩放基本稳定
- 单行六线谱横向翻页
- 相邻页面之间存在一定重叠

对于持续平滑滚动、明显倾斜拍摄、频繁缩放、谱面被大面积遮挡或复杂彩色谱面，自动处理效果可能下降。

图像处理无法恢复视频画面之外、严重模糊或被遮挡的内容。非常细小的技巧符号也会受到源视频清晰度限制。少量与谱面形态相似且长期静止的背景纹理可能被保留下来。

当页面匹配或小节边界不确定时，程序倾向于保留内容并在检查结果中提示，而不是静默删除可能属于乐谱的部分；如果启用了报告文件，这些信息也会写入 `.report.json`。

“每行目标小节数”是排版目标而不是绝对限制。如果某处存在跨小节连线，程序可能允许相邻行的小节数量发生少量变化，以尽量保持连线两端位于同一行。

---

## CLI / 命令行

```bash
python main.py "视频.mp4"
python main.py "视频.mp4" -o "输出.pdf"
python main.py "视频.mp4" --dpi 300 --bars-per-row 4 --sample-fps 6
python main.py "视频.mp4" --report
python main.py "视频.mp4" --debug-dir debug
python main.py "视频.mp4" --region X Y W H --polarity bright
```

常用参数：

- `-o / --output`：指定输出 PDF 路径
- `--sample-fps`：设置翻页检测采样频率
- `--dpi`：设置 PDF 输出 DPI
- `--bars-per-row`：设置每行目标小节数
- `--report`：额外生成 `.report.json` 检查报告；默认不生成
- `--debug-dir`：保存中间处理截图；不指定时默认不保存
- `--region`：手动指定谱面区域
- `--polarity`：指定谱面明暗类型

不同视频的谱面坐标通常不同，因此 `--region` 更适合在已经确定区域坐标时使用；一般情况下建议先使用自动定位或图形界面框选。

---

## Project Structure / 项目结构

```text
main.py                     命令行入口
gui.py                      图形界面入口
install_windows.bat         Windows 依赖安装脚本
launch_windows.bat          Windows 启动脚本
requirements.txt            Python 依赖

tabextract/
├── region.py               谱面区域定位
├── pages.py                稳定页面与翻页检测
├── clean.py                谱面图像清洗
├── geometry.py             谱线和小节线检测
├── stitch.py               相邻页面匹配与拼接
├── rowlayout.py            小节分行、连线保护和行宽处理
├── pdfout.py               A4 PDF 排版输出与终止线
├── pipeline.py             完整处理流程
└── support.py              通用辅助功能
```

---

## Attribution / 代码来源

本项目的实现由 **ChatGPT Work** 中的 **GPT-6 Astra** 生成。

**GPT-5.6 Sol** 负责将项目上传到 GitHub，以及后续 README 和仓库内容维护。

Project implementation was generated with **GPT-6 Astra in ChatGPT Work**.

**GPT-5.6 Sol** was used for GitHub publishing and subsequent README/repository maintenance.

---

## Fonts / 字体

发布包可以附带 `assets/NotoSansSC-Regular.ttf`。如果项目目录中没有该字体，程序会尝试使用系统中文字体，并在必要时使用 ReportLab 提供的中文 CID 字体作为回退。

Noto Sans SC 使用 SIL Open Font License 1.1，相关许可文件见 `assets/OFL.txt`。

---

## Copyright & Responsible Use / 版权与合理使用

本项目是一个用于整理视频画面中**已经可见的六线谱内容**的技术工具。请仅处理你本人拥有、已经获得权利人授权，或在适用法律和平台规则下可以合法处理的内容。

本项目的 MIT License **只授权本仓库中的代码**，并不授予任何原视频、谱面、编曲、音乐作品、教学内容或其他第三方素材的使用权。通过本工具导出的谱面也不会因为本项目采用 MIT License 而自动获得 MIT 授权。

许多创作者会投入时间制作、编排和校对谱面，并通过售卖谱子获得合理收益。请尊重这些创作者的劳动与商业模式，**不要使用本工具未经许可提取、传播、公开发布或替代购买付费谱面**。

用户应自行判断其对输入视频及其中谱面内容的使用是否获得授权并符合适用的版权法律、平台规则和其他权利要求。

This project is a technical tool for organizing tablature that is **already visible in video frames**. Please use it only with content that you own, are authorized to use, or are otherwise legally permitted to process under applicable law and platform rules.

The MIT License for this repository **applies only to the source code in this project**. It does not grant any rights to source videos, tablature, arrangements, musical compositions, educational content, or other third-party material. Output generated by this tool does not automatically become MIT-licensed merely because the software itself is MIT-licensed.

Many creators invest substantial time in creating, arranging, and proofreading tablature and may rely on selling their scores for income. Please respect their work and business models. **Do not use this tool to extract, redistribute, publicly share, or substitute for purchasing paid tablature without permission.**

Users are responsible for ensuring that their use of source videos and tablature complies with applicable copyright law, platform rules, and the rights of content creators.

---

## License / 许可证

本项目代码采用 **MIT License**。你可以在遵守许可证条款的前提下自由使用、复制、修改、合并、发布和分发本项目代码。该许可仅适用于本项目代码，不覆盖通过本工具处理的第三方视频、谱面或其他内容。

This project's source code is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details. This license applies to the project code only and does not cover third-party videos, tablature, or other content processed with the tool.
