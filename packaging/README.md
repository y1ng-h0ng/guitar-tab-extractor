# Windows 打包

目标为 Windows 10 1809 / Windows 11 x64，内置 Python、Tcl/Tk、拖放、视频解码、PDF 和链接下载组件。使用 PyInstaller 的目录模式，避免每次打开时解压整个运行时；Inno Setup 提供当前用户安装、快捷方式和卸载。安装不需要管理员权限。

在 Windows x64、包含 Tcl/Tk 的 Python 3.12 环境中：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r packaging/requirements-build.txt
.\.venv\Scripts\python.exe packaging/build_windows.py --iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

安装编译器：[Inno Setup](https://jrsoftware.org/isdl.php)，本次验证版本 6.7.3。构建脚本不自动下载或安装任何工具。默认输出到 `dist/releases`，支持 `--output`；仅需 ZIP 时省略 `--iscc`。先构建、后单独封装时可用 `--reuse-bundle`，该选项不会更新 EXE，发布前请确保它对应当前源码。

产物包含 `Setup-x64.exe`（传入 ISCC 时）、`Portable-x64.zip`、使用说明、构建清单和 SHA-256。不要将构建产物提交 Git。构建清单记录源码提交、未提交状态、Python 和打包依赖版本。发布前应从干净的已提交源码重新构建，或验证构建内容与最终提交一致。

验证实际 EXE：

```powershell
.\dist\GuitarTabExtractor\tabextract-worker.exe --smoke-test .\build\bundle-check
.\dist\GuitarTabExtractor\tabextract-worker.exe --cli "本地视频.mp4" -o "测试.pdf"
.\.venv\Scripts\python.exe packaging/verify_windows.py --bundle .\dist\GuitarTabExtractor --output .\build\验证 --video "本地视频.mp4"
```

`--smoke-test` 构造短视频并验证 GUI/拖放加载、中文路径、视频编码解码、谱面处理、PDF 输出及下载子进程。它生成测试文件；只能传入专用测试目录。窗口版 EXE 也接受此参数。实际发布还需测试 HTTP 链接下载、安装后运行、升级和卸载，卸载后用户输出与缓存应保留。命令行使用控制台 worker，图形界面使用 `GuitarTabExtractor.exe`。

安装包目前未做代码签名；不要通过关闭系统安全检查规避提示。维护者可在分发前用自己的证书签名，再重新生成哈希。

中文安装向导翻译来自 [kira-96/Inno-Setup-Chinese-Simplified-Translation](https://github.com/kira-96/Inno-Setup-Chinese-Simplified-Translation)，MIT 许可证随 `ChineseSimplified.LICENSE` 保存。构建脚本收集所携带运行时的许可证；不要删除产物中的 `licenses`。
