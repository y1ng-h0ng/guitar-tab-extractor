"""Command-line entry point. Run python main.py --help for all options."""

import argparse
from pathlib import Path
import sys
import time
from tabextract import run_pipeline
from tabextract.support import parse_bars_per_row, validate_options
from tabextract.download import VideoSource, video_url, default_output


def main(argv=None):
    parser = argparse.ArgumentParser(description="从吉他演示视频提取谱面，去重后按完整小节导出 A4 PDF")
    parser.add_argument("video", help="本地视频路径、HTTP(S) 视频链接或含单个链接的分享文本")
    parser.add_argument("-o", "--output", help="输出 PDF；本地视频默认保存在旁边，链接默认保存在 output 文件夹")
    parser.add_argument("--sample-fps", type=float, default=6.0, help="采样频率，0.5–30，默认 6")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 内嵌图像最低 DPI，72–600，默认 300；不降低提取图分辨率")
    parser.add_argument("--bars-per-row", type=parse_bars_per_row, default=None, metavar="auto|1-8",
                        help="默认 auto 自动分行；填写 1–8 自定义每行目标小节数；所有行统一长度")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="手动指定谱面矩形；默认自动定位")
    parser.add_argument("--polarity", choices=["auto", "bright", "dark"], default="auto", help="白字暗底用 bright，黑字白底用 dark")
    parser.add_argument("--debug-dir", help="保存谱面框、清洗前后截图、各行和拼接长图；默认不保存")
    parser.add_argument("--report", action="store_true", help="额外生成 .report.json 检查报告；默认不生成")
    args = parser.parse_args(argv)
    start = time.time()
    source = VideoSource()
    try:
        validate_options(args.sample_fps, args.dpi, args.bars_per_row)
        out = args.output or default_output(args.video)
        if Path(out).suffix.lower() != '.pdf':
            raise ValueError('输出文件需要使用 .pdf 扩展名')
        url = video_url(args.video)
        local = source.resolve(args.video) if url else args.video
        summary = run_pipeline(
            local,
            out,
            sample_fps=args.sample_fps,
            dpi=args.dpi,
            bars_per_row=args.bars_per_row,
            region=args.region,
            polarity=args.polarity,
            debug_dir=args.debug_dir,
            generate_report=args.report,
            source_url=url,
        )
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    finally:
        source.close()
    print(f"\n完成：{summary['video_pages']} 个源页面，{summary['score_rows']} 行谱表，{summary['pdf_pages']} 页 PDF")
    print(f"输出：{summary['output_pdf']}")
    if summary.get("report_path"):
        print(f"报告：{summary['report_path']}")
    print(f"耗时：{time.time() - start:.1f} 秒")
    if summary["warnings"]:
        print("待核对：")
        for warning in summary["warnings"]:
            print("  " + warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
