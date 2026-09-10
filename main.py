"""Command-line entry point. Run python main.py --help for all options."""

import argparse
import os
import sys
import time
from tabextract import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="从吉他演示视频提取谱面，去重后按完整小节导出 A4 PDF")
    parser.add_argument("video", help="视频路径")
    parser.add_argument("-o", "--output", help="输出 PDF；默认与视频同名加 _吉他谱.pdf")
    parser.add_argument("--sample-fps", type=float, default=6.0, help="采样频率，0.5–30，默认 6")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 内嵌图像 DPI，72–600，默认 300")
    parser.add_argument("--bars-per-row", type=int, default=4, help="每行目标小节数，1–8，默认 4；连音处允许局部调整")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="手动指定谱面矩形；默认自动定位")
    parser.add_argument("--polarity", choices=["auto", "bright", "dark"], default="auto", help="白字暗底用 bright，黑字白底用 dark")
    parser.add_argument("--debug-dir", help="保存谱面框、清洗前后截图、各行和拼接长图；默认不保存")
    parser.add_argument("--report", action="store_true", help="额外生成 .report.json 检查报告；默认不生成")
    args = parser.parse_args(argv)
    out = args.output or os.path.splitext(args.video)[0] + "_吉他谱.pdf"
    start = time.time()
    try:
        summary = run_pipeline(
            args.video,
            out,
            sample_fps=args.sample_fps,
            dpi=args.dpi,
            bars_per_row=args.bars_per_row,
            region=args.region,
            polarity=args.polarity,
            debug_dir=args.debug_dir,
            generate_report=args.report,
        )
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
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
