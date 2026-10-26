"""Command-line entry point. Run python main.py --help for all options."""

import argparse
from pathlib import Path
import sys
import time
from tabextract import run_pipeline
from tabextract.support import parse_bars_per_row, validate_options, time_bounds
from tabextract.download import VideoSource, video_url, default_output, probe_video
from tabextract.cache import CacheStore
import json


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
    parser.add_argument('--start', help='开始时间：秒数或 时:分:秒')
    parser.add_argument('--end', help='结束时间：秒数或 时:分:秒')
    parser.add_argument('--part', type=int, help='链接中的分 P 序号，从 1 开始')
    parser.add_argument('--format', dest='format_id', help='--info 返回的可用清晰度编号；默认自动')
    parser.add_argument('--info', action='store_true', help='只解析链接，列出分 P 和清晰度，不下载视频')
    parser.add_argument('--cache-dir', help='缓存目录；源码版默认项目内 .tabextract-cache，打包版默认当前用户 AppData')
    parser.add_argument('--no-cache', action='store_true', help='禁用持久缓存，链接下载在退出时清理')
    args = parser.parse_args(argv)
    start = time.time()
    cache = None if args.no_cache else CacheStore(args.cache_dir)
    source = VideoSource(cache)
    try:
        validate_options(args.sample_fps, args.dpi, args.bars_per_row)
        time_range = time_bounds((args.start, args.end)) if args.start or args.end else None
        if args.part is not None and args.part < 1:
            raise ValueError('分 P 序号必须是正整数')
        if cache:
            cache.prune()
        out = args.output or default_output(args.video)
        if Path(out).suffix.lower() != '.pdf':
            raise ValueError('输出文件需要使用 .pdf 扩展名')
        url = video_url(args.video)
        if args.info:
            info = probe_video(url, args.part)
            info.pop('_listing', None)
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
        source_options = dict(part=args.part, format_id=args.format_id)
        local = source.resolve(args.video, **source_options) if url else args.video
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
            source_options=source_options if url else None,
            time_range=time_range,
            cache_store=cache,
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
