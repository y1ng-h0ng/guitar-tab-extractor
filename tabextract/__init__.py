"""吉他谱视频提取工具包：从教学视频中还原完整吉他谱并导出 PDF。"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
__version__ = "1.0.0"
