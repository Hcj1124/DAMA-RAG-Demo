"""內建預設路徑固定於此 checkout；CLI 明確傳入的路徑仍以呼叫者工作目錄為準。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(relative: str) -> Path:
    """解析專案內建資源路徑，不受目前程序工作目錄影響。"""
    return PROJECT_ROOT / relative
