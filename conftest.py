"""让测试无需安装即可从 src 导入 kb_agent。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[0] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
