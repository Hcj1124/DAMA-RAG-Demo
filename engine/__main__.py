"""讓 ``python -m engine`` 成為 ``dama-rag`` 命令列程式的別名。"""

from engine.cli import main

raise SystemExit(main())
