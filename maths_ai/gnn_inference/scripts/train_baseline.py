from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from atp_lean_gnn.training import train_main


if __name__ == "__main__":
    raise SystemExit(train_main())
