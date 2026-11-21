"""Run the proving service: ``python -m maths_ai.api [--config service.json] [--accepted-gnn]``.

``--config`` (or ``MATHS_AI_SERVICE_CONFIG``) enables the MCTS actor-critic
backend; ``--accepted-gnn`` (or ``MATHS_AI_ACCEPTED_GNN=1``) enables the
accepted-GNN backend, whose paths come from ``maths_ai.core.config``. At least
one is required.

Exactly one Uvicorn process, without reload: it owns the runtime processes and
the state directories. Scale Lean work with the configured collection workers,
not with Uvicorn workers.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import (
    ACCEPTED_ENV,
    CONFIG_ENV,
    _env_flag,
    create_app,
    default_accepted_factory,
    default_service_factory,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mathematics AI ATP proving service")
    parser.add_argument("--config", default=os.environ.get(CONFIG_ENV), help="MCTS service JSON config")
    parser.add_argument(
        "--accepted-gnn",
        action="store_true",
        default=_env_flag(ACCEPTED_ENV),
        help="serve the accepted-GNN backend",
    )
    parser.add_argument("--host", default=os.environ.get("MATHS_AI_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MATHS_AI_API_PORT", "8000")))
    args = parser.parse_args(argv)
    if not args.config and not args.accepted_gnn:
        parser.error(f"--config/{CONFIG_ENV}, --accepted-gnn/{ACCEPTED_ENV}=1, or both are required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app(
        default_service_factory(Path(args.config)) if args.config else None,
        accepted_factory=default_accepted_factory() if args.accepted_gnn else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
