"""Run the proving service: ``python -m maths_ai.api --config service.json``.

Exactly one Uvicorn process, without reload: it owns the runtime process and
the state directory. Scale Lean work with the configured collection workers,
not with Uvicorn workers.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import CONFIG_ENV, create_app, default_service_factory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MCTS actor-critic proving service")
    parser.add_argument("--config", default=os.environ.get(CONFIG_ENV), help="service JSON config")
    parser.add_argument("--host", default=os.environ.get("MATHS_AI_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MATHS_AI_API_PORT", "8000")))
    args = parser.parse_args(argv)
    if not args.config:
        parser.error(f"--config or {CONFIG_ENV} is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app(default_service_factory(Path(args.config)))
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
