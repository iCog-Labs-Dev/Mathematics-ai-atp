"""Run the maths_ai API service: ``python -m maths_ai.api``.

Host/port are configurable via the API settings (``MATHS_AI_API_HOST`` /
``MATHS_AI_API_PORT``) in ``maths_ai.core.config``. They can also be
overridden with ``--host`` and ``--port``.
"""

import argparse

import uvicorn

from maths_ai.api.app import create_app
from maths_ai.core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mathematics AI ATP API")
    parser.add_argument("--host", default=settings.api_host, help="Bind address")
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Bind port (default: %(default)s)",
    )
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
