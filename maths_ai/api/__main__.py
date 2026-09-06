"""Run the maths_ai API service: ``python -m maths_ai.api``.

Host/port are configurable via the API settings (``MATHS_AI_API_HOST`` /
``MATHS_AI_API_PORT``) in ``maths_ai.core.config``.
"""

import uvicorn

from maths_ai.api.app import create_app
from maths_ai.core.config import settings


def main() -> None:
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
