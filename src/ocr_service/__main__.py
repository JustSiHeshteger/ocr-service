"""Точка входа: `python -m ocr_service` поднимает uvicorn с настройками."""

from __future__ import annotations

import logging

import uvicorn

from ocr_service.app import create_app
from ocr_service.config import Settings


def main() -> int:
    """Запустить сервис; настройки читаются один раз и отдаются приложению."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger(__name__).info(
        "ocr-service запущен: %s:%d, max_pixels=%d, min_confidence=%.2f, workers=%d",
        settings.host,
        settings.port,
        settings.max_pixels,
        settings.min_confidence,
        settings.workers,
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
