"""Общие фикстуры: страницы с «текстом», который OCR обязан прочитать.

Страницы рисуются кодом (`cv2.putText`), без файлов шрифтов: тесты сервиса
должны работать и на Windows, и в slim-образе, где DejaVu лежит по другому пути.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

BACKGROUND = 255
INK = 0
#: Высота текста на тестовой странице: мелкий шрифт RapidOCR не прочитает.
TEXT_SCALE = 1.6


def encode_png(image: np.ndarray) -> bytes:
    """Матрица BGR → байты PNG (формат, которым воркер отправляет страницу)."""
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """Нарисовать крупный текст — так, чтобы движок гарантированно его увидел."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (INK,) * 3, 3,
                cv2.LINE_AA)


def page_array(text: str = "HELLO WORLD", width: int = 900, height: int = 300) -> np.ndarray:
    """Светлая страница с рамочкой панели и крупным текстом внутри."""
    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    cv2.rectangle(image, (5, 5), (width - 6, height - 6), (200, 200, 200), 2)
    draw_text(image, text, (60, 170))
    return image


def two_line_page() -> np.ndarray:
    """Страница с двумя строками: верхняя и нижняя — для проверки порядка чтения."""
    image = np.full((500, 900, 3), BACKGROUND, np.uint8)
    draw_text(image, "FIRST LINE", (60, 120))
    draw_text(image, "SECOND LINE", (60, 420))
    return image


@pytest.fixture
def page_png() -> bytes:
    """Готовая страница в PNG — тело запроса `/recognize`."""
    return encode_png(page_array())


@pytest.fixture
def two_lines_png() -> bytes:
    """Страница с двумя строками — для проверки порядка строк."""
    return encode_png(two_line_page())


@pytest.fixture
def big_page_png() -> bytes:
    """Страница больше 1 Мпикс — для проверки лимита `OCR_MAX_PIXELS`."""
    return encode_png(page_array(width=1100, height=1000))


@pytest.fixture(autouse=True)
def _fresh_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбросить кэш самопроверки `/ready`: иначе первый успешный прогон
    залипает, и тесты неготовности проверяют кэш, а не логику."""
    import ocr_service.recognizer as module

    monkeypatch.setattr(module, "_PROBE_RESULT", None)


@pytest.fixture
def anyio_backend() -> str:
    """Асинхронные тесты идут на asyncio: trio в зависимостях сервиса нет."""
    return "asyncio"
