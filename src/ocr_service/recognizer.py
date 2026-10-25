"""Распознавание страниц: чистые функции и обёртка над движком RapidOCR.

Модуль держит всю логику контракта `/recognize`: проверку размера, декодирование,
преобразование четырёхточечных боксов движка в прямоугольники контракта,
фильтрацию по уверенности и порядок чтения. Тесты вызывают функции напрямую,
поэтому API-слой остаётся тонкой обвязкой статусов.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ocr_service.config import Settings


@dataclass(frozen=True)
class Box:
    """Рамка строки в пикселях исходной страницы."""

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class TextLine:
    """Распознанная строка: текст, рамка и уверенность."""

    text: str
    box: Box
    confidence: float


@dataclass(frozen=True)
class Recognition:
    """Результат страницы: строки в порядке чтения и размеры страницы."""

    lines: list[TextLine]
    width: int
    height: int


class OcrError(ValueError):
    """Ошибка задания: код уходит в тело ответа и в логи воркера."""

    code = "OCR_FAILED"


class InvalidImage(OcrError):
    """Тело `image` не читается как картинка."""

    code = "INVALID_IMAGE"


class ImageTooLarge(OcrError):
    """Страница больше лимита `OCR_MAX_PIXELS` или `OCR_MAX_BYTES`."""

    code = "IMAGE_TOO_LARGE"


class RecognitionFailed(OcrError):
    """Движок распознавания упал на валидной картинке."""

    code = "OCR_FAILED"


def probe_size(data: bytes) -> tuple[int, int] | None:
    """Размер страницы из заголовка контейнера: `(ширина, высота)` или `None`.

    Нужен, чтобы отклонить «PNG-бомбу» ДО декодирования: файл на десятки
    килобайт разворачивается в гигабайты памяти, и проверка после `imdecode`
    уже бесполезна — под падает по OOM. Читаем только заголовок (PNG IHDR,
    JPEG SOFn, WebP VP8X/VP8/VP8L), а незнакомый формат отдаём декодеру и
    обычной проверке площади.
    """
    return _png_size(data) or _jpeg_size(data) or _webp_size(data)


def _png_size(data: bytes) -> tuple[int, int] | None:
    """IHDR идёт сразу после сигнатуры: ширина и высота — big-endian uint32."""
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height) if width and height else None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Идём по маркерам до SOF0..SOF15 (кроме DHT/JPG/DAC), где лежат размеры."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        # маркеры без длины сегмента
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        is_sof = 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        if is_sof:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return (width, height) if width and height else None
        if length < 2:
            return None
        index += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    """`VP8X` хранит размеры минус единицу в 24 битах, `VP8`/`VP8L` — иначе."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    kind = data[12:16]
    if kind == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)
    if kind == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height)
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def decode_image(data: bytes, settings: Settings | None = None) -> np.ndarray:
    """Декодировать картинку в BGR-матрицу OpenCV.

    Лимит `OPENCV_IO_MAX_IMAGE_PIXELS` выставляется окружением процесса (образ,
    манифест): OpenCV читает его один раз за жизнь процесса, поэтому запись из
    кода на горячем пути бесполезна. Своё исключение лимита переводим в
    контрактный 413 — иначе клиент получал бы 500 на «бомбе».
    """
    cfg = settings or Settings()
    if not data:
        raise InvalidImage("пустое тело image")
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageTooLarge(
            f"страница больше лимита {cfg.max_pixels} px (предохранитель OpenCV)"
        ) from exc
    if image is None:
        raise InvalidImage("image не читается как картинка (PNG/WEBP/JPEG) либо больше лимита")
    return image


def page_size(data: bytes) -> tuple[int, int] | None:
    """Размер страницы в пикселях — по возможности до декодирования.

    Быстрый путь — свой разбор заголовков PNG/JPEG/WebP. Остальные форматы,
    которые понимает OpenCV (BMP, TIFF, PXM), разбираются лениво через Pillow:
    он читает только заголовок и не выделяет память под пиксели.
    """
    size = probe_size(data)
    if size is not None:
        return size
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def build_lines(
    raw: Any, *, min_confidence: float, max_lines: int, width: int, height: int
) -> list[TextLine]:
    """Превратить ответ движка в строки контракта.

    Отбрасываются: пустой текст, строки ниже порога уверенности, рамки за
    границей страницы и вырожденные рамки. Порядок — сверху вниз, затем слева
    направо: вёрстка перевода опирается на него, чтобы диалог читался в том же
    порядке, что и в оригинале.
    """
    lines: list[TextLine] = []
    for item in raw or []:
        # формат RapidOCR: [[4 точки], текст, уверенность строкой]
        if not isinstance(item, list | tuple) or len(item) < 3:
            continue
        points, text, score = item[0], item[1], item[2]
        text = str(text or "").strip()
        if not text:
            continue
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        box = _box_from_points(points, width, height)
        if box is None:
            continue
        lines.append(TextLine(text=text, box=box, confidence=confidence))

    # порядок чтения: сверху вниз, затем слева направо
    lines.sort(key=lambda line: (line.box.y, line.box.x))
    return lines[:max_lines]


def _box_from_points(points: Any, width: int, height: int) -> Box | None:
    """Прямоугольник, охватывающий контур движка, обрезанный по границе страницы."""
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    left = max(0, min(width, int(round(min(xs)))))
    top = max(0, min(height, int(round(min(ys)))))
    right = max(0, min(width, int(round(max(xs)))))
    bottom = max(0, min(height, int(round(max(ys)))))
    if right <= left or bottom <= top:
        return None
    return Box(x=left, y=top, w=right - left, h=bottom - top)


def recognize(data: bytes, settings: Settings, *, runner: Any = None) -> Recognition:
    """Проверить размер, декодировать и распознать строки страницы."""
    # размер проверяется до декодирования: у «бомбы» заголовок читается, а
    # пиксели выделять уже нельзя
    size = page_size(data)
    if size is not None and size[0] * size[1] > settings.max_pixels:
        raise ImageTooLarge(
            f"страница {size[0]}×{size[1]} больше лимита {settings.max_pixels} px"
        )

    page = decode_image(data, settings)
    height, width = page.shape[:2]
    if width * height > settings.max_pixels:
        raise ImageTooLarge(f"страница {width}×{height} больше лимита {settings.max_pixels} px")

    run = runner or _run_engine
    try:
        raw = run(page)
    except OcrError:
        raise
    except Exception as exc:  # noqa: BLE001 - отказ сервиса обязан быть читаемым
        raise RecognitionFailed(f"распознавание упало: {exc}") from exc

    lines = build_lines(
        raw,
        min_confidence=settings.min_confidence,
        max_lines=settings.max_lines,
        width=width,
        height=height,
    )
    return Recognition(lines=lines, width=width, height=height)


#: Ленивый синглтон движка: ONNX-модели грузятся секунды, поэтому это делается
#: при первом задании, а не при импорте модуля (иначе `/health` не успевал бы).
_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()

#: RapidOCR не потокобезопасен: вызовы сериализуются, параллелизм заданий
#: ограничивается семафором в API-слое.
_CALL_LOCK = threading.Lock()


def _engine() -> Any:
    """Движок RapidOCR, созданный один раз на процесс."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                from rapidocr_onnxruntime import RapidOCR

                _ENGINE = RapidOCR()
    return _ENGINE


def _run_engine(page: np.ndarray) -> Any:
    """Прогнать страницу через настоящий движок под замком."""
    engine = _engine()
    with _CALL_LOCK:
        result, _elapsed = engine(page)
    return result or []


class Engine:
    """Обёртка над движком: ленивая инициализация и сериализация вызовов.

    `runner` подменяется в тестах: API-слой и разбор ответа проверяются без
    ONNX-инференса, а реальный движок остаётся один на процесс.
    """

    def __init__(self, settings: Settings, *, runner: Any = None) -> None:
        self._settings = settings
        self._runner = runner

    def recognize(self, data: bytes) -> Recognition:
        """Распознать страницу; движок инициализируется при первом вызове."""
        return recognize(data, self._settings, runner=self._runner)


#: Вердикт самопроверки и момент его получения. Кэш ограничен по времени:
#: сломанный после старта движок обязан снова давать 503, иначе pod навсегда
#: остаётся Ready (liveness движок не трогает и рестарта не будет).
_PROBE_RESULT: tuple[bool, str | None] | None = None
_PROBE_CHECKED_AT = 0.0
_PROBE_LOCK = threading.Lock()


def readiness(
    settings: Settings | None = None,
    *,
    ttl: float | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[bool, str | None]:
    """Готовность сервиса: `(ok, причина)` для пробы `/ready`.

    Проба не спрашивает «импортировался ли RapidOCR» — при сломанной сборке
    импорт уронил бы сам процесс. Она делает настоящий мини-прогон на
    синтетической странице с крупным текстом: это ровно тот путь, которым
    пойдёт живое задание. Успешный вердикт кэшируется на `readiness_ttl`,
    отрицательный не кэшируется вовсе: стартующие модели должны получить
    второй шанс на следующей пробе.
    """
    global _PROBE_RESULT, _PROBE_CHECKED_AT
    cfg = settings or Settings()
    lifetime = cfg.readiness_ttl if ttl is None else ttl
    now = (clock or time.monotonic)()

    cached = _PROBE_RESULT
    if cached is not None and cached[0] and now - _PROBE_CHECKED_AT < lifetime:
        return cached
    with _PROBE_LOCK:
        cached = _PROBE_RESULT
        now = (clock or time.monotonic)()
        if cached is not None and cached[0] and now - _PROBE_CHECKED_AT < lifetime:
            return cached
        result = _self_check(cfg)
        _PROBE_CHECKED_AT = now
        _PROBE_RESULT = result if result[0] else None
        return result


def _self_check(settings: Settings | None = None) -> tuple[bool, str | None]:
    """Мини-прогон распознавания: ловит битые ONNX-модели до первого задания."""
    cfg = settings or Settings()
    page = np.full((300, 900, 3), 255, np.uint8)
    cv2.putText(page, "READY", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3,
                cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", page)
    if not ok:
        return False, "opencv не может отдать PNG"
    try:
        result = recognize(buffer.tobytes(), cfg)
    except OcrError as exc:
        return False, f"движок не распознаёт: {exc}"
    except Exception as exc:  # noqa: BLE001 - любая причина значит «не готов»
        return False, f"движок не распознаёт: {exc}"
    if not any("READY" in line.text.upper() for line in result.lines):
        return False, "движок не прочитал контрольную строку"
    return True, None

