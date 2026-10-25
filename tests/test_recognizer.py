"""Тесты распознавания: рамки, фильтры, порядок строк, лимиты и готовность."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr_service import recognizer
from ocr_service.config import Settings

#: Ответ движка в его собственном формате: [4 точки, текст, уверенность строкой].
RAW_SINGLE = [
    [[[60.0, 120.0], [400.0, 120.0], [400.0, 180.0], [60.0, 180.0]], "FIRST LINE", "0.93"],
]


def test_probe_size_reads_png_header() -> None:
    """Размер читается из IHDR — до декодирования, иначе бомба съест память."""
    import cv2

    image = np.full((300, 400, 3), 255, np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    assert recognizer.probe_size(buffer.tobytes()) == (400, 300)


def test_probe_size_reads_jpeg_and_webp_headers() -> None:
    import cv2

    image = np.full((240, 320, 3), 255, np.uint8)
    for extension in (".jpg", ".webp"):
        ok, buffer = cv2.imencode(extension, image)
        assert ok, extension
        assert recognizer.probe_size(buffer.tobytes()) == (320, 240), extension


def test_probe_size_returns_none_for_unknown_container() -> None:
    """Незнакомый формат — не повод отказывать: решает декодер."""
    assert recognizer.probe_size(b"GIF89a" + b"\x00" * 64) is None
    assert recognizer.probe_size(b"") is None


def test_decode_image_rejects_non_image() -> None:
    with pytest.raises(recognizer.InvalidImage) as error:
        recognizer.decode_image(b"not an image at all")
    assert error.value.code == "INVALID_IMAGE"


def test_build_lines_converts_four_points_to_bbox() -> None:
    lines = recognizer.build_lines(
        RAW_SINGLE, min_confidence=0.3, max_lines=10, width=900, height=300
    )
    assert len(lines) == 1
    line = lines[0]
    assert line.text == "FIRST LINE"
    assert (line.box.x, line.box.y, line.box.w, line.box.h) == (60, 120, 340, 60)
    assert line.confidence == pytest.approx(0.93)


def test_build_lines_drops_low_confidence_empty_and_degenerate() -> None:
    raw = [
        [[[10, 10], [90, 10], [90, 40], [10, 40]], "KEEP", "0.90"],
        [[[10, 60], [90, 60], [90, 90], [10, 90]], "BELOW THRESHOLD", "0.10"],
        [[[10, 110], [90, 110], [90, 140], [10, 140]], "   ", "0.95"],
        [[[50, 150], [50, 150], [50, 150], [50, 150]], "FLAT", "0.95"],
    ]
    lines = recognizer.build_lines(raw, min_confidence=0.3, max_lines=10, width=200, height=200)
    assert [line.text for line in lines] == ["KEEP"]


def test_build_lines_clips_boxes_to_page_bounds() -> None:
    """Рамка за границей страницы не должна ломать вёрстку отрицательным x/y."""
    raw = [
        [[[-20, -15], [120, -15], [120, 40], [-20, 40]], "EDGE", "0.90"],
        # начинается внутри страницы (150 < 200), но уходит за её границы
        [[[150, 150], [320, 150], [320, 260], [150, 260]], "PARTIAL", "0.90"],
        # целиком за страницей: такой строки на странице нет
        [[[10, 260], [120, 260], [120, 300], [10, 300]], "BELOW", "0.90"],
    ]
    lines = recognizer.build_lines(raw, min_confidence=0.3, max_lines=10, width=200, height=200)
    texts = [line.text for line in lines]
    assert "EDGE" in texts
    edge = next(line for line in lines if line.text == "EDGE")
    assert edge.box.x == 0 and edge.box.y == 0
    # частичное перекрытие обрезается по границе страницы, а не выбрасывается
    partial = next(line for line in lines if line.text == "PARTIAL")
    assert partial.box.x + partial.box.w <= 200
    assert partial.box.y + partial.box.h <= 200
    # строка целиком вне страницы отбрасывается: она не нарисована
    assert "BELOW" not in texts


def test_build_lines_sorts_top_down_then_left_to_right() -> None:
    raw = [
        [[[600, 300], [820, 300], [820, 340], [600, 340]], "RIGHT LOWER", "0.9"],
        [[[60, 100], [300, 100], [300, 140], [60, 140]], "TOP LEFT", "0.9"],
        [[[600, 100], [820, 100], [820, 140], [600, 140]], "TOP RIGHT", "0.9"],
        [[[60, 300], [300, 300], [300, 340], [60, 340]], "LEFT LOWER", "0.9"],
    ]
    lines = recognizer.build_lines(raw, min_confidence=0.3, max_lines=10, width=900, height=400)
    assert [line.text for line in lines] == [
        "TOP LEFT",
        "TOP RIGHT",
        "LEFT LOWER",
        "RIGHT LOWER",
    ]


def test_build_lines_caps_line_count() -> None:
    """Шумная текстура даёт десятки ложных детекций: вёрстка держит лимит."""
    raw = [
        [
            [[10, top], [90, top], [90, top + 15], [10, top + 15]],
            f"LINE {index}",
            "0.9",
        ]
        for index, top in ((index, 10 + index * 20) for index in range(20))
    ]
    lines = recognizer.build_lines(raw, min_confidence=0.3, max_lines=5, width=200, height=600)
    assert len(lines) == 5
    assert [line.text for line in lines] == ["LINE 0", "LINE 1", "LINE 2", "LINE 3", "LINE 4"]


def test_recognize_rejects_decompression_bomb_before_decoding() -> None:
    """Заголовок обещает 20000×20000: отказ до декодирования, иначе OOM."""
    header = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (20_000).to_bytes(4, "big")
        + (20_000).to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
        + b"\x00\x00\x00\x00"
    )
    settings = Settings(max_pixels=1_000_000)
    with pytest.raises(recognizer.ImageTooLarge) as error:
        recognizer.recognize(header, settings, runner=lambda _: [])
    assert error.value.code == "IMAGE_TOO_LARGE"


def test_recognize_uses_injected_runner_and_reports_page_size(page_png: bytes) -> None:
    """Функция распознавания не знает про движок: runner подменяется в тестах."""
    seen: list[tuple[int, int]] = []

    def runner(page: np.ndarray) -> list[list[object]]:
        seen.append((page.shape[1], page.shape[0]))
        return RAW_SINGLE

    result = recognizer.recognize(page_png, Settings(), runner=runner)
    assert seen == [(900, 300)]
    assert result.width == 900 and result.height == 300
    assert [line.text for line in result.lines] == ["FIRST LINE"]


def test_recognize_wraps_engine_crash_into_recognition_failed(page_png: bytes) -> None:
    """Падение движка на валидной картинке — отказ сервиса, а не задание."""
    def runner(_: np.ndarray) -> list[list[object]]:
        raise RuntimeError("onnx сессия развалилась")

    with pytest.raises(recognizer.RecognitionFailed) as error:
        recognizer.recognize(page_png, Settings(), runner=runner)
    assert error.value.code == "OCR_FAILED"


def test_real_engine_reads_plain_text_page(page_png: bytes) -> None:
    """Настоящий прогон RapidOCR: страница с крупным текстом читается."""
    result = recognizer.recognize(page_png, Settings())
    text = " ".join(line.text for line in result.lines)
    assert "HELLO" in text
    assert result.lines[0].box.w > 0 and result.lines[0].box.h > 0
    assert all(0 <= line.confidence <= 1 for line in result.lines)


def test_readiness_is_cached_and_true() -> None:
    ok, reason = recognizer.readiness()
    assert ok is True and reason is None
    assert recognizer.readiness() == (True, None)


def test_readiness_rechecks_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Кэш готовности живёт ограниченное время.

    Иначе pod, у которого движок сломался после первого успеха, остаётся Ready
    навсегда: liveness не трогает движок и рестарта не будет.
    """
    calls: list[int] = []

    def self_check(_settings: Settings | None = None) -> tuple[bool, str | None]:
        calls.append(1)
        return True, None

    monkeypatch.setattr(recognizer, "_self_check", self_check)
    monkeypatch.setattr(recognizer, "_PROBE_RESULT", None)
    clock = [0.0]

    assert recognizer.readiness(Settings(), ttl=10.0, clock=lambda: clock[0]) == (True, None)
    assert recognizer.readiness(Settings(), ttl=10.0, clock=lambda: clock[0]) == (True, None)
    assert len(calls) == 1, "в пределах ttl самопроверка не повторяется"

    clock[0] = 11.0
    assert recognizer.readiness(Settings(), ttl=10.0, clock=lambda: clock[0]) == (True, None)
    assert len(calls) == 2, "после ttl вердикт пересчитывается"


def test_readiness_does_not_latch_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отрицательный вердикт не залипает: следующий запрос пробует снова."""
    answers = iter([(False, "модели не загрузились"), (True, None)])
    monkeypatch.setattr(recognizer, "_self_check", lambda *_: next(answers))
    monkeypatch.setattr(recognizer, "_PROBE_RESULT", None)

    assert recognizer.readiness(Settings(), ttl=60.0)[0] is False
    assert recognizer.readiness(Settings(), ttl=60.0)[0] is True


def test_page_size_reads_formats_without_a_header_probe(page_png: bytes) -> None:
    """BMP/TIFF/PXM заголовок не разбирается — размер берётся ленивым разбором.

    Без этого «бомба» в формате, который понимает OpenCV, но не понимает наш
    разбор заголовка, доходит до декодирования и убивает pod по OOM.
    """
    import cv2

    ok, buffer = cv2.imencode(".bmp", np.full((120, 160, 3), 255, np.uint8))
    assert ok
    assert recognizer.page_size(buffer.tobytes()) == (160, 120)


def test_recognize_rejects_unprobeable_oversized_page_before_decode() -> None:
    """Страница в формате без разбора заголовка отсекается по размеру."""
    import cv2

    ok, buffer = cv2.imencode(".bmp", np.full((1000, 1100, 3), 255, np.uint8))
    assert ok
    with pytest.raises(recognizer.ImageTooLarge):
        recognizer.recognize(buffer.tobytes(), Settings(max_pixels=1_000_000))


def test_decode_maps_the_opencv_pixel_limit_to_413(page_png: bytes,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Предохранитель OpenCV поднимает исключение — оно обязано стать 413.

    Для форматов, которые не умеет разобрать ни наш заголовок, ни Pillow,
    срабатывает лимит самого OpenCV (`cv2.error`). Без перевода в
    `ImageTooLarge` клиент получал бы 500 вместо контрактного 413.
    """
    def failing_decode(*_: object, **__: object) -> np.ndarray:
        raise cv2.error("(-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS")

    monkeypatch.setattr(recognizer.cv2, "imdecode", failing_decode)
    with pytest.raises(recognizer.ImageTooLarge):
        recognizer.decode_image(page_png, Settings(max_pixels=12_345_678))
