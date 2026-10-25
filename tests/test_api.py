"""Тесты HTTP-слоя: статусы, форма ошибок и пробы по контракту ocr-service.yaml."""

from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_service import recognizer
from ocr_service.app import BodyLimitMiddleware, _read_limited, create_app
from ocr_service.config import Settings

#: Ответ движка в его формате: [4 точки, текст, уверенность строкой].
RAW = [[[[60.0, 120.0], [400.0, 120.0], [400.0, 180.0], [60.0, 180.0]], "HELLO WORLD", "0.93"]]


@pytest.fixture
def client() -> TestClient:
    """Клиент с подменённым движком: HTTP-слой проверяется без ONNX-инференса."""
    app = create_app(Settings(), runner=lambda _: RAW)
    return TestClient(app)


def test_health_reports_alive(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "reason": None}


def test_ready_reports_degraded_with_503() -> None:
    """Неготовый движок — 503: воркер деградирует на оверлей фазы 1."""
    app = create_app(Settings(), probe=lambda: (False, "модели не загрузились"),
                     runner=lambda _: RAW)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "reason": "модели не загрузились"}


def test_recognize_returns_lines_in_contract_shape(client: TestClient,
                                                   page_png: bytes) -> None:
    response = client.post("/recognize", files={"image": ("page.png", page_png, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"lines", "width", "height"}
    assert body["width"] == 900 and body["height"] == 300
    line = body["lines"][0]
    assert set(line) == {"text", "box", "confidence"}
    assert line["text"] == "HELLO WORLD"
    assert line["box"] == {"x": 60, "y": 120, "w": 340, "h": 60}
    assert line["confidence"] == pytest.approx(0.93)


def test_empty_page_is_not_an_error() -> None:
    """Страница без текста — валидный ответ с пустым списком, а не 4xx."""
    app = create_app(Settings(), runner=lambda _: [])
    response = TestClient(app).post(
        "/recognize", files={"image": ("page.png", b"\x00", "image/png")}
    )
    # картинка-заглушка не декодируется: нужен реальный PNG без текста
    assert response.status_code == 400
    blank = np.full((120, 160, 3), 255, np.uint8)
    import cv2

    ok, buffer = cv2.imencode(".png", blank)
    assert ok
    response = TestClient(app).post(
        "/recognize", files={"image": ("page.png", buffer.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["lines"] == []


def test_invalid_image_returns_400_with_contract_error(client: TestClient) -> None:
    response = client.post("/recognize", files={"image": ("page.png", b"garbage", "image/png")})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "INVALID_IMAGE"
    assert isinstance(body["message"], str) and body["message"]


def test_missing_image_returns_422_invalid_request(client: TestClient) -> None:
    """Разбор запроса падает — тело всё равно в форме контракта."""
    response = client.post("/recognize", files={"wrong": ("page.png", b"x", "image/png")})
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


def test_oversized_page_returns_413(big_page_png: bytes) -> None:
    app = create_app(Settings(max_pixels=1_000_000), runner=lambda _: RAW)
    response = TestClient(app).post(
        "/recognize", files={"image": ("page.png", big_page_png, "image/png")}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "IMAGE_TOO_LARGE"


def test_page_over_byte_limit_returns_413() -> None:
    """Лимит байт — отдельный предохранитель, а не следствие лимита пикселей.

    Шумная страница 1400×1400 честно проходит по пикселям (1.96 Мпикс против
    лимита 40 Мпикс), но весит больше мегабайта: отказать обязан лимит тела.
    """
    app = create_app(Settings(max_bytes=1_048_576), runner=lambda _: RAW)
    response = TestClient(app).post(
        "/recognize", files={"image": ("page.png", noisy_page_bytes(), "image/png")}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "IMAGE_TOO_LARGE"


@pytest.mark.anyio
async def test_declared_oversized_body_is_rejected_before_parsing() -> None:
    """`Content-Length` больше лимита — отказ до буферизации тела.

    Без этой проверки Starlette сначала принимает и спуливает файл целиком, и
    гигабайтная «страница» успевает занять диск пода до нашего 413.
    """
    reached: list[str] = []

    async def downstream(scope: dict, receive: object, send: object) -> None:
        reached.append("обработчик")

    middleware = BodyLimitMiddleware(downstream, limit=1024)
    messages: list[dict] = []

    async def receive() -> dict:
        raise AssertionError("тело не должно читаться при отказе по заголовку")

    async def send(message: dict) -> None:
        messages.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/recognize",
            "headers": [(b"content-length", b"10485760")],
        },
        receive,
        send,
    )

    assert reached == []
    assert messages[0]["status"] == 413
    assert b"IMAGE_TOO_LARGE" in messages[1]["body"]


@pytest.mark.anyio
async def test_body_within_limit_reaches_the_handler() -> None:
    """Лимит не должен ломать обычный запрос."""
    reached: list[str] = []

    async def downstream(scope: dict, receive: object, send: object) -> None:
        reached.append("обработчик")

    middleware = BodyLimitMiddleware(downstream, limit=2048)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        return None

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/recognize",
            "headers": [(b"content-length", b"1024")],
        },
        receive,
        send,
    )
    assert reached == ["обработчик"]


def test_malformed_multipart_returns_contract_error() -> None:
    """Битый multipart — тоже `OcrError`, а не `{"detail": ...}` от Starlette."""
    app = create_app(Settings(), runner=lambda _: RAW)
    response = TestClient(app).post(
        "/recognize",
        content=b"--boundary\r\nContent-Disposition: form-data\r\n\r\n--boundary--",
        headers={"content-type": "multipart/form-data; boundary=boundary"},
    )
    # 400 в контракте закреплён за нечитаемой картинкой: ошибка формы — это 422
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "INVALID_REQUEST"
    assert isinstance(body["message"], str) and body["message"]


def test_unknown_path_keeps_its_status() -> None:
    """404 не переписывается в 422: иначе опечатка в URL выглядит как форма запроса."""
    app = create_app(Settings(), runner=lambda _: RAW)
    response = TestClient(app).get("/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.anyio
async def test_read_limited_stops_at_the_byte_limit() -> None:
    """Chunked-передача идёт без `Content-Length`: лимит обязан сработать на чтении."""

    class FakeUpload:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def read(self, _size: int = -1) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

    with pytest.raises(recognizer.ImageTooLarge):
        await _read_limited(FakeUpload([b"a" * 700, b"b" * 700]), 1000)

    payload = await _read_limited(FakeUpload([b"a" * 400, b"b" * 400]), 1000)
    assert payload == b"a" * 400 + b"b" * 400


@pytest.mark.anyio
async def test_ready_reports_degraded_when_the_probe_hangs() -> None:
    """Зависшая самопроверка не должна держать пробу до таймаута k8s."""

    def hanging_probe() -> tuple[bool, str | None]:
        time.sleep(5.0)
        return True, None

    app = create_app(Settings(readiness_timeout=0.2), probe=hanging_probe, runner=lambda _: RAW)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ocr") as client:
        started = time.monotonic()
        response = await client.get("/ready")
        elapsed = time.monotonic() - started
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert elapsed < 1.0, f"проба держалась {elapsed:.2f} с"


@pytest.mark.anyio
async def test_ready_does_not_stall_the_event_loop() -> None:
    """Самопроверка грузит ONNX-модели: она не имеет права держать loop.

    Иначе на старте пода `/health` (liveness с `timeoutSeconds: 3`) и текущие
    `/recognize` встают в очередь за загрузкой моделей.
    """

    def slow_probe() -> tuple[bool, str | None]:
        time.sleep(0.4)
        return True, None

    app = create_app(Settings(), probe=slow_probe, runner=lambda _: RAW)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ocr") as client:
        started = time.monotonic()
        ready_call = asyncio.create_task(client.get("/ready"))
        await asyncio.sleep(0.05)
        health = await client.get("/health")
        elapsed = time.monotonic() - started
        # если самопроверка держит loop, /health не успевает обогнать /ready
        assert not ready_call.done(), "/health ждал самопроверку в том же потоке"
        assert health.status_code == 200
        assert elapsed < 0.25, f"/health ждал самопроверку {elapsed:.2f} с"
        assert (await ready_call).status_code == 200


def test_engine_crash_returns_500_ocr_failed(page_png: bytes) -> None:
    """Отказ сервиса ретраибелен и обязан быть в форме контракта."""
    def runner(_: np.ndarray) -> list[list[object]]:
        raise RuntimeError("onnx упал")

    app = create_app(Settings(), runner=runner)
    response = TestClient(app).post(
        "/recognize", files={"image": ("page.png", page_png, "image/png")}
    )
    assert response.status_code == 500
    assert response.json()["code"] == "OCR_FAILED"


def noisy_page_bytes() -> bytes:
    """Шумная страница: PNG не сжимает её, поэтому файл заметно больше 1 МиБ."""
    import cv2

    noise = np.random.default_rng(7).integers(0, 256, (1400, 1400, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", noise)
    assert ok
    payload = buffer.tobytes()
    assert len(payload) > 1_048_576
    return payload
