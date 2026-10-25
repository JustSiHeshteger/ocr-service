"""HTTP-слой сервиса: FastAPI-приложение по контракту ocr-service.yaml.

Все статусы объявлены контрактом: 200 — строки страницы, 400 `INVALID_IMAGE`,
413 `IMAGE_TOO_LARGE`, 422 `INVALID_REQUEST`, 500 `OCR_FAILED`. Тело любой
ошибки — `OcrError`, включая ошибки разбора запроса: клиент генерируется по
контракту и ждёт одинаковую форму, а не `{"detail": [...]}` от FastAPI.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ocr_service import recognizer
from ocr_service.config import Settings

_LOG = logging.getLogger(__name__)

#: Проба готовности: `(ok, причина)`. Вынесена в параметр, чтобы тесты могли
#: проверить 503 без поломки реального окружения.
ReadinessProbe = Callable[[], tuple[bool, str | None]]

#: Статус ответа по классу ошибки: контракт фиксирует их явно.
_STATUS = {
    recognizer.InvalidImage: 400,
    recognizer.ImageTooLarge: 413,
    recognizer.RecognitionFailed: 500,
}

#: Код ошибки для запроса, не прошедшего собственную валидацию FastAPI.
INVALID_REQUEST = "INVALID_REQUEST"


class BodyLimitMiddleware:
    """Отказ по `Content-Length` до того, как тело будет принято и разобрано.

    Starlette разбирает multipart раньше обработчика и спуливает файл на диск,
    поэтому лимит внутри обработчика срабатывает уже после буферизации: тело
    размером с гигабайт успевает занять диск пода. Заголовок известен заранее,
    поэтому отказ дешевле и точнее.
    """

    def __init__(self, app: Any, *, limit: int) -> None:
        self._app = app
        self._limit = limit

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            declared = _declared_length(scope)
            if declared is not None and declared > self._limit:
                await self._reject(send, declared)
                return
        await self._app(scope, receive, send)

    async def _reject(self, send: Any, declared: int) -> None:
        _LOG.warning("задание отклонено: %s (заявлено %d байт)",
                     recognizer.ImageTooLarge.code, declared)
        body = json.dumps(
            {
                "code": recognizer.ImageTooLarge.code,
                "message": f"страница больше лимита {self._limit} байт",
            },
            ensure_ascii=False,
        ).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _declared_length(scope: dict) -> int | None:
    """`Content-Length` запроса, если клиент его прислал."""
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def create_app(
    settings: Settings | None = None,
    *,
    probe: ReadinessProbe | None = None,
    runner: Any = None,
) -> FastAPI:
    """Собрать приложение; `probe` и `runner` подменяются в тестах."""
    cfg = settings or Settings()
    # проба по умолчанию получает настройки развёртывания: кэш/TTL/таймаут —
    # свойства окружения, а не константы модуля
    readiness_probe = probe or (lambda: recognizer.readiness(cfg))
    app = FastAPI(
        title="OCR Service",
        version="1.0.0",
        description="Распознавание текста на страницах комиксов (арт-слой автоперевода)",
    )
    # настройки доступны и тестам, и __main__: один источник правды
    app.state.settings = cfg
    app.state.engine = recognizer.Engine(cfg, runner=runner)
    # распознавание — CPU-работа: параллелизм ограничен, иначе несколько страниц
    # одновременно выбирают лимит памяти пода и под падает по OOM. Слот берётся
    # до чтения тела: иначе очередь держит по копии страницы на каждый запрос
    app.state.slots = asyncio.Semaphore(cfg.workers)
    # отдельный пул под пробы: зависшая самопроверка не должна занимать потоки
    # общего исполнителя, через который идёт распознавание страниц
    app.state.probe_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-probe")
    # отказ по заявленному размеру тела — до разбора multipart
    app.add_middleware(BodyLimitMiddleware, limit=cfg.max_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Ошибка разбора запроса — в форме контракта, а не в форме FastAPI.

        422 для отсутствующего `image` — тот же код ответа, что у ошибки
        разбора, но свой код ошибки: клиенту важно различать «сервис не понял
        запрос» и «картинка не читается».
        """
        fields = ", ".join(".".join(str(part) for part in item["loc"][1:]) for item in exc.errors())
        message = f"запрос не соответствует контракту: {fields or 'нет полей'}"
        _LOG.warning("задание отклонено: %s (%s)", INVALID_REQUEST, message)
        return JSONResponse({"code": INVALID_REQUEST, "message": message}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Ошибки самого Starlette (битый multipart) — тоже `OcrError`.

        Разбор тела живёт внутри фреймворка и падает раньше обработчика: без
        этого перехвата клиент получал бы `{"detail": ...}`, хотя контракт
        обещает `{"code", "message"}`. Переписывается только ошибка *формы
        запроса* (400 от разбора multipart) — она становится контрактным 422
        `INVALID_REQUEST`, потому что 400 в контракте закреплён за нечитаемой
        картинкой. Остальные статусы (404 неверного пути, 405 метода, 429)
        сохраняются: иначе опечатка в `TRANSLATION_OCR_URL` выглядела бы как
        ошибка разбора запроса.
        """
        status = 422 if exc.status_code == 400 else exc.status_code
        if status >= 500:
            code = recognizer.RecognitionFailed.code
        else:
            code = INVALID_REQUEST
        message = str(exc.detail or "запрос не разобран")
        _LOG.warning("задание отклонено: %s (%s)", code, message)
        return JSONResponse({"code": code, "message": message}, status_code=status)

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Liveness: процесс жив. Зависимостей нет — статус всегда `ok`."""
        return {"status": "ok", "reason": None}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness: 200 только если сервис реально может распознавать."""
        # самопроверка грузит ONNX-модели и берёт общий с заданиями замок движка:
        # отдельный однопоточный пул не даёт зависшей пробе выесть потоки
        # `asyncio.to_thread`, а таймаут — вернуть 503 раньше пробы k8s
        loop = asyncio.get_running_loop()
        try:
            ok, reason = await asyncio.wait_for(
                loop.run_in_executor(app.state.probe_pool, readiness_probe),
                timeout=cfg.readiness_timeout,
            )
        except TimeoutError:
            ok, reason = False, f"самопроверка не уложилась в {cfg.readiness_timeout:g} с"
        body = {"status": "ok" if ok else "degraded", "reason": None if ok else reason}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/recognize")
    async def recognize_page(image: Annotated[UploadFile, File()]) -> JSONResponse:
        """Распознать строки страницы; ответ — `RecognizeResponse`."""
        try:
            # слот берётся до чтения тела: иначе очередь держит по копии страницы
            # на каждый запрос, и лимит `OCR_WORKERS` не ограничивает память
            async with app.state.slots:
                payload = await _read_limited(image, cfg.max_bytes)
                # распознавание — CPU-работа на секунды: не блокируем loop
                result = await asyncio.to_thread(app.state.engine.recognize, payload)
        except recognizer.OcrError as exc:
            status = next(
                (code for kind, code in _STATUS.items() if isinstance(exc, kind)), 500
            )
            _LOG.warning("задание отклонено: %s (%s)", exc.code, exc)
            return JSONResponse({"code": exc.code, "message": str(exc)}, status_code=status)
        except Exception as exc:  # noqa: BLE001 - отказ сервиса обязан быть читаемым
            _LOG.exception("распознавание упало: %s", exc)
            failed = recognizer.RecognitionFailed(f"распознавание упало: {exc}")
            return JSONResponse({"code": failed.code, "message": str(failed)}, status_code=500)

        return JSONResponse({
            "lines": [
                {
                    "text": line.text,
                    "box": {"x": line.box.x, "y": line.box.y, "w": line.box.w, "h": line.box.h},
                    "confidence": line.confidence,
                }
                for line in result.lines
            ],
            "width": result.width,
            "height": result.height,
        })

    return app


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Прочитать тело страницы, не превышая лимит.

    Основной отказ по размеру даёт `BodyLimitMiddleware` (по `Content-Length`),
    но заголовка может не быть — например, при chunked-передаче. Тогда лимит
    проверяется на чтении: без этого гигабайтная страница съела бы память пода
    ещё до проверки размеров картинки.
    """
    payload = bytearray()
    while True:
        chunk = await upload.read(1_048_576)
        if not chunk:
            break
        payload += chunk
        if len(payload) > limit:
            raise recognizer.ImageTooLarge(f"страница больше лимита {limit} байт")
    return bytes(payload)
