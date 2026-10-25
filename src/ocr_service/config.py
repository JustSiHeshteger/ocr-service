"""Настройки сервиса: переменные окружения с префиксом `OCR_`."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Лимит пикселей страницы по умолчанию: распознавание идёт за O(пиксели), а
#: детектор RapidOCR растягивает страницу до своего входа — гигантский скан
#: занял бы CPU на минуты.
DEFAULT_MAX_PIXELS = 40_000_000

#: Лимит строк на страницу по умолчанию: защита вёрстки от «страницы из 10k
#: строк» (шумная текстура даёт десятки ложных детекций).
DEFAULT_MAX_LINES = 512


class Settings(BaseSettings):
    """Настройки процесса: всё через `OCR_*`, дефолты — для compose/k8s."""

    model_config = SettingsConfigDict(env_prefix="OCR_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(8096, ge=1, le=65535)
    log_level: str = "INFO"
    #: Лимит площади страницы: больше — 413, а не минуты CPU.
    max_pixels: int = Field(DEFAULT_MAX_PIXELS, ge=1_000_000)
    #: Лимит размера загружаемой страницы: больше — 413, не читая тело целиком.
    max_bytes: int = Field(32_000_000, ge=1_048_576)
    #: Порог уверенности: строки ниже отбрасываются на стороне сервиса, чтобы
    #: воркер не затирал страницу по ложным детекциям.
    min_confidence: float = Field(0.3, ge=0.0, le=1.0)
    #: Лимит строк на страницу: защита вёрстки от шумной текстуры.
    max_lines: int = Field(DEFAULT_MAX_LINES, ge=1)
    #: Сколько страниц распознавать одновременно: каждая держит матрицы страницы.
    workers: int = Field(2, ge=1, le=8)
    #: Сколько секунд вердикт `/ready` считается свежим. Ноль — проверять каждую
    #: пробу (дорого), большое значение — не заметить сломанный движок.
    readiness_ttl: float = Field(30.0, ge=0.0, le=600.0)
    #: Таймаут самой пробы: ответ обязан прийти раньше, чем k8s оборвёт запрос
    #: (`timeoutSeconds` в манифесте), иначе под остаётся в «неизвестно».
    readiness_timeout: float = Field(8.0, gt=0.0, le=120.0)
