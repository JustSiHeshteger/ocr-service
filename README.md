# ocr-service

Сервис распознавания текста на страницах комиксов — поставщик строк для
автоперевода: воркер скачивает страницу из S3, отправляет её сюда, получает
строки с рамками, переводит их и рисует поверх страницы.

Контракт: `books-platform-openapi/openapi/ocr-service.yaml`.

## Что делает

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/recognize` | распознать строки (multipart: `image`) |
| GET | `/health` | liveness: процесс жив (зависимостей нет) |
| GET | `/ready` | readiness: прогрев движка + мини-прогон на синтетической странице |

Движок — RapidOCR (ONNX Runtime, CPU, офлайн): модели лежат внутри образа,
GPU и сеть не нужны. На стенде страница комикса 2984×4585 распознаётся
примерно за 5 с на одно ядро (7-30 строк текста); маленькая синтетическая
страница из самопроверки — доли секунды.

Гарантии контракта:

- **строки в порядке чтения** — сверху вниз, затем слева направо: вёрстка
  перевода опирается на этот порядок;
- **рамка — прямоугольник** в пикселях исходной страницы, обрезанный по её
  границам: четырёхточечный контур детектора превращается в bbox, поэтому
  воркеру не нужно повторять геометрию;
- **пустой список строк допустим** — страница без текста не ошибка;
- **порог уверенности** (`OCR_MIN_CONFIDENCE`) режется на стороне сервиса:
  ложные детекции на текстуре не должны затирать страницу;
- **ошибки машинно-читаемы**: `INVALID_IMAGE` (400), `IMAGE_TOO_LARGE` (413),
  `INVALID_REQUEST` (422), `OCR_FAILED` (500). Тело любой ошибки — `OcrError`,
  включая ошибки разбора запроса: клиент генерируется по контракту и ждёт одну
  форму, а не `{"detail": [...]}`.

Лимиты — часть контракта с платформой:

- **площадь страницы** проверяется по заголовку контейнера (PNG IHDR, JPEG
  SOFn, WebP VP8X) до декодирования: «PNG-бомба» иначе съедает память раньше,
  чем сработает отказ;
- **размер тела** отсекается по `Content-Length` до разбора multipart (запрос
  больше `OCR_MAX_BYTES` получает 413, не занимая диск пода), и повторно
  проверяется на чтении — для chunked-передач без заголовка;
- **число строк** ограничено `OCR_MAX_LINES`: шумная текстура даёт десятки
  ложных детекций, и вёрстка не должна на них разваливаться;
- **параллелизм** ограничен `OCR_WORKERS`: каждая страница держит матрицу
  страницы и вход детектора, поэтому лимит памяти пода и лимит параллелизма
  задаются вместе.

## Настройки

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `OCR_HOST` / `OCR_PORT` | `0.0.0.0` / `8096` | адрес и порт |
| `OCR_MIN_CONFIDENCE` | `0.3` | порог уверенности строки (0–1) |
| `OCR_MAX_PIXELS` | `40000000` | лимит площади страницы (иначе 413) |
| `OCR_MAX_BYTES` | `32000000` | лимит размера тела запроса (байты) |
| `OCR_MAX_LINES` | `512` | лимит строк на страницу |
| `OCR_WORKERS` | `2` | сколько страниц держать в работе одновременно (инференс сериализован: RapidOCR не потокобезопасен) |
| `OCR_READINESS_TTL` | `30` | секунд, сколько успешный вердикт `/ready` считается свежим |
| `OCR_READINESS_TIMEOUT` | `8` | таймаут самой пробы `/ready` (меньше `timeoutSeconds` пробы в манифесте) |
| `OCR_LOG_LEVEL` | `INFO` | уровень логов |
| `OPENCV_IO_MAX_IMAGE_PIXELS` | `40000000` | тот же предел для OpenCV: страницы в BMP/TIFF/PXM упираются в него, а не в OOM. Читается процессом один раз, поэтому задаётся окружением (Dockerfile, манифест), а не из кода |

Порог и лимиты — свойство развёртывания, а не задания: они не приходят в
запросе, иначе страницы одной книги распознавались бы по-разному.

## Запуск

```bash
# локально (нужен python 3.12+)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
# rapidocr-onnxruntime ставится без зависимостей: его opencv-python тянет GUI,
# а сервису нужен headless-вариант (см. Dockerfile)
.venv\Scripts\python.exe -m pip install --no-deps rapidocr-onnxruntime==1.2.3
$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m ocr_service

# образ
docker build -t bookplatform/ocr-service:dev .

# в контейнерах `readOnlyRootFilesystem` обязателен writable `/tmp`:
# Starlette спуливает части мультипарта больше 1 МиБ во временный файл
docker run --rm -p 8096:8096 --read-only --tmpfs /tmp bookplatform/ocr-service:dev

# в составе стенда
cd docker; docker compose --env-file .env up -d ocr-service translation-worker
```

Пример запроса:

```bash
curl -sS http://localhost:8096/recognize \
  -F 'image=@page.png;type=application/octet-stream'
# {"lines":[{"text":"I AM THE NIGHT","box":{"x":60,"y":120,"w":340,"h":60},
#            "confidence":0.93}],"width":1600,"height":2460}
```

## Тесты

```bash
.venv\Scripts\python.exe -m pytest    # 43 теста
.venv\Scripts\python.exe -m ruff check .
```

`tests/test_deployment.py` сверяет сервис с файлом контракта (пути, коды,
`required`, имена схем) и с k8s-манифестом (runAsNonRoot, uid, read-only rootfs,
лимиты ресурсов, writable `/tmp` под спулинг мультипарта, пробы, `preStop`,
NetworkPolicy «только воркер»). В опубликованном зеркале сервиса
этих файлов нет — тесты там пропускаются, в монорепо обязательны.

Проверяется контракт, а не «вызвался ли ONNX»: контур детектора превращается в
bbox, строки ниже порога и пустой текст отбрасываются, рамки обрезаются по
границам страницы, порядок строк — сверху вниз, лимит строк держится, «PNG-
бомба» отклоняется до декодирования, лимит площади даёт 413, битая картинка —
400, пропущенное поле и битый multipart — 422 `INVALID_REQUEST` (одна форма тела
для всех ошибок), лимит байт — 413, сбой движка — 500 `OCR_FAILED`, `/ready`
делает мини-прогон, не залипает на отказе и пересчитывает вердикт по TTL. Один тест прогоняет настоящий
RapidOCR на синтетической странице: он ловит «модели в образе битые».

## Место в конвейере

```
ComicTranslationRequested
      ↓
translation-worker: скачивание страницы, OCR
      ↓ POST /recognize (page)
ocr-service  →  строки: text + box + confidence
      ↓
translation-worker: TRANSLATE (LLM/DeepL) → LAYOUT → RENDER
      ↓ POST /erase (page + рамки) в inpaint-service
      ↓
translation-worker: вёрстка перевода (Pillow) → web.webp в MinIO
      ↓
ComicTranslationCompleted.pages[].imageKey → book-service → imageUrl у TextLayer
```

Включается на стороне воркера: `TRANSLATION_OCR_PROVIDER=http` и
`TRANSLATION_OCR_URL=http://ocr-service:8096/recognize`. Если сервис недоступен,
воркер получает `OCR_FAILED` (или таймаут) и при
`TRANSLATION_STRICT_RENDER=false` деградирует на оверлей фазы 1 с warning
`page N: art layer skipped` — книга дочитывается.
