# syntax=docker/dockerfile:1
# OCR-сервис арт-слоя автоперевода комиксов.
# Контекст сборки — каталог ocr-service (контракт сервису не нужен: он получает
# страницу и отдаёт строки, а не события).
FROM python:3.12-slim AS runtime

# OpenCV-headless не тянет GUI, но его бинарник зависит от сборки GLib и OpenMP:
# без них `import cv2` падает на старте, а не в запросе. curl нужен HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
# rapidocr-onnxruntime объявляет зависимость на opencv-python (GUI-сборка),
# поэтому ставится без зависимостей: остальные приезжают из requirements.txt
# headless-вариантом, иначе `import cv2` требует libGL.
RUN pip install --no-deps rapidocr-onnxruntime==1.2.3 \
    && pip install -r requirements.txt

COPY src/ ./src/
# Пакет не ставим: src-layout подключается через PYTHONPATH. Так образ не
# зависит от сборки wheel и остаётся работоспособным при readOnlyRootFilesystem.
ENV PYTHONPATH=/app/src

# Сборочный прогон готовности: ловит битые ONNX-модели и недопоставленные
# зависимости (например, Pillow, без которого RapidOCR падает на первом
# задании) на сборке, а не на первом запросе стенда.
RUN python -c "import sys; from ocr_service import recognizer; ok, reason = recognizer.readiness(); print('readiness:', ok, reason); sys.exit(0 if ok else 1)"

# Непривилегированный пользователь: распознавание не требует прав в системе, а
# k8s-манифест требует runAsNonRoot (uid фиксирован — см. k8s/ocr-service.yaml).
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin ocr
USER 10001:10001

# Лимит страницы: 40000000 пикселей ≈ 7000×5700 — с запасом на двухстраничные
# развороты сканов; всё, что больше, сервис отклоняет 413 (контракт).
# OPENCV_IO_MAX_IMAGE_PIXELS — тот же предел для самого OpenCV: страницы в
# форматах без разбора заголовка (BMP/TIFF/PXM) упираются в него, а не в OOM.
ENV OCR_HOST=0.0.0.0 \
    OCR_PORT=8096 \
    OCR_MIN_CONFIDENCE=0.3 \
    OCR_MAX_PIXELS=40000000 \
    OCR_MAX_BYTES=32000000 \
    OCR_MAX_LINES=512 \
    OCR_READINESS_TTL=30 \
    OCR_WORKERS=2 \
    OPENCV_IO_MAX_IMAGE_PIXELS=40000000

EXPOSE 8096

# HEALTHCHECK смотрит на готовность, как compose и k8s: контейнер, который жив,
# но не может распознавать, не должен считаться здоровым
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8096/ready || exit 1

CMD ["python", "-m", "ocr_service"]
