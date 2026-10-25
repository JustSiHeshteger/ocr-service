"""Проверки, которые нельзя выразить через HTTP: контракт и манифест.

Оба файла живут в монорепо, но не входят в само зеркало сервиса, поэтому при
отсутствии каталога тесты пропускаются: в опубликованном репозитории сервиса
проверять нечего, в монорепо они обязательны.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parent
CONTRACT = REPO_ROOT / "books-platform-openapi" / "openapi" / "ocr-service.yaml"
MANIFEST = REPO_ROOT / "k8s" / "ocr-service.yaml"

needs_contract = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="контракта нет в этом репозитории"
)
needs_manifest = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="манифеста нет в этом репозитории"
)


@needs_contract
def test_contract_matches_the_service() -> None:
    """Контракт обязан совпадать с кодом, а не с самогенерируемой схемой FastAPI."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    paths = contract["paths"]
    assert set(paths) == {"/recognize", "/health", "/ready"}

    recognize = paths["/recognize"]["post"]
    assert recognize["operationId"] == "recognizePage"
    assert recognize["requestBody"]["required"] is True
    content = recognize["requestBody"]["content"]["multipart/form-data"]["schema"]
    multipart = contract["components"]["schemas"][content["$ref"].rsplit("/", 1)[-1]]
    assert set(multipart["required"]) == {"image"}

    ok = recognize["responses"]["200"]["content"]
    assert "application/json" in ok
    assert ok["application/json"]["schema"]["$ref"].endswith("/RecognizeResponse")

    statuses = {str(code) for code in recognize["responses"]}
    assert {"200", "400", "413", "422", "500"} <= statuses
    for code in ("400", "413", "422", "500"):
        ref = recognize["responses"][code]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("/OcrError"), f"{code} обязан отдавать OcrError"

    error = contract["components"]["schemas"]["OcrError"]["properties"]["code"]
    assert {"INVALID_IMAGE", "INVALID_REQUEST", "IMAGE_TOO_LARGE", "OCR_FAILED"} <= set(
        error["enum"]
    )

    response = contract["components"]["schemas"]["RecognizeResponse"]
    assert set(response["required"]) == {"lines"}
    line = contract["components"]["schemas"]["RecognizedLine"]
    assert set(line["required"]) == {"text", "box"}
    box = contract["components"]["schemas"]["Box"]
    assert set(box["required"]) == {"x", "y", "w", "h"}

    health = contract["components"]["schemas"]["HealthStatus"]
    assert set(health["required"]) == {"status", "reason"}


@needs_contract
def test_contract_declares_readiness_and_liveness_semantics() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    ready = contract["paths"]["/ready"]["get"]
    assert set(ready["responses"]) == {"200", "503"}
    health = contract["paths"]["/health"]["get"]
    assert set(health["responses"]) == {"200"}


@needs_manifest
def test_manifest_runs_pod_without_root_and_drops_capabilities() -> None:
    """k8s-манифест — часть сервиса: он обязан соответствовать образу."""
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert pod["securityContext"]["runAsNonRoot"] is True
    # uid совпадает с `USER 10001:10001` в Dockerfile: иначе pod не стартует
    assert pod["securityContext"]["runAsUser"] == 10001
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["image"] == "bookplatform/ocr-service:dev"
    assert container["ports"][0]["containerPort"] == 8096
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["lifecycle"]["preStop"]["exec"]["command"]


@needs_manifest
def test_manifest_gives_writable_tmp_for_multipart_spooling() -> None:
    """Голая `readOnlyRootFilesystem` ломает `/recognize` на страницах > 1 МиБ.

    Starlette спуливает загружаемый файл на диск, когда он больше 1 МиБ
    (`SpooledTemporaryFile`), а реальная страница комикса весит больше. Без
    записываемого `/tmp` сервис отвечал бы 400 на каждой книге — при этом
    пробы и метрики оставались бы зелёными.
    """
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    mounts = {mount["mountPath"]: mount["name"] for mount in container.get("volumeMounts", [])}
    volumes = {volume["name"]: volume for volume in pod.get("volumes", [])}

    assert "/tmp" in mounts, "readOnlyRootFilesystem требует writable /tmp под мультипарт"
    volume = volumes[mounts["/tmp"]]
    assert "emptyDir" in volume
    size_limit = volume["emptyDir"].get("sizeLimit")
    assert size_limit, "у emptyDir обязан быть sizeLimit: диск пода не бесконечен"
    assert re.fullmatch(r"\d+Mi", size_limit) and int(size_limit.removesuffix("Mi")) >= 64


@needs_manifest
def test_manifest_tmp_is_really_used_by_the_process() -> None:
    """`TMPDIR` обязан указывать в тот же том, что смонтирован."""
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item.get("value") for item in container.get("env", [])}

    assert env.get("TMPDIR") == "/tmp"
    assert env.get("OCR_MAX_PIXELS")
    mounts = {mount["mountPath"] for mount in container.get("volumeMounts", [])}
    assert env["TMPDIR"] in mounts


@needs_manifest
def test_manifest_hardens_the_container_and_bounds_resources() -> None:
    """Границы пода — часть контракта развёртывания, а не пожелание.

    ONNX-инференс на большой странице легко выходит за полтора гигабайта, а
    отсутствие лимитов превращает соседа по ноде в жертву нашего пика.
    """
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    pod = next(doc for doc in docs if doc["kind"] == "Deployment")["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    resources = container["resources"]
    assert resources["requests"]["cpu"] and resources["requests"]["memory"]
    assert resources["limits"]["cpu"] and resources["limits"]["memory"]


@needs_manifest
def test_network_policy_limits_ingress_to_the_worker() -> None:
    """Сервис внутренний: снаружи к нему не ходит никто, кроме воркера."""
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    policies = [doc for doc in docs if doc["kind"] == "NetworkPolicy"]
    assert policies, "без NetworkPolicy сервис открыт всему кластеру"
    policy = policies[0]
    assert policy["spec"]["podSelector"]["matchLabels"]
    assert policy["spec"]["policyTypes"] == ["Ingress"]

    ingress = policy["spec"]["ingress"][0]
    selector = ingress["from"][0]["podSelector"]["matchLabels"]
    assert selector["app"] == "translation-worker"
    port = ingress["ports"][0]["port"]
    assert port == 8096


def test_image_sets_opencv_pixel_limit() -> None:
    """Тот же лимит нужен и в образе: OpenCV читает переменную сам.

    Для форматов без разбора заголовка это единственный предохранитель,
    который срабатывает до выделения памяти.
    """
    dockerfile = (SERVICE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "OPENCV_IO_MAX_IMAGE_PIXELS" in dockerfile
