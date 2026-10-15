"""Small, same-origin FoodVision HTTP server.

The standard-library server is deliberately useful without the optional model
dependencies.  A missing or invalid model is reported as unavailable; it is
never represented by a made-up prediction.
"""

from __future__ import annotations

import importlib
import json
import math
import mimetypes
import os
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError):
        return default
    return parsed


DEFAULT_PORT = _env_int("FOODVISION_PORT", 8000)
MAX_UPLOAD_BYTES = max(1, _env_int("FOODVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_DIMENSION = max(1, _env_int("FOODVISION_MAX_IMAGE_DIMENSION", 4096))
MAX_IMAGE_PIXELS = max(1, _env_int("FOODVISION_MAX_IMAGE_PIXELS", 25_000_000))
MAX_TOP_K = 5
DEFAULT_TOP_K = 3
MAX_MULTIPART_OVERHEAD = 1024 * 1024

SUPPORTED_PREPROCESSING = {
    "inception_v3",
    "mobilenet_v2",
    "rescale_0_1",
    "none",
}
SUPPORTED_OUTPUT_TYPES = {"probabilities", "logits"}

# These are exact names used by the earlier project configuration.  Values are
# not matched by substring, which prevents a typo from silently changing the
# model's input contract.
PREPROCESSING_ALIASES = {
    "inception_v3": "inception_v3",
    "mobilenet_v2": "mobilenet_v2",
    "rescale_0_1": "rescale_0_1",
    "none": "none",
    "inception_v3_preprocess_input": "inception_v3",
    "mobilenet_v2_preprocess_input": "mobilenet_v2",
    "mobile_net_preprocess_input": "mobilenet_v2",
}

STATIC_IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
STATIC_EXACT_FILES = {"index.html", "app.js", "styles.css"}
STATIC_DOC_FILES = {"README.md", "MODEL_HANDOFF.md", "PRESENTATION_GUIDE.md"}


def load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


CONFIG = load_json(DATA_DIR / "config.json", {})
if not isinstance(CONFIG, dict):
    CONFIG = {}
METRICS = load_json(DATA_DIR / "model-metrics.json", {})
if not isinstance(METRICS, dict):
    METRICS = {}
FOODS = load_json(DATA_DIR / "foods.json", {})
if not isinstance(FOODS, dict):
    FOODS = {}
FOODS = FOODS.get("foods", {}) if isinstance(FOODS.get("foods", {}), dict) else {}


class ConfigurationError(ValueError):
    """A model handoff setting is not safe to use."""


class MultipartError(ValueError):
    """The request body is not a valid multipart form."""


class InvalidImageError(ValueError):
    """The uploaded bytes are not a bounded, decodable image."""


class InvalidModelOutput(ValueError):
    """The model returned a shape or value outside its declared contract."""


def _validate_class_names(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("Class names are missing or invalid")
    names: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip() or any(
            character in item for character in "\r\n\x00"
        ):
            raise ConfigurationError("Class names are missing or invalid")
        if item in seen:
            raise ConfigurationError("Class names must be unique and ordered")
        seen.add(item)
        names.append(item)
    return names


def _read_json_file(path: Path) -> tuple[Any, bool]:
    """Return (value, exists); malformed JSON is represented by value None."""
    if not path.is_file():
        return None, False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), True
    except (OSError, json.JSONDecodeError):
        return None, True


def _load_project_labels() -> tuple[list[str], str, str]:
    model_labels_path = Path(
        os.environ.get("FOODVISION_CLASS_NAMES_PATH", str(MODEL_DIR / "class_names.json"))
    )
    if model_labels_path.is_file():
        raw, _ = _read_json_file(model_labels_path)
        try:
            return _validate_class_names(raw), "model_handoff", ""
        except ConfigurationError:
            return [], "model_handoff", "Model class names are missing or invalid"

    raw, exists = _read_json_file(DATA_DIR / "class_names.json")
    if not exists:
        return [], "project_placeholder", "Project class names are missing"
    try:
        return _validate_class_names(raw), "project_placeholder", ""
    except ConfigurationError:
        return [], "project_placeholder", "Project class names are missing or invalid"


CLASS_NAMES, CLASS_SOURCE, CLASS_NAMES_ERROR = _load_project_labels()


def _canonical_preprocessing(value: Any) -> str:
    if not isinstance(value, str) or value not in PREPROCESSING_ALIASES:
        raise ConfigurationError("Unsupported preprocessing configuration")
    canonical = PREPROCESSING_ALIASES[value]
    if canonical not in SUPPORTED_PREPROCESSING:
        raise ConfigurationError("Unsupported preprocessing configuration")
    return canonical


def _validated_input_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ConfigurationError("Invalid model input size")
    return value


def _validated_text(value: Any, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value or any(character in value for character in "\r\n\x00"):
        raise ConfigurationError("Invalid model metadata")
    return value


def _environment_value(*names: str) -> str | None:
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return None


def _load_model_settings() -> dict[str, Any]:
    config_path_text = os.environ.get("FOODVISION_MODEL_CONFIG_PATH")
    config_path = Path(config_path_text) if config_path_text else MODEL_DIR / "model_config.json"
    raw_config, config_exists = _read_json_file(config_path)
    if config_exists and not isinstance(raw_config, dict):
        raise ConfigurationError("Model configuration is invalid")
    model_config = raw_config if isinstance(raw_config, dict) else {}

    default_input_size = CONFIG.get("inputSize", 299)
    default_preprocessing = CONFIG.get("preprocessing", "inception_v3")
    name = _environment_value("FOODVISION_MODEL_NAME")
    version = _environment_value("FOODVISION_MODEL_VERSION")
    input_size = _environment_value("FOODVISION_INPUT_SIZE", "FOODVISION_MODEL_INPUT_SIZE")
    preprocessing = _environment_value(
        "FOODVISION_PREPROCESSING", "FOODVISION_MODEL_PREPROCESSING"
    )
    output_type = _environment_value("FOODVISION_OUTPUT_TYPE")

    if input_size is not None:
        try:
            input_size_value: Any = int(input_size, 10)
        except (TypeError, ValueError):
            raise ConfigurationError("Invalid model input size") from None
    else:
        input_size_value = model_config.get("inputSize", default_input_size)

    preprocessing_value = (
        preprocessing
        if preprocessing is not None
        else model_config.get("preprocessing", default_preprocessing)
    )
    output_type_value = output_type if output_type is not None else model_config.get(
        "outputType", "probabilities"
    )
    if not isinstance(output_type_value, str) or output_type_value not in SUPPORTED_OUTPUT_TYPES:
        raise ConfigurationError("Unsupported model output type")

    return {
        "name": _validated_text(
            name if name is not None else model_config.get("name"), "FoodVision team model"
        ),
        "version": _validated_text(
            version if version is not None else model_config.get("version"), "unreleased"
        ),
        "inputSize": _validated_input_size(input_size_value),
        "preprocessing": _canonical_preprocessing(preprocessing_value),
        "outputType": output_type_value,
    }


def _shape_as_list(shape: Any) -> list[Any] | None:
    try:
        if hasattr(shape, "as_list"):
            shape = shape.as_list()
        if isinstance(shape, (list, tuple)):
            return list(shape)
    except Exception:
        return None
    return None


def _model_output_dimension(model: Any) -> int | None:
    shape = getattr(model, "output_shape", None)
    shapes = shape if isinstance(shape, list) and shape and isinstance(shape[0], (list, tuple)) else None
    if shapes is not None:
        if len(shapes) != 1:
            return None
        shape = shapes[0]
    dimensions = _shape_as_list(shape)
    if dimensions is None or len(dimensions) != 2 or dimensions[0] not in (None, 1):
        return None
    output_dimension = dimensions[-1]
    if isinstance(output_dimension, bool) or not isinstance(output_dimension, int):
        return None
    return output_dimension if output_dimension > 0 else None


def _output_vector(raw_output: Any) -> list[Any]:
    """Extract only a single classification vector; reject ambiguous outputs."""
    if hasattr(raw_output, "numpy") and callable(raw_output.numpy):
        raw_output = raw_output.numpy()

    shape = getattr(raw_output, "shape", None)
    dimensions = _shape_as_list(shape) if shape is not None else None
    if dimensions is not None:
        if len(dimensions) == 1:
            vector = raw_output
        elif len(dimensions) == 2 and dimensions[0] == 1:
            vector = raw_output[0]
        else:
            raise InvalidModelOutput("Model output shape is invalid")
        try:
            values = list(vector)
        except Exception:
            raise InvalidModelOutput("Model output shape is invalid") from None
    elif isinstance(raw_output, (list, tuple)):
        if len(raw_output) == 1 and isinstance(raw_output[0], (list, tuple)):
            values = list(raw_output[0])
        elif all(not isinstance(item, (list, tuple)) for item in raw_output):
            values = list(raw_output)
        else:
            raise InvalidModelOutput("Model output shape is invalid")
    else:
        raise InvalidModelOutput("Model output shape is invalid")

    if any(isinstance(item, (list, tuple)) for item in values):
        raise InvalidModelOutput("Model output shape is invalid")
    return values


def _validated_probabilities(raw_output: Any, class_count: int, output_type: str) -> list[float]:
    """Validate declared probabilities or explicitly convert declared logits."""
    values = _output_vector(raw_output)
    if len(values) != class_count:
        raise InvalidModelOutput("Model output dimension does not match class names")
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise InvalidModelOutput("Model output contains invalid values") from None
    if not all(math.isfinite(value) for value in numeric):
        raise InvalidModelOutput("Model output contains invalid values")

    if output_type == "probabilities":
        if any(value < 0.0 or value > 1.0 for value in numeric):
            raise InvalidModelOutput("Model probabilities are outside the valid range")
        if not math.isclose(sum(numeric), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise InvalidModelOutput("Model probabilities do not sum to one")
        return numeric
    if output_type != "logits":
        raise InvalidModelOutput("Unsupported model output type")

    maximum = max(numeric)
    exponentials: list[float] = []
    try:
        for value in numeric:
            difference = value - maximum
            exponentials.append(0.0 if difference == float("-inf") else math.exp(difference))
        total = sum(exponentials)
    except (OverflowError, ValueError):
        raise InvalidModelOutput("Model logits are invalid") from None
    if not math.isfinite(total) or total <= 0:
        raise InvalidModelOutput("Model logits are invalid")
    probabilities = [value / total for value in exponentials]
    if not all(math.isfinite(value) for value in probabilities):
        raise InvalidModelOutput("Model logits are invalid")
    return probabilities


class ModelAdapter:
    """Lazy, validated adapter around an optional Keras model."""

    def __init__(self) -> None:
        self.model: Any = None
        self.numpy: Any = None
        self.image_class: Any = None
        self.image_ops: Any = None
        self.preprocess: Callable[[Any], Any] | None = None
        self.load_error = CLASS_NAMES_ERROR
        self.class_names = list(CLASS_NAMES)
        self.class_source = CLASS_SOURCE
        self.model_path = Path(
            os.environ.get("FOODVISION_MODEL_PATH", str(MODEL_DIR / "foodvision_model.keras"))
        )
        self.model_name = "FoodVision team model"
        self.model_version = "unreleased"
        self.input_size = 299
        self.preprocessing = "inception_v3"
        self.output_type = "probabilities"
        self._predict_lock = threading.Lock()
        self._try_load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _try_load(self) -> None:
        try:
            settings = _load_model_settings()
            self.model_name = settings["name"]
            self.model_version = settings["version"]
            self.input_size = settings["inputSize"]
            self.preprocessing = settings["preprocessing"]
            self.output_type = settings["outputType"]
        except ConfigurationError as exc:
            self.load_error = str(exc)
            return

        if self.load_error:
            return
        if not self.model_path.is_file():
            self.load_error = "Model file not found"
            return

        # Optional ML packages are imported only after a model file is present.
        try:
            import numpy as np  # type: ignore
            from PIL import Image, ImageOps  # type: ignore
            import tensorflow as tf  # type: ignore
        except ImportError:
            self.load_error = "Optional model dependencies are unavailable"
            return
        except Exception:
            self.load_error = "Optional model dependencies are unavailable"
            return

        try:
            loaded_model = tf.keras.models.load_model(
                str(self.model_path), compile=False, safe_mode=True
            )
            output_dimension = _model_output_dimension(loaded_model)
            if output_dimension != len(self.class_names):
                self.load_error = "Model output dimension does not match class names"
                return
            self.numpy = np
            self.image_class = Image
            self.image_ops = ImageOps
            self.model = loaded_model
            self.preprocess = self._preprocess_array
        except ConfigurationError as exc:
            self.load_error = str(exc)
        except Exception:
            self.model = None
            self.load_error = "Model could not be loaded or validated"

    def _preprocess_array(self, array: Any) -> Any:
        if self.preprocessing == "none":
            return array
        if self.preprocessing in {"inception_v3", "mobilenet_v2"}:
            return (array / 127.5) - 1.0
        if self.preprocessing == "rescale_0_1":
            return array / 255.0
        raise ConfigurationError("Unsupported preprocessing configuration")

    def _decode_image(self, image_bytes: bytes) -> Any:
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise InvalidImageError("Image exceeds upload limit")
        try:
            with self.image_class.open(BytesIO(image_bytes)) as probe:
                width, height = probe.size
                if (
                    not isinstance(width, int)
                    or not isinstance(height, int)
                    or width < 1
                    or height < 1
                    or width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise InvalidImageError("Image dimensions exceed the limit")
                probe.verify()

            with self.image_class.open(BytesIO(image_bytes)) as source:
                image = self.image_ops.exif_transpose(source)
                rgb_image = image.convert("RGB")
                resized = rgb_image.resize((self.input_size, self.input_size))
                return self.numpy.asarray(resized).astype("float32")
        except InvalidImageError:
            raise
        except Exception:
            raise InvalidImageError("Uploaded bytes are not a valid image") from None

    def predict(self, image_bytes: bytes, top_k: int) -> tuple[list[dict[str, Any]], int]:
        if not self.available or self.numpy is None or self.image_class is None or self.preprocess is None:
            raise RuntimeError("Model is unavailable")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError("topK is outside the supported range")

        started = time.perf_counter()
        array = self._decode_image(image_bytes)
        array = self.preprocess(array)
        batch = self.numpy.expand_dims(array, axis=0)
        with self._predict_lock:
            raw_output = self.model.predict(batch, verbose=0)
        probabilities = _validated_probabilities(raw_output, len(self.class_names), self.output_type)
        ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)[
            :top_k
        ]
        predictions: list[dict[str, Any]] = []
        for index in ranked:
            class_id = self.class_names[index]
            food = FOODS.get(class_id, {})
            fallback_label = class_id.replace("_", " ").title()
            predictions.append(
                {
                    "classId": class_id,
                    "label": food.get("displayName", fallback_label),
                    "confidence": round(probabilities[index], 6),
                }
            )
        if not predictions:
            raise InvalidModelOutput("Model returned no classes")
        latency_ms = round((time.perf_counter() - started) * 1000)
        return predictions, latency_ms

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "inputSize": self.input_size,
            "preprocessing": self.preprocessing,
            "available": self.available,
        }


MODEL = ModelAdapter()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def json_response(handler: SimpleHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    try:
        encoded = _json_bytes(payload)
    except (TypeError, ValueError):
        encoded = _json_bytes(
            {"status": "error", "error": "INTERNAL_ERROR", "message": "The server could not encode its response."}
        )
        status = HTTPStatus.INTERNAL_SERVER_ERROR
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(encoded)


def error_response(handler: SimpleHTTPRequestHandler, code: str, message: str, status: int) -> None:
    json_response(handler, {"status": "error", "error": code, "message": message}, status)


def parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Parse a multipart body with the stdlib MIME parser without trimming data."""
    envelope = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode(
        "ascii", "strict"
    )
    try:
        header_message = BytesParser(policy=policy.default).parsebytes(envelope)
        if header_message.get_content_type() != "multipart/form-data":
            raise MultipartError("Multipart content type is required")
        boundary_text = header_message.get_param("boundary", header="content-type")
        if not isinstance(boundary_text, str) or not boundary_text or len(boundary_text) > 200:
            raise MultipartError("Multipart boundary is missing")
        boundary = boundary_text.encode("utf-8", "strict")
        if any(byte < 32 or byte == 127 for byte in boundary):
            raise MultipartError("Multipart boundary is invalid")
        if not body.startswith(b"--" + boundary) or not (
            b"--" + boundary + b"--" in body
        ):
            raise MultipartError("Multipart body is incomplete")
        message = BytesParser(policy=policy.default).parsebytes(envelope + body)
    except (UnicodeError, ValueError, IndexError):
        raise MultipartError("Multipart body is invalid") from None

    if not message.is_multipart():
        raise MultipartError("Multipart body is invalid")
    fields: dict[str, bytes] = {}
    for part in message.iter_parts():
        if part.is_multipart():
            raise MultipartError("Nested multipart fields are not supported")
        field_name = part.get_param("name", header="content-disposition")
        if not isinstance(field_name, str) or not field_name or field_name in fields:
            raise MultipartError("Multipart field is invalid")
        value = part.get_payload(decode=True)
        if not isinstance(value, bytes):
            raise MultipartError("Multipart field is invalid")
        fields[field_name] = value
    return fields


def parse_form_field(body: bytes, content_type: str, field_name: str) -> bytes | None:
    """Compatibility helper for callers that need one multipart field."""
    try:
        return parse_multipart(body, content_type).get(field_name)
    except MultipartError:
        return None


def _read_content_length(handler: SimpleHTTPRequestHandler) -> int | None:
    if handler.headers.get("Transfer-Encoding"):
        error_response(
            handler,
            "INVALID_CONTENT_LENGTH",
            "Chunked uploads are not supported.",
            HTTPStatus.BAD_REQUEST,
        )
        return None
    raw = handler.headers.get("Content-Length")
    if raw is None or not raw.strip():
        error_response(
            handler,
            "LENGTH_REQUIRED",
            "Content-Length is required for uploads.",
            HTTPStatus.LENGTH_REQUIRED,
        )
        return None
    try:
        length = int(raw, 10)
    except (TypeError, ValueError):
        error_response(handler, "INVALID_CONTENT_LENGTH", "Content-Length is invalid.", HTTPStatus.BAD_REQUEST)
        return None
    if length < 1:
        error_response(handler, "INVALID_UPLOAD", "The upload body is empty.", HTTPStatus.BAD_REQUEST)
        return None
    max_request_bytes = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD
    if length > max_request_bytes:
        error_response(
            handler,
            "UPLOAD_TOO_LARGE",
            "The upload exceeds the request size limit.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
        return None
    return length


def _load_build_insights() -> Callable[..., Any] | None:
    """Import knowledge only when an insights request actually needs it."""
    for module_name in ("backend.knowledge", "knowledge"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        builder = getattr(module, "build_insights", None)
        if callable(builder):
            return builder
    return None


class FoodVisionHandler(SimpleHTTPRequestHandler):
    server_version = "FoodVisionHTTP/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _static_file(self, request_path: str) -> Path | None:
        try:
            decoded = unquote(request_path)
        except Exception:
            return None
        if not decoded.startswith("/") or "\x00" in decoded or "\\" in decoded:
            return None
        relative = decoded.lstrip("/")
        if relative == "":
            relative = "index.html"
        parts = relative.split("/")
        if any(not part or part in {".", ".."} or part.startswith(".") for part in parts):
            return None

        allowed = False
        if len(parts) == 1 and parts[0] in STATIC_EXACT_FILES:
            allowed = True
        elif len(parts) == 1 and parts[0] in STATIC_DOC_FILES and parts[0].endswith(".md"):
            allowed = True
        elif len(parts) == 1 and Path(parts[0]).suffix.lower() in STATIC_IMAGE_SUFFIXES:
            allowed = True
        elif len(parts) == 2 and parts[0] == "docs" and parts[1].lower().endswith(".md"):
            allowed = True
        elif len(parts) == 2 and parts[0] == "data":
            suffix = Path(parts[1]).suffix.lower()
            allowed = suffix == ".json" or suffix in STATIC_IMAGE_SUFFIXES
        elif len(parts) == 2 and parts[0] == "foods-example":
            allowed = Path(parts[1]).suffix.lower() in STATIC_IMAGE_SUFFIXES
        if not allowed:
            return None

        candidate = (ROOT / Path(*parts)).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def _serve_static(self, request_path: str) -> None:
        path = self._static_file(request_path)
        if path is None:
            error_response(self, "NOT_FOUND", "Resource not found.", HTTPStatus.NOT_FOUND)
            return
        try:
            content = path.read_bytes()
        except OSError:
            error_response(self, "NOT_FOUND", "Resource not found.", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        request_path = parsed.path
        if request_path == "/health":
            json_response(
                self,
                {"status": "ok", "service": "foodvision", "modelAvailable": MODEL.available},
            )
            return
        if request_path == "/api/v1/status":
            json_response(
                self,
                {
                    "status": "ok",
                    "modelAvailable": MODEL.available,
                    "model": MODEL.metadata(),
                    "classNames": list(MODEL.class_names),
                    "classSource": MODEL.class_source,
                    "referenceMetrics": METRICS.get("reference", {}),
                    "deployedMetrics": METRICS.get("deployed", {}),
                    "loadError": MODEL.load_error or None,
                },
            )
            return
        if request_path == "/api/v1/insights":
            self._handle_insights(parsed.query)
            return
        if request_path.startswith("/api/"):
            error_response(self, "NOT_FOUND", "API endpoint not found.", HTTPStatus.NOT_FOUND)
            return
        self._serve_static(request_path)

    def _handle_insights(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True)
        labels = values.get("label", [])
        if len(labels) != 1 or not labels[0] or len(labels[0]) > 200 or any(
            character in labels[0] for character in "\r\n\x00"
        ):
            error_response(self, "INVALID_LABEL", "A single food label is required.", HTTPStatus.BAD_REQUEST)
            return
        online_values = values.get("online", ["0"])
        if len(online_values) != 1 or online_values[0] not in {"0", "1"}:
            error_response(self, "INVALID_ONLINE", "online must be 0 or 1.", HTTPStatus.BAD_REQUEST)
            return
        builder = _load_build_insights()
        if builder is None:
            error_response(
                self,
                "INSIGHTS_UNAVAILABLE",
                "The knowledge layer is unavailable.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        try:
            result = builder(labels[0], online=online_values[0] == "1")
            if not isinstance(result, dict):
                raise ValueError
            _json_bytes(result)
        except Exception:
            error_response(
                self,
                "INSIGHTS_UNAVAILABLE",
                "The knowledge layer could not build insights.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        json_response(self, result)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/v1/predict":
            error_response(self, "NOT_FOUND", "API endpoint not found.", HTTPStatus.NOT_FOUND)
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            error_response(
                self,
                "INVALID_CONTENT_TYPE",
                "Send the image as multipart/form-data.",
                HTTPStatus.BAD_REQUEST,
            )
            return
        content_length = _read_content_length(self)
        if content_length is None:
            return
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            error_response(self, "INVALID_UPLOAD", "The upload body is incomplete.", HTTPStatus.BAD_REQUEST)
            return
        try:
            fields = parse_multipart(body, content_type)
        except MultipartError:
            error_response(self, "INVALID_MULTIPART", "The multipart upload is invalid.", HTTPStatus.BAD_REQUEST)
            return

        image_bytes = fields.get("image")
        if not image_bytes:
            error_response(self, "MISSING_IMAGE", "The image field is required.", HTTPStatus.BAD_REQUEST)
            return
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            error_response(
                self,
                "UPLOAD_TOO_LARGE",
                "The image exceeds the upload limit.",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        top_k = DEFAULT_TOP_K
        if "topK" in fields:
            try:
                top_k_text = fields["topK"].decode("utf-8")
                top_k = int(top_k_text, 10)
            except (UnicodeDecodeError, ValueError):
                error_response(self, "INVALID_TOP_K", "topK must be an integer from 1 to 5.", HTTPStatus.BAD_REQUEST)
                return
            if not 1 <= top_k <= MAX_TOP_K:
                error_response(self, "INVALID_TOP_K", "topK must be an integer from 1 to 5.", HTTPStatus.BAD_REQUEST)
                return

        if not MODEL.available:
            error_response(
                self,
                "MODEL_UNAVAILABLE",
                "The trained model is unavailable; no prediction was made.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        request_id = str(uuid.uuid4())
        try:
            predictions, latency_ms = MODEL.predict(image_bytes, top_k)
        except InvalidImageError:
            error_response(self, "INVALID_IMAGE", "The uploaded bytes are not a valid bounded image.", HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            error_response(
                self,
                "MODEL_INFERENCE_FAILED",
                "The model could not classify this image.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        if not predictions:
            error_response(
                self,
                "MODEL_INFERENCE_FAILED",
                "The model returned no valid prediction.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        best_food = FOODS.get(predictions[0]["classId"])
        json_response(
            self,
            {
                "status": "success",
                "requestId": request_id,
                "mode": "live",
                "model": MODEL.metadata(),
                "predictions": predictions,
                "latencyMs": latency_ms,
                "food": best_food,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[foodvision] {self.address_string()} - {format % args}")


def main() -> None:
    host = os.environ.get("FOODVISION_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, DEFAULT_PORT), FoodVisionHandler)
    print(f"FoodVision server running at http://{host}:{DEFAULT_PORT}")
    print(f"Live model available: {MODEL.available}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FoodVision server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
