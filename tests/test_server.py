"""Focused contract tests for the dependency-free HTTP surface."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import unittest
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import server  # noqa: E402


def multipart_body(image: bytes = b"image-bytes", top_k: str = "3", boundary: str = "test-boundary") -> tuple[bytes, str]:
    chunks = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.bin\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n".encode("ascii"),
        image,
        b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"topK\"\r\n\r\n{top_k}\r\n".encode(
            "ascii"
        ),
        f"--{boundary}--\r\n".encode("ascii"),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class FakeModel:
    available = True
    class_names = ["chicken_wings", "hot_dog"]
    class_source = "model_handoff"
    load_error = ""

    def __init__(self) -> None:
        self.received: tuple[bytes, int] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "fake-model",
            "version": "test",
            "inputSize": 2,
            "preprocessing": "none",
            "available": True,
        }

    def predict(self, image: bytes, top_k: int) -> tuple[list[dict[str, Any]], int]:
        self.received = (image, top_k)
        return (
            [
                {"classId": "hot_dog", "label": "Hot dog", "confidence": 0.8},
                {"classId": "chicken_wings", "label": "Chicken wings", "confidence": 0.2},
            ][:top_k],
            7,
        )


class UnavailableModel:
    available = False
    class_names: list[str] = []
    class_source = "project_placeholder"
    load_error = "Model file not found"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "FoodVision team model",
            "version": "unreleased",
            "inputSize": 299,
            "preprocessing": "inception_v3",
            "available": False,
        }


class ServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_model = server.MODEL
        self.original_limit = server.MAX_UPLOAD_BYTES
        self.original_builder_loader = server._load_build_insights
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.FoodVisionHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        server.MODEL = self.original_model
        server.MAX_UPLOAD_BYTES = self.original_limit
        server._load_build_insights = self.original_builder_loader

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | bytes, dict[str, str]]:
        host, port = self.httpd.server_address
        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response_body = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        connection.close()
        if response_headers.get("content-type", "").startswith("application/json"):
            return response.status, json.loads(response_body), response_headers
        return response.status, response_body, response_headers

    def test_live_prediction_uses_multipart_bytes_and_top_k(self) -> None:
        fake = FakeModel()
        server.MODEL = fake
        image = b"\x89PNG\x00\r\n--not-the-boundary\r\n\xff\x00"
        body, content_type = multipart_body(image=image, top_k="1")
        status, payload, _ = self.request(
            "POST",
            "/api/v1/predict",
            body,
            {"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["mode"], "live")
        self.assertEqual(len(payload["predictions"]), 1)
        self.assertEqual(payload["predictions"][0]["classId"], "hot_dog")
        self.assertEqual(fake.received, (image, 1))

    def test_unavailable_model_is_a_503_error(self) -> None:
        server.MODEL = UnavailableModel()
        body, content_type = multipart_body()
        status, payload, _ = self.request(
            "POST",
            "/api/v1/predict",
            body,
            {"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload, {
            "status": "error",
            "error": "MODEL_UNAVAILABLE",
            "message": "The trained model is unavailable; no prediction was made.",
        })

    def test_status_exposes_ordered_classes_and_metrics_without_cors(self) -> None:
        server.MODEL = UnavailableModel()
        status, payload, headers = self.request("GET", "/api/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["modelAvailable"], False)
        self.assertEqual(payload["classNames"], [])
        self.assertIn("referenceMetrics", payload)
        self.assertIn("deployedMetrics", payload)
        self.assertIn("loadError", payload)
        self.assertNotIn("access-control-allow-origin", headers)

    def test_insights_route_lazy_loads_mock_knowledge(self) -> None:
        calls: list[tuple[str, bool]] = []

        def build_insights(label: str, *, online: bool) -> dict[str, Any]:
            calls.append((label, online))
            return {
                "classId": "chicken_wings",
                "label": label,
                "overview": "overview",
                "recipe": [],
                "nutrition": {},
                "sources": [],
                "retrieval": {"online": online},
                "restaurants": [],
            }

        server._load_build_insights = lambda: build_insights
        status, payload, _ = self.request(
            "GET", "/api/v1/insights?label=" + quote("chicken_wings") + "&online=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["retrieval"]["online"], True)
        self.assertEqual(calls, [("chicken_wings", True)])

    def test_static_allowlist_denies_private_and_traversal_paths(self) -> None:
        status, _, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        for path in ("/backend/server.py", "/../backend/server.py", "/%2e%2e/backend/server.py", "/data/"):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload["status"], "error")

    def test_upload_limit_is_checked_before_prediction(self) -> None:
        server.MODEL = FakeModel()
        server.MAX_UPLOAD_BYTES = 4
        body, content_type = multipart_body(image=b"12345")
        status, payload, _ = self.request(
            "POST",
            "/api/v1/predict",
            body,
            {"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"], "UPLOAD_TOO_LARGE")

    def test_bad_multipart_is_a_uniform_json_error(self) -> None:
        server.MODEL = FakeModel()
        body = b"not multipart"
        status, payload, headers = self.request(
            "POST",
            "/api/v1/predict",
            body,
            {"Content-Type": "multipart/form-data", "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "INVALID_MULTIPART")
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")

    def test_label_order_and_output_contract_are_strict(self) -> None:
        self.assertEqual(server._validate_class_names(["z", "a"]), ["z", "a"])
        with self.assertRaises(server.ConfigurationError):
            server._validate_class_names(["z", "z"])
        self.assertEqual(
            server._canonical_preprocessing("inception_v3_preprocess_input"), "inception_v3"
        )
        self.assertEqual(server._canonical_preprocessing("none"), "none")
        with self.assertRaises(server.ConfigurationError):
            server._canonical_preprocessing("inception_v3_preprocess_input_typo")
        self.assertEqual(
            server._validated_probabilities([[0.25, 0.75]], 2, "probabilities"), [0.25, 0.75]
        )
        with self.assertRaises(server.InvalidModelOutput):
            server._validated_probabilities([0.25, 0.25], 2, "probabilities")
        with self.assertRaises(server.InvalidModelOutput):
            server._validated_probabilities([0.2, 0.3, 0.5], 2, "probabilities")
        logits = server._validated_probabilities([0.0, 1.0], 2, "logits")
        self.assertAlmostEqual(sum(logits), 1.0)


if __name__ == "__main__":
    unittest.main()
