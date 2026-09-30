"""Regression coverage for API dependency health reporting."""

from __future__ import annotations

import unittest

import httpx

from api.cache import QueryCache
from api.embeddings import OpenAIEmbedder
from api.main import healthz
from api.service import MemoryService
from api.settings import Settings


class _FakePool:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def fetchval(self, _query: str) -> int:
        if self.fail:
            raise RuntimeError("database unavailable")
        return 7


class _FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def ping(self) -> bool:
        if self.fail:
            raise RuntimeError("redis unavailable")
        return True


def _embedder(handler):
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="test-key",
    )
    embedder = OpenAIEmbedder(settings)
    embedder._http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return embedder


class HealthServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthy_dependencies_report_ok(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                request=request,
            )

        embedder = _embedder(handler)
        try:
            cache = QueryCache(_FakeRedis(), 300)
            service = MemoryService(
                _FakePool(),
                embedder,
                cache,
                Settings(),
            )

            result = await service.health()

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["checks"], {"postgres": "ok", "cache": "ok", "embedder": "ok"})
            self.assertEqual(result["embedder"], "openai")
            self.assertEqual(result["embedder_status"], "ok")
            self.assertEqual(result["memories"], 7)
            self.assertEqual(result["errors"], {})
        finally:
            await embedder.aclose()

    async def test_redis_failure_does_not_hide_other_dependency_states(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                request=request,
            )

        embedder = _embedder(handler)
        try:
            service = MemoryService(
                _FakePool(),
                embedder,
                QueryCache(_FakeRedis(fail=True), 300),
                Settings(),
            )

            result = await service.health()

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["checks"]["postgres"], "ok")
            self.assertEqual(result["checks"]["cache"], "error")
            self.assertEqual(result["checks"]["embedder"], "ok")
            self.assertIn("Redis PING failed", result["errors"]["cache"])
        finally:
            await embedder.aclose()

    async def test_embedder_http_failure_reports_provider_and_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="service unavailable", request=request)

        embedder = _embedder(handler)
        try:
            service = MemoryService(
                _FakePool(),
                embedder,
                QueryCache(_FakeRedis(), 300),
                Settings(),
            )

            result = await service.health()

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["checks"]["postgres"], "ok")
            self.assertEqual(result["checks"]["cache"], "ok")
            self.assertEqual(result["checks"]["embedder"], "error")
            self.assertIn("openai https://api.openai.com/v1/embeddings", result["errors"]["embedder"])
            self.assertIn("HTTPStatusError", result["errors"]["embedder"])
        finally:
            await embedder.aclose()

    async def test_embedder_connection_failure_reports_provider(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        embedder = _embedder(handler)
        try:
            service = MemoryService(
                _FakePool(),
                embedder,
                QueryCache(_FakeRedis(), 300),
                Settings(),
            )

            result = await service.health()

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["checks"]["embedder"], "error")
            self.assertIn("openai https://api.openai.com/v1/embeddings", result["errors"]["embedder"])
            self.assertIn("ConnectError", result["errors"]["embedder"])
        finally:
            await embedder.aclose()

    async def test_postgres_failure_reports_error_without_counting_memory(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                request=request,
            )

        embedder = _embedder(handler)
        try:
            service = MemoryService(
                _FakePool(fail=True),
                embedder,
                QueryCache(_FakeRedis(), 300),
                Settings(),
            )

            result = await service.health()

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["checks"]["postgres"], "error")
            self.assertEqual(result["checks"]["cache"], "ok")
            self.assertEqual(result["checks"]["embedder"], "ok")
            self.assertIsNone(result["memories"])
            self.assertIn("PostgreSQL probe failed", result["errors"]["postgres"])
        finally:
            await embedder.aclose()


class HealthRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_unhealthy_service_maps_to_503(self) -> None:
        class UnhealthyService:
            async def health(self) -> dict[str, object]:
                return {
                    "status": "error",
                    "checks": {"postgres": "ok", "cache": "error", "embedder": "ok"},
                    "errors": {"cache": "Redis PING failed"},
                }

        with self.assertRaises(httpx.HTTPStatusError):
            # FastAPI exposes HTTPException, but importing it only for a test
            # makes this test less direct than asserting its status/detail below.
            await self._assert_http_exception(UnhealthyService())

    async def _assert_http_exception(self, service) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            await healthz(service)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail["status"], "error")
        self.assertEqual(caught.exception.detail["errors"]["cache"], "Redis PING failed")


if __name__ == "__main__":
    unittest.main()