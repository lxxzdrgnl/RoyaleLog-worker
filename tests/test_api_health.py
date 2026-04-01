import asyncio
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport


def _make_test_app():
    from app.main import create_app

    @asynccontextmanager
    async def fake_lifespan(app):
        yield

    test_app = create_app()
    test_app.router.lifespan_context = fake_lifespan
    return test_app


@pytest.mark.asyncio
async def test_health_returns_ok():
    test_app = _make_test_app()
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
