import asyncio
import json
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport


def _make_test_app(tmp_path):
    from app.main import create_app
    from app.core.config import Settings

    @asynccontextmanager
    async def fake_lifespan(app):
        yield

    test_app = create_app()
    test_app.router.lifespan_context = fake_lifespan

    settings = Settings()
    settings.state_file_path = str(tmp_path / "training_state.json")
    settings.model_dir = str(tmp_path)
    test_app.state.settings = settings
    test_app.state.model_lock = asyncio.Lock()
    test_app.state.predictors = {}
    return test_app


@pytest.mark.asyncio
async def test_train_status_idle(tmp_path):
    test_app = _make_test_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/train/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


@pytest.mark.asyncio
async def test_train_reset_window(tmp_path):
    test_app = _make_test_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/train?reset_window=true")
    assert resp.status_code == 202
    assert "Patch date" in resp.json()["message"]

    state_path = test_app.state.settings.state_file_path
    with open(state_path) as f:
        state = json.load(f)
    assert "patch_date" in state


@pytest.mark.asyncio
async def test_train_409_when_running(tmp_path):
    test_app = _make_test_app(tmp_path)
    state_path = test_app.state.settings.state_file_path
    with open(state_path, "w") as f:
        json.dump({"status": "running"}, f)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/train")
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "TRAINING_CONFLICT"
    assert "timestamp" in body
