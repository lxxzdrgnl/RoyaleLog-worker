import asyncio
import pytest
import numpy as np
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from app.ml.feature_builder import FeatureBuilder


def _make_test_app(with_model: bool = True):
    from app.main import create_app

    @asynccontextmanager
    async def fake_lifespan(app):
        yield

    test_app = create_app()
    test_app.router.lifespan_context = fake_lifespan

    card_rows = [(26000000 + i, "NORMAL", None) for i in range(120)]
    card_rows += [(26000000, "EVOLUTION", 1), (159000000, "NORMAL", None)]
    test_app.state.feature_builder = FeatureBuilder.from_card_rows(card_rows)
    test_app.state.model_lock = asyncio.Lock()

    predictors = {}
    if with_model:
        mock = MagicMock()
        mock.predict.return_value = np.array([0.62])
        mock.version = "test-v1"
        predictors["pathOfLegend"] = mock
    test_app.state.predictors = predictors
    return test_app


_PREDICT_PAYLOAD = {
    "request_id": "test-uuid-1234",
    "deck_card_ids": [26000000 + i for i in range(8)],
    "deck_card_levels": [13] * 8,
    "deck_evo_levels": [1, 0, 0, 0, 0, 0, 0, 0],
    "opponent_card_ids": [26000010 + i for i in range(8)],
    "opponent_card_levels": [12] * 8,
    "opponent_evo_levels": [0] * 8,
    "battle_type": "pathOfLegend",
    "league_number": 7,
}


@pytest.mark.asyncio
async def test_predict_matchup_returns_probability():
    test_app = _make_test_app()
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/predict/matchup", json=_PREDICT_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "test-uuid-1234"
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["model_version"] == "test-v1"


@pytest.mark.asyncio
async def test_predict_503_when_no_model():
    test_app = _make_test_app(with_model=False)
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/predict/matchup", json=_PREDICT_PAYLOAD)
    assert resp.status_code == 503
    body = resp.json()
    # Should use standard error format
    assert "code" in body
    assert body["code"] == "MODEL_NOT_LOADED"
    assert "timestamp" in body
    assert "path" in body


@pytest.mark.asyncio
async def test_predict_422_when_deck_too_short():
    test_app = _make_test_app()
    bad_payload = dict(_PREDICT_PAYLOAD)
    bad_payload["deck_card_ids"] = [26000000, 26000001]  # only 2 cards
    bad_payload["deck_card_levels"] = [13, 13]
    bad_payload["deck_evo_levels"] = [0, 0]
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/predict/matchup", json=bad_payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "code" in body
    assert body["code"] == "VALIDATION_FAILED"
    assert "details" in body
