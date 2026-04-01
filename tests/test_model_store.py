import os
import numpy as np
import pytest
import lightgbm as lgb
from app.ml.model_store import ModelStore
from app.ml.lgbm_predictor import LgbmPredictor


@pytest.fixture
def store(tmp_path):
    return ModelStore(str(tmp_path))


@pytest.fixture
def dummy_booster():
    rng = np.random.default_rng(42)
    X = rng.random((50, 5)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    dataset = lgb.Dataset(X, label=y)
    return lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 4},
        dataset, num_boost_round=3,
    )


def test_save_and_load_with_battle_type(store, dummy_booster):
    predictor = LgbmPredictor(dummy_booster, version="lgbm-v1")
    store.save_current("pathOfLegend", predictor)
    loaded = store.load_current("pathOfLegend")
    assert loaded is not None
    assert loaded.version == "lgbm-v1"


def test_load_returns_none_when_no_model(store):
    assert store.load_current("pathOfLegend") is None


def test_save_creates_backup(store, dummy_booster):
    old = LgbmPredictor(dummy_booster, version="lgbm-v1")
    store.save_current("pathOfLegend", old)

    new = LgbmPredictor(dummy_booster, version="lgbm-v2")
    store.save_current("pathOfLegend", new)

    backup_path = os.path.join(store._dir, "matchup_pathOfLegend_backup.joblib")
    assert os.path.exists(backup_path)

    loaded = store.load_current("pathOfLegend")
    assert loaded.version == "lgbm-v2"


def test_different_modes_independent(store, dummy_booster):
    pol = LgbmPredictor(dummy_booster, version="pol-v1")
    store.save_current("pathOfLegend", pol)

    assert store.load_current("ladder") is None
    assert store.load_current("pathOfLegend").version == "pol-v1"


def test_has_current(store, dummy_booster):
    assert store.has_current("pathOfLegend") is False
    store.save_current("pathOfLegend", LgbmPredictor(dummy_booster, "v1"))
    assert store.has_current("pathOfLegend") is True
