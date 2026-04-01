import numpy as np
import pytest
from app.ml.predictor import Predictor
from app.ml.lgbm_predictor import LgbmPredictor


def test_predictor_is_abstract():
    with pytest.raises(TypeError):
        Predictor()


def test_lgbm_predictor_predict_returns_probability():
    import lightgbm as lgb
    rng = np.random.default_rng(42)
    X = rng.random((100, 10)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 4},
        dataset, num_boost_round=5,
    )
    predictor = LgbmPredictor(booster, version="test-v1")
    prob = predictor.predict(X[0:1])
    assert 0.0 <= prob[0] <= 1.0
    assert predictor.version == "test-v1"
