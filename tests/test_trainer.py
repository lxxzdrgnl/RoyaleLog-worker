import numpy as np
import pytest
from app.ml.trainer import Trainer, TrainResult
from app.ml.feature_builder import FeatureBuilder
from app.ml.model_store import ModelStore


@pytest.fixture
def card_rows():
    return [
        (26000000, "NORMAL", None),
        (26000000, "EVOLUTION", 1),
        (26000001, "NORMAL", None),
        (26000002, "NORMAL", None),
        (26000003, "NORMAL", None),
        (26000004, "NORMAL", None),
        (26000005, "NORMAL", None),
        (26000006, "NORMAL", None),
        (26000007, "NORMAL", None),
        (159000000, "NORMAL", None),
    ]


@pytest.fixture
def builder(card_rows):
    return FeatureBuilder.from_card_rows(card_rows)


@pytest.fixture
def store(tmp_path):
    return ModelStore(str(tmp_path))


def _make_fake_data(builder: FeatureBuilder, battle_type: str, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    card_pool = sorted(builder.api_id_to_index.keys())
    X_rows, y_rows = [], []
    for _ in range(n):
        my_cards = list(rng.choice(card_pool, size=8, replace=False))
        opp_cards = list(rng.choice(card_pool, size=8, replace=False))
        my_levels = [int(rng.integers(8, 15)) for _ in range(8)]
        opp_levels = [int(rng.integers(8, 15)) for _ in range(8)]
        my_evo = [int(rng.integers(0, 2)) if c in builder.evo_card_to_index else 0 for c in my_cards]
        opp_evo = [int(rng.integers(0, 2)) if c in builder.evo_card_to_index else 0 for c in opp_cards]
        vec = builder.build_matchup_vector(
            battle_type=battle_type,
            deck_card_ids=my_cards, deck_card_levels=my_levels, deck_evo_levels=my_evo,
            opponent_card_ids=opp_cards, opponent_card_levels=opp_levels, opponent_evo_levels=opp_evo,
            avg_level=float(np.mean(my_levels)),
            evolution_count=sum(1 for e in my_evo if e > 0),
            league_number=int(rng.integers(0, 10)),
            starting_trophies=int(rng.integers(4000, 7000)),
        )
        X_rows.append(vec)
        y_rows.append(int(rng.integers(0, 2)))
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def test_first_model_always_saves(builder, store):
    X, y = _make_fake_data(builder, "pathOfLegend", 200)
    trainer = Trainer(
        builder=builder, store=store, battle_type="pathOfLegend",
        min_child_samples=5, early_stopping_rounds=5, accuracy_margin=0.005,
    )
    result = trainer.train_from_arrays(X, y, val_start_index=150)
    assert result.saved is True
    assert store.has_current("pathOfLegend")


def test_rejects_worse_model(builder, store):
    X, y = _make_fake_data(builder, "pathOfLegend", 200)
    trainer = Trainer(
        builder=builder, store=store, battle_type="pathOfLegend",
        min_child_samples=5, early_stopping_rounds=5, accuracy_margin=0.005,
    )
    trainer.train_from_arrays(X, y, val_start_index=150)

    # accuracy_margin=0.99 means new model needs to be 99% better — impossible
    trainer2 = Trainer(
        builder=builder, store=store, battle_type="pathOfLegend",
        min_child_samples=5, early_stopping_rounds=5, accuracy_margin=0.99,
    )
    result2 = trainer2.train_from_arrays(X, y, val_start_index=150)
    assert result2.saved is False


def test_different_modes_save_independently(builder, store):
    X_pol, y_pol = _make_fake_data(builder, "pathOfLegend", 200, seed=1)
    X_lad, y_lad = _make_fake_data(builder, "ladder", 200, seed=2)

    t_pol = Trainer(builder=builder, store=store, battle_type="pathOfLegend",
                    min_child_samples=5, early_stopping_rounds=5, accuracy_margin=0.005)
    t_lad = Trainer(builder=builder, store=store, battle_type="ladder",
                    min_child_samples=5, early_stopping_rounds=5, accuracy_margin=0.005)

    t_pol.train_from_arrays(X_pol, y_pol, val_start_index=150)
    t_lad.train_from_arrays(X_lad, y_lad, val_start_index=150)

    assert store.has_current("pathOfLegend")
    assert store.has_current("ladder")
