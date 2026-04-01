import os
import pytest


def test_settings_loads_defaults():
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_PORT", "5434")
    os.environ.setdefault("DB_NAME", "royalelog")
    os.environ.setdefault("DB_USER", "royale")
    os.environ.setdefault("DB_PASSWORD", "test")
    from app.core.config import Settings
    s = Settings()
    assert s.db_host == "localhost"
    assert s.window_days == 3
    assert s.val_days == 1
    assert s.min_train_rows == 10000
    assert s.accuracy_margin == 0.005
    assert s.worker_port == 8082
    assert s.train_battle_types == ["pathOfLegend", "ladder"]
    assert s.state_file_path == "./training_state.json"


def test_settings_db_url():
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_PORT", "5434")
    os.environ.setdefault("DB_NAME", "royalelog")
    os.environ.setdefault("DB_USER", "royale")
    os.environ.setdefault("DB_PASSWORD", "test")
    from app.core.config import Settings
    s = Settings()
    assert "postgresql" in s.db_url
    assert "royalelog" in s.db_url
