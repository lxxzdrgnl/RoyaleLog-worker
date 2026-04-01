import pytest
from app.core.config import Settings


def test_settings_loads_defaults(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5434")
    monkeypatch.setenv("DB_NAME", "royalelog")
    monkeypatch.setenv("DB_USER", "royale")
    monkeypatch.setenv("DB_PASSWORD", "test")
    s = Settings()
    assert s.db_host == "localhost"
    assert s.window_days == 3
    assert s.val_days == 1
    assert s.min_train_rows == 10000
    assert s.accuracy_margin == 0.005
    assert s.worker_port == 8082
    assert s.train_battle_types == ["pathOfLegend", "ladder"]
    assert s.state_file_path == "./training_state.json"


def test_settings_db_url(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5434")
    monkeypatch.setenv("DB_NAME", "royalelog")
    monkeypatch.setenv("DB_USER", "royale")
    monkeypatch.setenv("DB_PASSWORD", "test")
    s = Settings()
    assert "postgresql" in s.db_url
    assert "royalelog" in s.db_url
