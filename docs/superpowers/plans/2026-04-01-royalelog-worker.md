# RoyaleLog Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI 기반 ML 서버 — 모드별(PoL, ladder) LightGBM 덱 매치업 승률 예측 + Airflow 트리거 재학습

**Architecture:** 단일 FastAPI 프로세스에서 `/predict/matchup` (동기 추론 <50ms, battle_type으로 모델 선택) + `/train` (BackgroundTasks → threadpool 비동기 학습, 모드별 순차) 제공. `cards` 테이블에서 피처 벡터 크기 동적 결정. `training_state.json`으로 학습 상태 영속화. MLflow Tracking Server에 실험 기록.

**Tech Stack:** Python 3.11+, FastAPI, LightGBM, SQLAlchemy, MLflow, Pydantic Settings, joblib, uvicorn

**Spec:** `docs/superpowers/specs/2026-04-01-royalelog-worker-design.md`

**주요 설계 결정:**
- 모드별 모델: pathOfLegend (avg_level 제외), ladder (avg_level 포함). 특별모드는 stats_fallback.
- `_run_training`은 동기 함수(`def`) — threadpool 자동 실행, 이벤트 루프 블로킹 방지
- 핫스왑은 `asyncio.run_coroutine_threadsafe()`로 이벤트 루프에 위임
- DB 세션은 context manager 패턴 사용

---

## File Structure

```
RoyaleLog-worker/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 앱, lifespan, router 등록
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py            # GET /health
│   │   ├── predict.py           # POST /predict/matchup
│   │   └── train.py             # POST /train, GET /train/status
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings (환경변수)
│   │   └── database.py          # SQLAlchemy engine + context manager
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── predictor.py         # Predictor ABC
│   │   ├── lgbm_predictor.py    # LightGBM Predictor 구현체
│   │   ├── feature_builder.py   # DB → 피처 행렬 변환 (모드별)
│   │   ├── model_store.py       # 모델 파일 로드/저장 (모드별)
│   │   └── trainer.py           # 학습 파이프라인 오케스트레이션
│   └── schemas/
│       ├── __init__.py
│       ├── predict.py           # PredictRequest / PredictResponse
│       └── train.py             # TrainResponse / TrainStatusResponse
├── models/                      # .joblib 파일 저장 (gitignore)
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # 공통 fixture
│   ├── test_config.py
│   ├── test_feature_builder.py
│   ├── test_predictor.py
│   ├── test_model_store.py
│   ├── test_trainer.py
│   ├── test_api_health.py
│   ├── test_api_predict.py
│   └── test_api_train.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```

---

### Task 1: 프로젝트 스캐폴딩 + 설정

**Files:**
- Create: `RoyaleLog-worker/requirements.txt`
- Create: `RoyaleLog-worker/.env.example`
- Create: `RoyaleLog-worker/.gitignore`
- Create: `RoyaleLog-worker/models/.gitkeep`
- Create: `RoyaleLog-worker/app/__init__.py`
- Create: `RoyaleLog-worker/app/core/__init__.py`
- Create: `RoyaleLog-worker/app/core/config.py`
- Create: `RoyaleLog-worker/app/core/database.py`
- Create: `RoyaleLog-worker/tests/__init__.py`
- Create: `RoyaleLog-worker/tests/test_config.py`

- [ ] **Step 1: requirements.txt 생성**

```txt
fastapi==0.115.12
uvicorn[standard]==0.34.2
sqlalchemy==2.0.40
psycopg2-binary==2.9.10
lightgbm==4.6.0
joblib==1.4.2
mlflow==2.22.0
pydantic-settings==2.9.1
numpy==2.2.4
pandas==2.2.3
scikit-learn==1.6.1
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==0.26.0
```

- [ ] **Step 2: .env.example 생성**

```env
DB_HOST=localhost
DB_PORT=5434
DB_NAME=royalelog
DB_USER=royale
DB_PASSWORD=

MLFLOW_TRACKING_URI=http://localhost:5000

MODEL_DIR=./models
STATE_FILE_PATH=./training_state.json
WINDOW_DAYS=3
VAL_DAYS=1
MIN_CHILD_SAMPLES=50
MIN_TRAIN_ROWS=10000
MIN_POST_PATCH_DAYS=3
EARLY_STOPPING_ROUNDS=50
ACCURACY_MARGIN=0.005
TRAIN_BATTLE_TYPES=pathOfLegend,ladder

WORKER_PORT=8082
```

- [ ] **Step 3: .gitignore 생성**

```
__pycache__/
*.pyc
.env
models/*.joblib
training_state.json
mlruns/
.pytest_cache/
*.egg-info/
dist/
venv/
```

- [ ] **Step 4: models/.gitkeep 생성**

빈 파일.

- [ ] **Step 5: config.py 테스트 작성**

```python
# tests/test_config.py
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
```

- [ ] **Step 6: 테스트 실행, 실패 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7: config.py 구현**

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # DB
    db_host: str = "localhost"
    db_port: int = 5434
    db_name: str = "royalelog"
    db_user: str = "royale"
    db_password: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model
    model_dir: str = "./models"
    state_file_path: str = "./training_state.json"
    window_days: int = 3
    val_days: int = 1
    min_child_samples: int = 50
    min_train_rows: int = 10000
    min_post_patch_days: int = 3
    early_stopping_rounds: int = 50
    accuracy_margin: float = 0.005
    train_battle_types: list[str] = ["pathOfLegend", "ladder"]

    # Server
    worker_port: int = 8082

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
```

- [ ] **Step 8: database.py 구현**

```python
# app/core/database.py
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

_engine = None
_SessionLocal = None


def init_db(db_url: str) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(db_url, pool_size=5, max_overflow=10)
    _SessionLocal = sessionmaker(bind=_engine)


@contextmanager
def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

- [ ] **Step 9: `__init__.py` 파일 생성**

`app/__init__.py`, `app/core/__init__.py`, `tests/__init__.py` — 모두 빈 파일.

- [ ] **Step 10: 테스트 통과 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 11: Commit**

```bash
git add RoyaleLog-worker/
git commit -m "feat(worker): scaffold project with config, database, and dependencies"
```

---

### Task 2: Schemas (Request/Response DTO)

**Files:**
- Create: `RoyaleLog-worker/app/schemas/__init__.py`
- Create: `RoyaleLog-worker/app/schemas/predict.py`
- Create: `RoyaleLog-worker/app/schemas/train.py`

- [ ] **Step 1: schemas/predict.py 작성**

```python
# app/schemas/predict.py
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    request_id: str = Field(..., description="Spring Boot가 생성한 UUID")
    deck_card_ids: list[int] = Field(..., min_length=8, max_length=9)
    deck_card_levels: list[int] = Field(..., min_length=8, max_length=9)
    deck_evo_levels: list[int] = Field(..., min_length=8, max_length=9)
    opponent_card_ids: list[int] = Field(..., min_length=8, max_length=9)
    opponent_card_levels: list[int] = Field(..., min_length=8, max_length=9)
    opponent_evo_levels: list[int] = Field(..., min_length=8, max_length=9)
    battle_type: str = "pathOfLegend"
    league_number: int | None = None
    starting_trophies: int | None = None


class PredictResponse(BaseModel):
    request_id: str
    win_probability: float
    model_version: str | None
```

- [ ] **Step 2: schemas/train.py 작성**

```python
# app/schemas/train.py
from datetime import datetime
from pydantic import BaseModel


class TrainResponse(BaseModel):
    status: str  # "accepted" or "conflict"
    message: str


class TrainStatusResponse(BaseModel):
    status: str  # idle, running, success, failed
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_version: str | None = None
    train_rows: int | None = None
    val_rows: int | None = None
    val_accuracy: float | None = None
    val_logloss: float | None = None
    error: str | None = None
```

- [ ] **Step 3: `app/schemas/__init__.py` 빈 파일 생성**

- [ ] **Step 4: Commit**

```bash
git add RoyaleLog-worker/app/schemas/
git commit -m "feat(worker): add request/response schemas"
```

---

### Task 3: FeatureBuilder — 모드별 피처 변환

**Files:**
- Create: `RoyaleLog-worker/app/ml/__init__.py`
- Create: `RoyaleLog-worker/app/ml/feature_builder.py`
- Create: `RoyaleLog-worker/tests/test_feature_builder.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_feature_builder.py
import numpy as np
import pytest
from app.ml.feature_builder import FeatureBuilder


@pytest.fixture
def card_rows():
    """(api_id, card_type, max_evo_level) 튜플 리스트."""
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


def test_pol_feature_count(builder):
    """PoL: (카드 + 진화 + 2수치) × 2."""
    names = builder.feature_names("pathOfLegend")
    unique_api_ids = 9
    evo_cards = 1
    numeric = 2  # evolution_count, league_number
    expected = (unique_api_ids + evo_cards + numeric) * 2
    assert len(names) == expected


def test_ladder_feature_count(builder):
    """Ladder: (카드 + 진화 + 3수치) × 2."""
    names = builder.feature_names("ladder")
    unique_api_ids = 9
    evo_cards = 1
    numeric = 3  # avg_level, evolution_count, starting_trophies
    expected = (unique_api_ids + evo_cards + numeric) * 2
    assert len(names) == expected


def test_encode_deck_presence_and_evo(builder):
    """카드 존재=1, 진화=1 인코딩."""
    card_ids = [26000000, 26000001, 26000002, 26000003,
                26000004, 26000005, 26000006, 159000000]
    evo_levels = [1, 0, 0, 0, 0, 0, 0, 0]
    vec = builder.encode_deck(card_ids, evo_levels)

    knight_idx = builder.api_id_to_index[26000000]
    assert vec[knight_idx] == 1

    evo_start = len(builder.api_id_to_index)
    evo_knight_idx = evo_start + builder.evo_card_to_index[26000000]
    assert vec[evo_knight_idx] == 1

    arrows_idx = builder.api_id_to_index[26000007]
    assert vec[arrows_idx] == 0


def test_build_matchup_vector_pol(builder):
    """PoL matchup 벡터: avg_level 미포함, league_number 포함."""
    my_cards = [26000000, 26000001, 26000002, 26000003,
                26000004, 26000005, 26000006, 159000000]
    opp_cards = [26000001, 26000002, 26000003, 26000004,
                 26000005, 26000006, 26000007, 159000000]
    vec = builder.build_matchup_vector(
        battle_type="pathOfLegend",
        deck_card_ids=my_cards, deck_evo_levels=[1, 0, 0, 0, 0, 0, 0, 0],
        opponent_card_ids=opp_cards, opponent_evo_levels=[0] * 8,
        league_number=7, starting_trophies=None,
        avg_level=12.5, evolution_count=1,
    )
    names = builder.feature_names("pathOfLegend")
    assert len(vec) == len(names)
    assert "my_avg_level" not in names
    assert "my_league_number" in names


def test_build_matchup_vector_ladder(builder):
    """Ladder matchup 벡터: avg_level 포함, starting_trophies 포함."""
    my_cards = [26000000, 26000001, 26000002, 26000003,
                26000004, 26000005, 26000006, 159000000]
    opp_cards = [26000001, 26000002, 26000003, 26000004,
                 26000005, 26000006, 26000007, 159000000]
    vec = builder.build_matchup_vector(
        battle_type="ladder",
        deck_card_ids=my_cards, deck_evo_levels=[0] * 8,
        opponent_card_ids=opp_cards, opponent_evo_levels=[0] * 8,
        league_number=None, starting_trophies=5500,
        avg_level=13.0, evolution_count=0,
    )
    names = builder.feature_names("ladder")
    assert len(vec) == len(names)
    assert "my_avg_level" in names
    assert "my_starting_trophies" in names
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_feature_builder.py -v`
Expected: FAIL

- [ ] **Step 3: feature_builder.py 구현**

```python
# app/ml/feature_builder.py
from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session


# 모드별 수치 피처 정의
MODE_NUMERIC_FEATURES: dict[str, list[str]] = {
    "pathOfLegend": ["evolution_count", "league_number"],
    "ladder": ["avg_level", "evolution_count", "starting_trophies"],
}

# 지원되지 않는 모드는 pathOfLegend 피처 사용
DEFAULT_NUMERIC = ["evolution_count", "league_number"]


class FeatureBuilder:
    """cards 테이블 기반 동적 피처 벡터 생성. 모드별 수치 피처 분기."""

    def __init__(
        self,
        api_id_to_index: dict[int, int],
        evo_card_to_index: dict[int, int],
    ):
        self.api_id_to_index = api_id_to_index
        self.evo_card_to_index = evo_card_to_index
        self._num_cards = len(api_id_to_index)
        self._num_evo = len(evo_card_to_index)
        self._deck_size = self._num_cards + self._num_evo

    @classmethod
    def from_card_rows(
        cls, rows: list[tuple[int, str, int | None]]
    ) -> FeatureBuilder:
        unique_api_ids: set[int] = set()
        evo_api_ids: set[int] = set()
        for api_id, card_type, max_evo_level in rows:
            unique_api_ids.add(api_id)
            if card_type == "EVOLUTION":
                evo_api_ids.add(api_id)

        sorted_ids = sorted(unique_api_ids)
        api_id_to_index = {aid: i for i, aid in enumerate(sorted_ids)}

        sorted_evo = sorted(evo_api_ids)
        evo_card_to_index = {aid: i for i, aid in enumerate(sorted_evo)}

        return cls(api_id_to_index, evo_card_to_index)

    @classmethod
    def from_db(cls, session: Session) -> FeatureBuilder:
        result = session.execute(
            text("SELECT DISTINCT api_id, card_type, max_evo_level FROM cards")
        )
        rows = [(r[0], r[1], r[2]) for r in result]
        return cls.from_card_rows(rows)

    def _numeric_features(self, battle_type: str) -> list[str]:
        return MODE_NUMERIC_FEATURES.get(battle_type, DEFAULT_NUMERIC)

    def feature_names(self, battle_type: str) -> list[str]:
        numeric = self._numeric_features(battle_type)
        sorted_ids = sorted(self.api_id_to_index, key=self.api_id_to_index.get)
        sorted_evo = sorted(self.evo_card_to_index, key=self.evo_card_to_index.get)
        names = []
        for prefix in ("my", "opp"):
            for aid in sorted_ids:
                names.append(f"{prefix}_card_{aid}")
            for aid in sorted_evo:
                names.append(f"{prefix}_evo_{aid}")
            for feat in numeric:
                names.append(f"{prefix}_{feat}")
        return names

    def encode_deck(
        self, card_ids: list[int], card_levels: list[int], evo_levels: list[int]
    ) -> np.ndarray:
        vec = np.zeros(self._deck_size, dtype=np.float32)
        for cid, lvl, evo in zip(card_ids, card_levels, evo_levels):
            idx = self.api_id_to_index.get(cid)
            if idx is not None:
                vec[idx] = float(lvl)  # 레벨값 (0~14)
            if evo > 0 and cid in self.evo_card_to_index:
                evo_idx = self._num_cards + self.evo_card_to_index[cid]
                vec[evo_idx] = 1
        return vec

    def _numeric_vector(
        self,
        battle_type: str,
        evolution_count: int,
        league_number: int | None,
        starting_trophies: int | None,
    ) -> np.ndarray:
        features = self._numeric_features(battle_type)
        values = {
            "evolution_count": float(evolution_count),
            "league_number": float(league_number) if league_number is not None else -1.0,
            "starting_trophies": float(starting_trophies) if starting_trophies is not None else -1.0,
        }
        return np.array([values[f] for f in features], dtype=np.float32)

    def build_matchup_vector(
        self,
        battle_type: str,
        deck_card_ids: list[int],
        deck_card_levels: list[int],
        deck_evo_levels: list[int],
        opponent_card_ids: list[int],
        opponent_card_levels: list[int],
        opponent_evo_levels: list[int],
        league_number: int | None,
        starting_trophies: int | None,
        evolution_count: int,
    ) -> np.ndarray:
        my_deck = self.encode_deck(deck_card_ids, deck_card_levels, deck_evo_levels)
        my_numeric = self._numeric_vector(
            battle_type, evolution_count, league_number, starting_trophies
        )
        my_side = np.concatenate([my_deck, my_numeric])

        opp_deck = self.encode_deck(opponent_card_ids, opponent_card_levels, opponent_evo_levels)
        # 상대 수치 피처: 상대 evo_count는 DB에 없음 → 0
        opp_numeric = self._numeric_vector(
            battle_type, evolution_count=0,
            league_number=league_number, starting_trophies=starting_trophies,
        )
        opp_side = np.concatenate([opp_deck, opp_numeric])

        return np.concatenate([my_side, opp_side])
```

- [ ] **Step 4: `app/ml/__init__.py` 빈 파일 생성**

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_feature_builder.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add RoyaleLog-worker/app/ml/ RoyaleLog-worker/tests/test_feature_builder.py
git commit -m "feat(worker): implement FeatureBuilder with per-mode numeric features"
```

---

### Task 4: Predictor ABC + LightGBM 구현체

**Files:**
- Create: `RoyaleLog-worker/app/ml/predictor.py`
- Create: `RoyaleLog-worker/app/ml/lgbm_predictor.py`
- Create: `RoyaleLog-worker/tests/test_predictor.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_predictor.py
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_predictor.py -v`
Expected: FAIL

- [ ] **Step 3: predictor.py 구현**

```python
# app/ml/predictor.py
from abc import ABC, abstractmethod
import numpy as np


class Predictor(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...
```

- [ ] **Step 4: lgbm_predictor.py 구현**

```python
# app/ml/lgbm_predictor.py
import numpy as np
import lightgbm as lgb
from app.ml.predictor import Predictor


class LgbmPredictor(Predictor):
    def __init__(self, booster: lgb.Booster, version: str):
        self._booster = booster
        self._version = version

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)

    @property
    def version(self) -> str:
        return self._version

    @property
    def booster(self) -> lgb.Booster:
        return self._booster
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_predictor.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add RoyaleLog-worker/app/ml/predictor.py RoyaleLog-worker/app/ml/lgbm_predictor.py RoyaleLog-worker/tests/test_predictor.py
git commit -m "feat(worker): add Predictor ABC and LightGBM implementation"
```

---

### Task 5: ModelStore — 모드별 모델 파일 관리

**Files:**
- Create: `RoyaleLog-worker/app/ml/model_store.py`
- Create: `RoyaleLog-worker/tests/test_model_store.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_model_store.py
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_model_store.py -v`
Expected: FAIL

- [ ] **Step 3: model_store.py 구현**

```python
# app/ml/model_store.py
from __future__ import annotations

import os
import shutil

import joblib

from app.ml.lgbm_predictor import LgbmPredictor


class ModelStore:
    """모드별 로컬 .joblib 파일 관리. Phase 3에서 MLflow Registry로 교체."""

    def __init__(self, model_dir: str):
        self._dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _current_path(self, battle_type: str) -> str:
        return os.path.join(self._dir, f"matchup_{battle_type}_current.joblib")

    def _backup_path(self, battle_type: str) -> str:
        return os.path.join(self._dir, f"matchup_{battle_type}_backup.joblib")

    def save_current(self, battle_type: str, predictor: LgbmPredictor) -> None:
        current = self._current_path(battle_type)
        backup = self._backup_path(battle_type)
        if os.path.exists(current):
            shutil.copy2(current, backup)
        joblib.dump(
            {"booster": predictor.booster, "version": predictor.version},
            current,
        )

    def load_current(self, battle_type: str) -> LgbmPredictor | None:
        path = self._current_path(battle_type)
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        return LgbmPredictor(data["booster"], data["version"])

    def has_current(self, battle_type: str) -> bool:
        return os.path.exists(self._current_path(battle_type))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_model_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add RoyaleLog-worker/app/ml/model_store.py RoyaleLog-worker/tests/test_model_store.py
git commit -m "feat(worker): add per-mode ModelStore for joblib persistence"
```

---

### Task 6: Trainer — 학습 파이프라인

**Files:**
- Create: `RoyaleLog-worker/app/ml/trainer.py`
- Create: `RoyaleLog-worker/tests/test_trainer.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_trainer.py
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
        my_evo = [int(rng.integers(0, 2)) if c in builder.evo_card_to_index else 0 for c in my_cards]
        opp_evo = [int(rng.integers(0, 2)) if c in builder.evo_card_to_index else 0 for c in opp_cards]
        vec = builder.build_matchup_vector(
            battle_type=battle_type,
            deck_card_ids=my_cards, deck_evo_levels=my_evo,
            opponent_card_ids=opp_cards, opponent_evo_levels=opp_evo,
            league_number=int(rng.integers(0, 10)),
            starting_trophies=int(rng.integers(4000, 7000)),
            avg_level=float(rng.uniform(10, 14)),
            evolution_count=sum(1 for e in my_evo if e > 0),
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: trainer.py 구현**

```python
# app/ml/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from app.ml.feature_builder import FeatureBuilder
from app.ml.lgbm_predictor import LgbmPredictor
from app.ml.model_store import ModelStore


@dataclass
class TrainResult:
    saved: bool
    model_version: str | None
    train_rows: int
    val_rows: int
    val_accuracy: float
    val_logloss: float
    prev_accuracy: float | None = None
    prev_logloss: float | None = None


class Trainer:
    def __init__(
        self,
        builder: FeatureBuilder,
        store: ModelStore,
        battle_type: str,
        min_child_samples: int = 50,
        early_stopping_rounds: int = 50,
        accuracy_margin: float = 0.005,
    ):
        self._builder = builder
        self._store = store
        self._battle_type = battle_type
        self._min_child_samples = min_child_samples
        self._early_stopping_rounds = early_stopping_rounds
        self._accuracy_margin = accuracy_margin

    def train_from_arrays(
        self, X: np.ndarray, y: np.ndarray, val_start_index: int
    ) -> TrainResult:
        X_train, y_train = X[:val_start_index], y[:val_start_index]
        X_val, y_val = X[val_start_index:], y[val_start_index:]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error"],
            "verbose": -1,
            "min_child_samples": self._min_child_samples,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_pre_filter": False,
        }

        booster = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(self._early_stopping_rounds, verbose=False)],
        )

        preds = booster.predict(X_val)
        val_accuracy = accuracy_score(y_val, (preds > 0.5).astype(int))
        val_logloss = log_loss(y_val, preds)

        version = f"lgbm-{self._battle_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        prev_accuracy, prev_logloss = None, None
        saved = False

        if not self._store.has_current(self._battle_type):
            new_pred = LgbmPredictor(booster, version)
            self._store.save_current(self._battle_type, new_pred)
            saved = True
        else:
            old = self._store.load_current(self._battle_type)
            old_preds = old.predict(X_val)
            prev_accuracy = accuracy_score(y_val, (old_preds > 0.5).astype(int))
            prev_logloss = log_loss(y_val, old_preds)

            if val_accuracy >= prev_accuracy + self._accuracy_margin and val_logloss < prev_logloss:
                new_pred = LgbmPredictor(booster, version)
                self._store.save_current(self._battle_type, new_pred)
                saved = True

        return TrainResult(
            saved=saved,
            model_version=version if saved else None,
            train_rows=len(y_train),
            val_rows=len(y_val),
            val_accuracy=val_accuracy,
            val_logloss=val_logloss,
            prev_accuracy=prev_accuracy,
            prev_logloss=prev_logloss,
        )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_trainer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add RoyaleLog-worker/app/ml/trainer.py RoyaleLog-worker/tests/test_trainer.py
git commit -m "feat(worker): add Trainer with per-mode training and model replacement"
```

---

### Task 7: match_features 상대 카드 컬럼 추가 (Flyway V14)

**Files:**
- Create: `RoyaleLog-api/api/src/main/resources/db/migration/V14__match_features_opponent_cards.sql`
- Modify: `RoyaleLog-api/core/src/main/java/com/rheon/royale/batch/analyzer/AnalyzerPersistenceService.java`

> DB 스키마 변경을 API/Train 라우터보다 먼저 처리. matchup 모델은 상대 카드 필수.

- [ ] **Step 1: Flyway V14 마이그레이션 작성**

```sql
-- V14__match_features_card_levels_and_opponent.sql
-- match_features에 카드별 레벨 + 상대 덱 정보 추가
-- Worker가 matchup 모델 학습 시 카드 레벨/상대 덱 피처 직접 조회
-- 부모 테이블에 추가하면 모든 파티션에 자동 전파

-- 카드별 레벨 (card_ids와 동일 순서, 0~14)
ALTER TABLE match_features ADD COLUMN IF NOT EXISTS card_levels              SMALLINT[];

-- 상대 덱 카드 정보
ALTER TABLE match_features ADD COLUMN IF NOT EXISTS opponent_card_ids        BIGINT[];
ALTER TABLE match_features ADD COLUMN IF NOT EXISTS opponent_card_levels     SMALLINT[];
ALTER TABLE match_features ADD COLUMN IF NOT EXISTS opponent_card_evo_levels SMALLINT[];
```

- [ ] **Step 2: AnalyzerPersistenceService INSERT SQL 수정**

`batchInsertMatchFeatures` 메서드를 아래와 같이 변경:

```java
public void batchInsertMatchFeatures(List<? extends AnalyzedBattle> items) {
    jdbcTemplate.batchUpdate("""
            INSERT INTO match_features
                (battle_id, deck_hash, refined_deck_hash,
                 opponent_hash, refined_opponent_hash,
                 battle_type, battle_date, avg_level, evolution_count, result,
                 league_number, starting_trophies,
                 card_ids, card_evo_levels,
                 opponent_card_ids, opponent_card_evo_levels,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
            ON CONFLICT (battle_id, battle_date)
            DO UPDATE SET
                deck_hash             = EXCLUDED.deck_hash,
                refined_deck_hash     = EXCLUDED.refined_deck_hash,
                opponent_hash         = EXCLUDED.opponent_hash,
                refined_opponent_hash = EXCLUDED.refined_opponent_hash,
                avg_level             = EXCLUDED.avg_level,
                evolution_count       = EXCLUDED.evolution_count,
                result                = EXCLUDED.result,
                league_number         = EXCLUDED.league_number,
                starting_trophies     = EXCLUDED.starting_trophies,
                card_ids              = EXCLUDED.card_ids,
                card_evo_levels       = EXCLUDED.card_evo_levels,
                opponent_card_ids     = EXCLUDED.opponent_card_ids,
                opponent_card_evo_levels = EXCLUDED.opponent_card_evo_levels,
                updated_at            = NOW()
            WHERE match_features.updated_at < EXCLUDED.updated_at
            """,
            items, items.size(),
            (ps, b) -> {
                ps.setString(1, b.battleId());
                ps.setString(2, b.deckHash());
                ps.setString(3, b.refinedDeckHash());
                ps.setString(4, b.opponentHash());
                ps.setString(5, b.refinedOpponentHash());
                ps.setString(6, b.battleType());
                ps.setDate(7, Date.valueOf(b.battleDate()));
                ps.setBigDecimal(8, b.avgLevel());
                ps.setInt(9, b.evolutionCount());
                ps.setInt(10, b.result());
                if (b.leagueNumber() != null) ps.setInt(11, b.leagueNumber());
                else ps.setNull(11, java.sql.Types.INTEGER);
                if (b.startingTrophies() != null) ps.setInt(12, b.startingTrophies());
                else ps.setNull(12, java.sql.Types.INTEGER);
                if (b.cardIds() != null)
                    ps.setArray(13, ps.getConnection().createArrayOf("bigint", b.cardIds()));
                else ps.setNull(13, java.sql.Types.ARRAY);
                if (b.evoLevels() != null)
                    ps.setArray(14, ps.getConnection().createArrayOf("smallint", b.evoLevels()));
                else ps.setNull(14, java.sql.Types.ARRAY);
                if (b.opponentCardIds() != null)
                    ps.setArray(15, ps.getConnection().createArrayOf("bigint", b.opponentCardIds()));
                else ps.setNull(15, java.sql.Types.ARRAY);
                if (b.opponentEvoLevels() != null)
                    ps.setArray(16, ps.getConnection().createArrayOf("smallint", b.opponentEvoLevels()));
                else ps.setNull(16, java.sql.Types.ARRAY);
            });
}
```

- [ ] **Step 3: Commit**

```bash
git add RoyaleLog-api/api/src/main/resources/db/migration/V14__match_features_opponent_cards.sql
git add RoyaleLog-api/core/src/main/java/com/rheon/royale/batch/analyzer/AnalyzerPersistenceService.java
git commit -m "feat: add opponent card columns to match_features (V14)"
```

---

### Task 8: API 라우터 (health, predict, train)

**Files:**
- Create: `RoyaleLog-worker/app/api/__init__.py`
- Create: `RoyaleLog-worker/app/api/health.py`
- Create: `RoyaleLog-worker/app/api/predict.py`
- Create: `RoyaleLog-worker/app/api/train.py`

- [ ] **Step 1: health.py 작성**

```python
# app/api/health.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: predict.py 작성**

```python
# app/api/predict.py
import asyncio
import logging

from fastapi import APIRouter, Request, HTTPException

from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict")
logger = logging.getLogger(__name__)


@router.post("/matchup", response_model=PredictResponse)
async def predict_matchup(req: PredictRequest, request: Request):
    lock: asyncio.Lock = request.app.state.model_lock

    async with lock:
        predictor = request.app.state.predictors.get(req.battle_type)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model loaded for battle_type={req.battle_type}",
        )

    builder = request.app.state.feature_builder
    vec = builder.build_matchup_vector(
        battle_type=req.battle_type,
        deck_card_ids=req.deck_card_ids,
        deck_evo_levels=req.deck_evo_levels,
        opponent_card_ids=req.opponent_card_ids,
        opponent_evo_levels=req.opponent_evo_levels,
        league_number=req.league_number,
        starting_trophies=req.starting_trophies,
        avg_level=0.0,  # 실시간 요청에서는 불명, 학습 시에도 상대 avg_level 0
        evolution_count=sum(1 for e in req.deck_evo_levels if e > 0),
    )

    import numpy as np
    probs = predictor.predict(vec.reshape(1, -1))

    logger.info("predict request_id=%s type=%s prob=%.4f",
                req.request_id, req.battle_type, probs[0])

    return PredictResponse(
        request_id=req.request_id,
        win_probability=round(float(probs[0]), 4),
        model_version=predictor.version,
    )
```

- [ ] **Step 3: train.py 작성**

```python
# app/api/train.py
import asyncio
import json
import logging
import os
from datetime import datetime, date, timezone, timedelta

import mlflow
import numpy as np
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Query
from sqlalchemy import text

from app.core.database import get_session
from app.ml.feature_builder import FeatureBuilder
from app.ml.trainer import Trainer
from app.schemas.train import TrainResponse, TrainStatusResponse

router = APIRouter(prefix="/train")
logger = logging.getLogger(__name__)


def _read_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"status": "idle"}
    with open(path, "r") as f:
        return json.load(f)


def _write_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, default=str)


@router.get("/status", response_model=TrainStatusResponse)
def train_status(request: Request):
    path = request.app.state.settings.state_file_path
    return TrainStatusResponse(**_read_state(path))


@router.post("", response_model=TrainResponse, status_code=202)
async def trigger_train(
    request: Request,
    background_tasks: BackgroundTasks,
    reset_window: bool = Query(False),
):
    settings = request.app.state.settings
    state_path = settings.state_file_path
    state = _read_state(state_path)

    if reset_window:
        state["patch_date"] = str(date.today())
        state["status"] = "idle"
        _write_state(state_path, state)
        return TrainResponse(status="accepted", message="Patch date recorded. Training paused.")

    patch_date_str = state.get("patch_date")
    if patch_date_str:
        patch_date = date.fromisoformat(patch_date_str)
        if date.today() < patch_date + timedelta(days=settings.min_post_patch_days):
            return TrainResponse(
                status="accepted",
                message=f"Warming up until {patch_date + timedelta(days=settings.min_post_patch_days)}. Skipped.",
            )

    if state.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    _write_state(state_path, {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    background_tasks.add_task(_run_training, request.app)
    return TrainResponse(status="accepted", message="Training started")


def _run_training(app):
    """동기 함수 — FastAPI가 threadpool에서 실행. 이벤트 루프 블로킹 방지."""
    settings = app.state.settings
    state_path = settings.state_file_path

    try:
        with get_session() as session:
            builder = FeatureBuilder.from_db(session)
            app.state.feature_builder = builder

        all_results = {}

        for battle_type in settings.train_battle_types:
            logger.info("Training %s model...", battle_type)
            result = _train_single_mode(app, builder, battle_type, settings)
            all_results[battle_type] = result

            if result and result.saved:
                new_predictor = app.state.model_store.load_current(battle_type)
                loop = asyncio.get_event_loop()
                future = asyncio.run_coroutine_threadsafe(
                    _swap_model(app, battle_type, new_predictor), loop
                )
                future.result(timeout=10)

        # 전체 결과에서 첫 번째 성공한 모드의 메트릭 기록
        best = next(
            (r for r in all_results.values() if r is not None and r.saved),
            next((r for r in all_results.values() if r is not None), None),
        )

        _write_state(state_path, {
            "status": "success",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "model_version": best.model_version if best else None,
            "train_rows": best.train_rows if best else 0,
            "val_rows": best.val_rows if best else 0,
            "val_accuracy": round(best.val_accuracy, 4) if best else None,
            "val_logloss": round(best.val_logloss, 4) if best else None,
        })

    except Exception as e:
        logger.exception("Training failed")
        _write_state(state_path, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })


def _train_single_mode(app, builder, battle_type, settings):
    """단일 모드 학습. 데이터 부족 시 None 반환."""
    with get_session() as session:
        cutoff = date.today() - timedelta(days=settings.window_days)
        result = session.execute(
            text("""
                SELECT card_ids, card_evo_levels,
                       opponent_card_ids, opponent_card_evo_levels,
                       battle_type, battle_date,
                       avg_level, evolution_count,
                       league_number, starting_trophies, result
                FROM match_features
                WHERE battle_type = :battle_type
                  AND battle_date >= :cutoff
                  AND card_ids IS NOT NULL
                  AND opponent_card_ids IS NOT NULL
            """),
            {"battle_type": battle_type, "cutoff": cutoff},
        )
        rows = result.fetchall()

    if len(rows) < settings.min_train_rows:
        logger.warning("%s: insufficient data (%d < %d), skipping",
                       battle_type, len(rows), settings.min_train_rows)
        return None

    X_list, y_list, dates = [], [], []
    for row in rows:
        card_ids = list(row[0]) if row[0] else []
        evo_levels = list(row[1]) if row[1] else [0] * len(card_ids)
        opp_card_ids = list(row[2]) if row[2] else []
        opp_evo_levels = list(row[3]) if row[3] else [0] * len(opp_card_ids)
        avg_level = float(row[6]) if row[6] else 0.0
        evo_count = int(row[7]) if row[7] else 0
        league = int(row[8]) if row[8] is not None else None
        trophies = int(row[9]) if row[9] is not None else None

        vec = builder.build_matchup_vector(
            battle_type=battle_type,
            deck_card_ids=card_ids, deck_evo_levels=evo_levels,
            opponent_card_ids=opp_card_ids, opponent_evo_levels=opp_evo_levels,
            league_number=league, starting_trophies=trophies,
            avg_level=avg_level, evolution_count=evo_count,
        )
        X_list.append(vec)
        y_list.append(int(row[10]))
        dates.append(row[5])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # time-series split
    val_cutoff = date.today() - timedelta(days=settings.val_days)
    val_mask = np.array([d >= val_cutoff for d in dates])
    train_mask = ~val_mask
    X_sorted = np.concatenate([X[train_mask], X[val_mask]])
    y_sorted = np.concatenate([y[train_mask], y[val_mask]])
    val_start = int(train_mask.sum())

    trainer = Trainer(
        builder=builder, store=app.state.model_store, battle_type=battle_type,
        min_child_samples=settings.min_child_samples,
        early_stopping_rounds=settings.early_stopping_rounds,
        accuracy_margin=settings.accuracy_margin,
    )
    train_result = trainer.train_from_arrays(X_sorted, y_sorted, val_start)

    # MLflow 기록
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"royalelog-{battle_type}")
    with mlflow.start_run():
        mlflow.log_params({
            "battle_type": battle_type,
            "window_days": settings.window_days,
            "val_days": settings.val_days,
            "min_child_samples": settings.min_child_samples,
            "feature_count": len(builder.feature_names(battle_type)),
        })
        mlflow.log_metrics({
            "val_accuracy": train_result.val_accuracy,
            "val_logloss": train_result.val_logloss,
            "train_rows": train_result.train_rows,
            "val_rows": train_result.val_rows,
        })
        if train_result.prev_accuracy is not None:
            mlflow.log_metrics({
                "prev_accuracy": train_result.prev_accuracy,
                "prev_logloss": train_result.prev_logloss,
            })

    logger.info("%s: saved=%s acc=%.4f logloss=%.4f",
                battle_type, train_result.saved,
                train_result.val_accuracy, train_result.val_logloss)

    return train_result


async def _swap_model(app, battle_type, new_predictor):
    """이벤트 루프에서 실행 — asyncio.Lock으로 안전하게 교체."""
    async with app.state.model_lock:
        app.state.predictors[battle_type] = new_predictor
```

- [ ] **Step 4: `app/api/__init__.py` 빈 파일 생성**

- [ ] **Step 5: Commit**

```bash
git add RoyaleLog-worker/app/api/
git commit -m "feat(worker): add health, predict, train API routers with per-mode support"
```

---

### Task 9: main.py — FastAPI 앱 조립 + Lifespan

**Files:**
- Create: `RoyaleLog-worker/app/main.py`
- Create: `RoyaleLog-worker/tests/conftest.py`
- Create: `RoyaleLog-worker/tests/test_api_health.py`
- Create: `RoyaleLog-worker/tests/test_api_predict.py`
- Create: `RoyaleLog-worker/tests/test_api_train.py`

- [ ] **Step 1: main.py 작성**

```python
# app/main.py
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings
from app.core.database import init_db, get_session
from app.ml.feature_builder import FeatureBuilder
from app.ml.model_store import ModelStore
from app.api import health, predict, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings

    init_db(settings.db_url)

    with get_session() as session:
        app.state.feature_builder = FeatureBuilder.from_db(session)

    store = ModelStore(settings.model_dir)
    app.state.model_store = store

    # 모드별 predictor dict
    predictors: dict = {}
    for bt in settings.train_battle_types:
        loaded = store.load_current(bt)
        if loaded:
            predictors[bt] = loaded
            logger.info("Loaded model: %s → %s", bt, loaded.version)
    app.state.predictors = predictors
    app.state.model_lock = asyncio.Lock()

    # training_state.json 복구
    state_path = settings.state_file_path
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        if state.get("status") == "running":
            state["status"] = "failed"
            state["error"] = "Worker restarted during training"
            with open(state_path, "w") as f:
                json.dump(state, f, default=str)
            logger.warning("Recovered stale training state: running → failed")

    logger.info("Worker started. Models: %s",
                {k: v.version for k, v in predictors.items()} or "None")

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="RoyaleLog Worker", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(train.router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.worker_port, reload=True)
```

- [ ] **Step 2: conftest.py 작성**

```python
# tests/conftest.py
import os

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5434")
os.environ.setdefault("DB_NAME", "royalelog")
os.environ.setdefault("DB_USER", "royale")
os.environ.setdefault("DB_PASSWORD", "test")
```

- [ ] **Step 3: test_api_health.py 작성**

```python
# tests/test_api_health.py
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
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
```

- [ ] **Step 4: test_api_predict.py 작성**

```python
# tests/test_api_predict.py
import asyncio
import numpy as np
import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from app.ml.feature_builder import FeatureBuilder


def _make_test_app(with_model=True):
    from app.main import create_app

    @asynccontextmanager
    async def fake_lifespan(app):
        yield

    test_app = create_app()
    test_app.router.lifespan_context = fake_lifespan

    card_rows = [
        (26000000 + i, "NORMAL", None) for i in range(120)
    ] + [(26000000, "EVOLUTION", 1), (159000000, "NORMAL", None)]
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


@pytest.mark.asyncio
async def test_predict_matchup_returns_probability():
    test_app = _make_test_app()
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.post("/predict/matchup", json={
            "request_id": "test-uuid-1234",
            "deck_card_ids": [26000000 + i for i in range(8)],
            "deck_evo_levels": [1, 0, 0, 0, 0, 0, 0, 0],
            "opponent_card_ids": [26000010 + i for i in range(8)],
            "opponent_evo_levels": [0] * 8,
            "battle_type": "pathOfLegend",
            "league_number": 7,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "test-uuid-1234"
        assert 0.0 <= body["win_probability"] <= 1.0
        assert body["model_version"] == "test-v1"


@pytest.mark.asyncio
async def test_predict_503_when_no_model():
    test_app = _make_test_app(with_model=False)
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.post("/predict/matchup", json={
            "request_id": "test-uuid",
            "deck_card_ids": [26000000 + i for i in range(8)],
            "deck_evo_levels": [0] * 8,
            "opponent_card_ids": [26000010 + i for i in range(8)],
            "opponent_evo_levels": [0] * 8,
        })
        assert resp.status_code == 503
```

- [ ] **Step 5: test_api_train.py 작성**

```python
# tests/test_api_train.py
import asyncio
import json
import os
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport
from app.core.config import Settings


def _make_test_app(tmp_path):
    from app.main import create_app

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
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/train/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"


@pytest.mark.asyncio
async def test_train_reset_window(tmp_path):
    test_app = _make_test_app(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
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

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.post("/train")
        assert resp.status_code == 409
```

- [ ] **Step 6: 테스트 실행**

Run: `cd RoyaleLog-worker && python -m pytest tests/test_api_health.py tests/test_api_predict.py tests/test_api_train.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add RoyaleLog-worker/app/main.py RoyaleLog-worker/tests/
git commit -m "feat(worker): FastAPI app with lifespan, per-mode predictors, API tests"
```

---

### Task 10: Dockerfile + docker-compose 업데이트

**Files:**
- Create: `RoyaleLog-worker/Dockerfile`
- Modify: `RoyaleLog-api/docker-compose.yml`

- [ ] **Step 1: Dockerfile 작성**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models

EXPOSE 8082

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
```

- [ ] **Step 2: docker-compose.yml에 worker + mlflow 추가**

기존 `volumes:` 섹션 앞에 서비스 추가:

```yaml
  # ── ML Worker (port 8082) ────────────────────────────────────────
  worker:
    build:
      context: ../RoyaleLog-worker
      dockerfile: Dockerfile
    container_name: royalelog-worker
    env_file: .env
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8082:8082"
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_started

  # ── MLflow Tracking Server ───────────────────────────────────────
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: royalelog-mlflow
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri postgresql://${DB_USER:-royale}:${DB_PASSWORD}@postgres:5432/${DB_NAME:-royalelog}
      --default-artifact-root /mlruns
      --host 0.0.0.0
    volumes:
      - mlflow_artifacts:/mlruns
    depends_on:
      postgres:
        condition: service_healthy
```

volumes 섹션:

```yaml
volumes:
  postgres-data:
  mlflow_artifacts:
```

- [ ] **Step 3: docker-compose 검증**

Run: `cd RoyaleLog-api && docker compose config`
Expected: YAML 파싱 성공

- [ ] **Step 4: Commit**

```bash
git add RoyaleLog-worker/Dockerfile RoyaleLog-api/docker-compose.yml
git commit -m "feat: add Worker Dockerfile and docker-compose mlflow/worker services"
```

---

### Task 11: 통합 테스트

- [ ] **Step 1: docker-compose up**

Run: `cd RoyaleLog-api && docker compose up -d --build worker mlflow`
Expected: 컨테이너 정상 기동

- [ ] **Step 2: 헬스체크**

Run: `curl http://localhost:8082/health`
Expected: `{"status":"ok"}`

- [ ] **Step 3: MLflow UI 확인**

Run: `curl -s http://localhost:5000/api/2.0/mlflow/experiments/list | head -c 200`
Expected: JSON 응답

- [ ] **Step 4: train/status 확인**

Run: `curl http://localhost:8082/train/status`
Expected: `{"status":"idle",...}`

- [ ] **Step 5: predict → 503 (모델 없음)**

Run: `curl -X POST http://localhost:8082/predict/matchup -H "Content-Type: application/json" -d '{"request_id":"test","deck_card_ids":[26000000,26000001,26000002,26000003,26000004,26000005,26000006,159000000],"deck_evo_levels":[0,0,0,0,0,0,0,0],"opponent_card_ids":[26000010,26000011,26000012,26000013,26000014,26000015,26000016,159000000],"opponent_evo_levels":[0,0,0,0,0,0,0,0],"battle_type":"pathOfLegend"}'`
Expected: 503

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(worker): complete RoyaleLog Worker FastAPI ML server (Phase 2)"
```
