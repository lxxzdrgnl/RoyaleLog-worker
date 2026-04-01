# RoyaleLog-worker

![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-02569B?style=flat-square&logo=microsoft&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-K3s-2496ED?style=flat-square&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%202-blue?style=flat-square)

---

FastAPI 기반 ML 서버. 모드별(pathOfLegend, ladder) LightGBM 덱 매치업 승률 예측 모델을 서빙하고, Airflow 트리거 기반 재학습 파이프라인을 제공한다.

### 아키텍처 위치

```
Vue.js → Spring Boot (BFF)
           ├─ POST /predict/matchup → Worker (500ms 타임아웃)
           │    정상: { win_probability, model_version }
           │    실패: Spring Boot가 stats_daily 직접 조회 → Fallback
           └─ Spring Boot → Vue.js 최종 응답 조립
```

> **원칙**: Worker는 ML 예측만 반환한다. Fallback(`stats_fallback`, `prior_fallback`)은 Spring Boot 책임.

### 모드별 모델 전략

```
pathOfLegend  → 레벨 고정, 순수 덱 상성 예측 (evolution_count, league_number)
ladder        → 카드별 레벨 차이가 승패에 직접 영향 (evolution_count, starting_trophies)
특별모드      → 데이터 부족 → 학습 스킵 → Spring Boot stats_fallback 위임
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.11+ |
| Framework | FastAPI + uvicorn |
| ML | LightGBM (Predictor ABC → NN 교체 가능 구조) |
| DB | SQLAlchemy + PostgreSQL (RoyaleLog-api와 동일 DB) |
| MLOps | MLflow Tracking Server (실험 기록, Phase 3에서 Registry 확장) |
| Serialization | joblib (.joblib 모델 파일) |

---

## 실행 방법

### 로컬

> **사전 조건**: PostgreSQL 16 실행 중 + `match_features` 테이블에 데이터 존재

```bash
# 1. 환경변수 파일 준비
cp .env.example .env
# .env 열어서 DB_PASSWORD 채우기

# 2. 가상환경 + 의존성 설치
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 실행
python -m app.main
# → http://localhost:8082
```

### Docker Compose (권장)

```bash
# RoyaleLog-api/ 디렉토리에서 실행
cd ../RoyaleLog-api

# 환경변수 준비
cp .env.example .env

# Worker + MLflow 실행 (PostgreSQL, Redis는 기존 서비스)
docker compose up --build worker mlflow -d

# 로그 확인
docker compose logs -f worker

# 종료
docker compose down
```

---

## 환경변수

`.env.example`을 복사해 `.env`로 사용합니다. `.env`는 절대 커밋하지 않습니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DB_HOST` | `localhost` | PostgreSQL 호스트 |
| `DB_PORT` | `5434` | PostgreSQL 포트 |
| `DB_NAME` | `royalelog` | 데이터베이스명 |
| `DB_USER` | `royale` | DB 유저 |
| `DB_PASSWORD` | — | **필수** DB 비밀번호 |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow Tracking Server 주소 |
| `MODEL_DIR` | `./models` | 학습된 모델 파일 저장 경로 |
| `STATE_FILE_PATH` | `./training_state.json` | 학습 상태 영속화 파일 경로 |
| `WINDOW_DAYS` | `3` | 학습 데이터 Sliding Window (일) |
| `VAL_DAYS` | `1` | 검증 데이터 기간 (일) |
| `MIN_CHILD_SAMPLES` | `50` | LightGBM regularization |
| `MIN_TRAIN_ROWS` | `10000` | 최소 학습 데이터 행 수 (미달 시 학습 스킵) |
| `MIN_POST_PATCH_DAYS` | `3` | 패치 후 워밍업 대기 기간 (일) |
| `EARLY_STOPPING_ROUNDS` | `50` | LightGBM early stopping |
| `ACCURACY_MARGIN` | `0.005` | 모델 교체 시 요구 정확도 마진 |
| `TRAIN_BATTLE_TYPES` | `pathOfLegend,ladder` | 학습 대상 배틀 모드 |
| `WORKER_PORT` | `8082` | 서버 포트 |

---

## 배포 주소

| 항목 | URL |
|------|-----|
| Base URL | `http://localhost:8082` |
| Health Check | `http://localhost:8082/health` |
| MLflow UI | `http://localhost:5000` |

---

## 패키지 구조

```
app/
├── main.py                     # FastAPI 앱, lifespan, router 등록
│
├── api/                        # HTTP 엔드포인트
│   ├── health.py               # GET /health
│   ├── predict.py              # POST /predict/matchup
│   └── train.py                # POST /train, GET /train/status
│
├── core/                       # 설정 및 인프라
│   ├── config.py               # Pydantic Settings (환경변수)
│   └── database.py             # SQLAlchemy engine + context manager
│
├── ml/                         # 머신러닝 레이어
│   ├── predictor.py            # Predictor ABC (인터페이스)
│   ├── lgbm_predictor.py       # LightGBM 구현체 (NN 교체 시 여기만 교환)
│   ├── feature_builder.py      # cards 테이블 기반 동적 피처 벡터 변환
│   ├── model_store.py          # 모드별 모델 파일 로드/저장 (Phase 3: MLflow Registry)
│   └── trainer.py              # 학습 파이프라인 오케스트레이션
│
└── schemas/                    # Pydantic DTO
    ├── predict.py              # PredictRequest / PredictResponse
    └── train.py                # TrainResponse / TrainStatusResponse
```

---

## API 명세

### 덱 매치업 승률 예측

```
POST /predict/matchup
```

- `battle_type`으로 모드별 모델 자동 선택 (pathOfLegend / ladder)
- 해당 모드 모델 없으면 503 → Spring Boot Fallback
- `request_id`: Spring Boot가 생성한 UUID, 분산 추적용 (양쪽 로그 매칭)

```json
// Request
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "deck_card_ids": [26000000, 26000001, 26000002, 26000003, 26000004, 26000005, 26000006, 159000000],
  "deck_card_levels": [14, 13, 14, 12, 13, 14, 11, 14],
  "deck_evo_levels": [1, 0, 0, 0, 0, 0, 0, 0],
  "opponent_card_ids": [26000010, 26000011, 26000012, 26000013, 26000014, 26000015, 26000016, 159000000],
  "opponent_card_levels": [13, 14, 12, 13, 14, 11, 14, 13],
  "opponent_evo_levels": [0, 0, 0, 0, 0, 0, 0, 0],
  "battle_type": "pathOfLegend",
  "league_number": 7,
  "starting_trophies": null
}

// Response
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "win_probability": 0.62,
  "model_version": "lgbm-pathOfLegend-20260401030000"
}
```

### 모델 재학습 트리거

```
POST /train
POST /train?reset_window=true    # 패치 대응: 워밍업 기간 시작
```

- Airflow 트리거 용도 (수동 호출 가능)
- 즉시 `202 Accepted` 반환, 실제 학습은 threadpool에서 백그라운드 실행
- `reset_window=true`: 패치일 기록, `MIN_POST_PATCH_DAYS` 이후에만 학습 실행
- 학습 중 중복 호출 시 `409 Conflict`
- 모드별 순차 학습 (pathOfLegend → ladder)

### 학습 상태 조회

```
GET /train/status
```

- Airflow 폴링 계약: `status == "success"` 또는 `"failed"` 까지 30초 간격 폴링

```json
{
  "status": "success",
  "started_at": "2026-04-01T03:00:00Z",
  "finished_at": "2026-04-01T03:04:30Z",
  "model_version": "lgbm-pathOfLegend-20260401030000",
  "train_rows": 42000,
  "val_rows": 14000,
  "val_accuracy": 0.762,
  "val_logloss": 0.481,
  "error": null
}
```

---

## 피처 엔지니어링

### 피처 벡터 구조 (내 덱 + 상대 덱 각각)

| 구역 | 크기 | 값 | 설명 |
|------|------|-----|------|
| 카드 레벨 벡터 | 동적 (cards 테이블 기준) | 0~14 (미사용=0, 사용=레벨) | `card_knight = 14` |
| 진화 스위치 벡터 | 동적 (진화 가능 카드) | 0 or 1 | `evo_knight = 1` |
| 수치 피처 | 2칸 (모드별) | 정수 | PoL: evo_count+league / ladder: evo_count+trophies |

**카드 레벨 인코딩**: 0/1이 아닌 실제 레벨값(0~14)을 사용. `avg_level`은 카드별 레벨의 합산이므로 중복 → 제거.

**벡터 크기 동적 결정**: `feature_builder.py`가 앱 시작 시 `cards` 테이블 조회. 신규 카드 패치 시 자동 갱신.

#### LightGBM의 학습 방식

```
1단계: "card_knight 칸이 14네? 14레벨 기사 사용 — 맷집 높음"
2단계: "evo_knight 칸이 1? 진화 기사 — 보호막 패시브 발동"
최종:  "14레벨 진화 기사 → 승률 대폭 상승" (Feature Interaction)
```

트리 기반 모델은 여러 피처를 동시에 엮어 스무고개식으로 판단한다.

---

## 학습 파이프라인

```
POST /train (Airflow 호출)
  │
  ├── 패치 워밍업 체크 (patch_date + MIN_POST_PATCH_DAYS 미경과 → 스킵)
  ├── 중복 학습 방지 (training_state.json == "running" → 409)
  │
  └── 모드별 순차 학습 (threadpool 실행, 이벤트 루프 블로킹 방지)
        │
        ├── 1. DB 조회: match_features WHERE battle_date >= today - 3일
        ├── 2. MIN_TRAIN_ROWS(10000) 미달 → 해당 모드 스킵
        ├── 3. 피처 빌드: cards 테이블 동적 조회 → 카드 레벨 벡터 + 진화 스위치
        ├── 4. train/val split: train=[today-3, today-2], val=[today-1]
        ├── 5. LightGBM 학습: early_stopping_rounds=50
        ├── 6. 모델 교체 조건:
        │       최초 학습 → 무조건 저장
        │       있음 → accuracy >= 기존 + 0.5% AND logloss < 기존
        ├── 7. MLflow 실험 기록
        └── 8. 모델 핫스왑 (asyncio.Lock → 서빙 무중단)
```

---

## 모델 파일 관리

```
models/
├── matchup_pathOfLegend_current.joblib   ← 서빙 중
├── matchup_pathOfLegend_backup.joblib    ← 롤백용
├── matchup_ladder_current.joblib
└── matchup_ladder_backup.joblib
```

| Phase | 관리 방식 |
|-------|-----------|
| Phase 2 (현재) | `model_store.py` → 로컬 .joblib 파일 |
| Phase 3 | `model_store.py` → `mlflow.pyfunc.load_model()` 교체 (단일 확장 포인트) |

---

## 패치 대응 전략

```
패치 당일
  └─ POST /train?reset_window=true → patch_date 기록, 구 모델 계속 서빙

패치 후 3일 (워밍업)
  └─ Airflow 정기 트리거 → patch_date + MIN_POST_PATCH_DAYS 미경과 → 스킵
  └─ 구 모델 서빙 유지 (약간 구식이지만 패치 직후 노이즈 데이터보다 나음)

3일 후
  └─ 정상 학습 재개 → 모델 교체 조건 충족 시 교체
```

Phase 3: 카드 win-rate 분포 코사인 유사도로 패치 자동 감지 예정.

---

## Fallback 체인

```
Spring Boot (BFF)
  │
  ├── [정상] Worker /predict/matchup → ML 추론 결과 반환
  │         source = "model"
  │
  ├── [Fallback 1] Worker 타임아웃(500ms) or 5xx
  │         → stats_decks_daily_current에서 덱 매치업 통계 조회
  │         source = "stats_fallback"
  │
  └── [Fallback 2] stats_daily에도 해당 덱 없음
            → 사전 확률 0.50 반환
            source = "prior_fallback"
```

> **원칙**: FastAPI는 `stats_fallback`을 모른다. Fallback 로직은 전부 Spring Boot 책임.

---

## 인프라

### docker-compose (RoyaleLog-api/docker-compose.yml에 추가)

| 서비스 | 이미지 | 포트 | 설명 |
|--------|--------|------|------|
| `worker` | 빌드 (../RoyaleLog-worker) | 8082 | ML 서버 |
| `mlflow` | ghcr.io/mlflow/mlflow | 5000 | Tracking Server (PostgreSQL 백엔드 스토어) |

- MLflow 테이블은 최초 실행 시 자동 생성 (Flyway 불필요)
- MLflow artifact는 named volume (`mlflow_artifacts`) 마운트 → 컨테이너 재시작에도 보존

---

## 성능 목표

| 지표 | 목표 |
|------|------|
| 추론 latency | < 50ms |
| 모델 정확도 | val accuracy >= 75% |
| 재학습 소요 시간 | < 5분 |
| Spring Boot 타임아웃 | 500ms |

---

## 미결 사항 (별도 설계)

- **덱 추천 기능** (`GET /api/v1/recommend/{playerTag}`): Spring Boot + Worker 협업 구조
- **Phase 3 자동 패치 감지**: 카드 win-rate 코사인 유사도 기반
- **Phase 3 Shadow Deployment**: 새 모델 트래픽 비교 후 교체
- **Phase 3 MLflow Model Registry**: `model_store.py` → `mlflow.pyfunc.load_model()` 교체
