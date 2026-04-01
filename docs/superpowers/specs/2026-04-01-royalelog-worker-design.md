# RoyaleLog Worker — FastAPI ML 서버 설계 스펙

**날짜**: 2026-04-01
**Phase**: 2
**상태**: 승인됨

---

## 1. 목표

`RoyaleLog-worker/` 에 FastAPI 기반 ML 서버를 구축한다.

- **메인**: 덱 vs 덱 매치업 승률 예측 (`/predict/matchup`)
- **학습**: Airflow 트리거 기반 LightGBM 재학습 (`/train`)
- **Fallback**: Worker 다운 or 타임아웃 시 Spring Boot가 `stats_daily` 기반 응답 (Worker 책임 아님)

---

## 2. 아키텍처

### 전체 흐름

```
Vue.js → Spring Boot (BFF)
           ├─ POST /predict/matchup → Worker (500ms 타임아웃)
           │    정상: { win_probability, model_version }
           │    실패: Spring Boot가 stats_daily 직접 조회 → { source: "stats_fallback" }
           └─ Spring Boot → Vue.js 최종 응답 조립
```

### Spring Boot 최종 응답 구조

```json
// 정상
{ "win_probability": 0.62, "source": "model", "model_version": "lgbm-v3" }

// Fallback 1: Worker 타임아웃/에러 → stats_daily 조회 성공
{ "win_probability": 0.51, "source": "stats_fallback", "model_version": null }

// Fallback 2: stats_daily에도 해당 덱 없음 → 사전 확률 반환
{ "win_probability": 0.50, "source": "prior_fallback", "model_version": null }
```

### Worker 응답 구조 (FastAPI만의 응답)

```json
{
  "request_id": "550e8400-e29b-41d4-a716",
  "win_probability": 0.62,
  "model_version": "lgbm-v3"
}
```

> **원칙**: FastAPI는 `stats_fallback`을 모른다. Fallback은 Spring Boot 책임.

### request_id 분산 추적

Spring Boot가 UUID 생성 → Request Body에 포함 → FastAPI가 Response에 에코.
양쪽 로그에 동일 `request_id`로 장애 추적 가능. Phase 3에서 OpenTelemetry trace ID로 승격.

---

## 3. 폴더 구조

```
RoyaleLog-worker/
├── app/
│   ├── main.py                  # FastAPI 앱 진입점
│   ├── api/
│   │   ├── predict.py           # POST /predict/matchup
│   │   ├── train.py             # POST /train, GET /train/status
│   │   └── health.py            # GET /health
│   ├── core/
│   │   ├── config.py            # 환경변수 (Pydantic Settings)
│   │   └── database.py          # SQLAlchemy 엔진 + 세션
│   ├── ml/
│   │   ├── predictor.py         # Predictor ABC (인터페이스)
│   │   ├── lgbm_predictor.py    # LightGBM 구현체
│   │   ├── trainer.py           # 학습 파이프라인
│   │   ├── feature_builder.py   # match_features → ML 입력 변환
│   │   └── model_store.py       # 모델 로드/저장 (MLflow 확장 포인트)
│   └── schemas/
│       ├── predict.py           # PredictRequest / PredictResponse
│       └── train.py             # TrainRequest / TrainResponse
├── models/                      # 학습된 모델 파일 (.joblib)
├── tests/
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 4. API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/predict/matchup` | 덱 vs 덱 승률 예측 (battle_type으로 모델 자동 선택) |
| POST | `/train` | 재학습 트리거 (202 즉시 반환, 모드별 순차 학습) |
| GET | `/train/status` | 학습 진행 상태 조회 (Airflow 폴링용) |
| GET | `/health` | 헬스체크 |

### GET /train/status

```json
// Response
{
  "status": "idle" | "running" | "success" | "failed",
  "started_at": "2026-04-01T03:00:00Z",   // null if idle
  "finished_at": "2026-04-01T03:04:30Z",  // null if running
  "model_version": "lgbm-v4",             // null if failed or idle
  "train_rows": 42000,
  "val_accuracy": 0.762,
  "val_logloss": 0.481,
  "error": null                            // 실패 시 에러 메시지
}
```

Airflow 폴링 계약: `status == "success"` 또는 `"failed"` 가 될 때까지 30초 간격으로 폴링.

### POST /predict/matchup

```json
// Request
{
  "request_id": "550e8400-e29b-41d4-a716",
  "deck_card_ids": [26000000, 26000001, ...],      // 8~9장 (타워 카드 포함)
  "deck_card_levels": [14, 13, 14, 12, 13, 14, 11, 14, 13],
  "deck_evo_levels": [0, 1, 0, 0, 0, 0, 0, 0, 0],
  "opponent_card_ids": [26000005, ...],
  "opponent_card_levels": [13, 14, 12, 13, 14, 11, 14, 13, 12],
  "opponent_evo_levels": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "battle_type": "pathOfLegend",
  "league_number": 7,           // pathOfLegend 전용 (0~9), 없으면 null
  "starting_trophies": null     // PvP 전용, 없으면 null
}

// Response
{
  "request_id": "550e8400-e29b-41d4-a716",
  "win_probability": 0.62,
  "model_version": "lgbm-v3"
}
```

---

## 5. 피처 엔지니어링

### 피처 벡터 구조 (내 덱 + 상대 덱 각각)

| 구역 | 크기 | 값 | 설명 |
|------|------|-----|------|
| 카드 레벨 벡터 | 동적 (cards 테이블 기준) | 0~14 (미사용=0, 사용=해당 레벨) | `card_knight = 14` |
| 진화 스위치 벡터 | 동적 (진화 가능 카드) | 0 or 1 | `evo_knight = 1` |
| 수치 피처 (모드별) | 2칸 | 정수 | 모드에 따라 다름 (아래 참조) |

**카드 레벨 인코딩**: 0/1(존재 여부)이 아닌 **실제 레벨값(0~14)**을 사용.
`avg_level`은 카드별 레벨의 합산이므로 중복 정보 → 제거.

**모드별 수치 피처:**

| 모드 | 수치 피처 | 이유 |
|------|-----------|------|
| pathOfLegend | evolution_count, league_number | 레벨 고정이라 카드 레벨값이 동일하지만 벡터 구조 통일 |
| ladder | evolution_count, starting_trophies | 카드별 레벨이 직접 반영되므로 avg_level 불필요 |

**피처 벡터 크기**: `feature_builder.py`가 앱 시작 시 `cards` 테이블을 조회해 동적으로 결정.
신규 카드 추가 패치 시 재학습 시점에 벡터 크기 자동 갱신 → 기존 모델과 크기 불일치 → 교체 조건 미충족으로 구 모델 유지 → 다음 재학습에서 새 크기로 정상 학습.

### 모델 관리

```
models/
├── matchup_pathOfLegend_current.joblib
├── matchup_pathOfLegend_backup.joblib
├── matchup_ladder_current.joblib
└── matchup_ladder_backup.joblib
```

`/predict/matchup` 요청의 `battle_type`으로 모델 자동 선택.
해당 모드 모델 없으면 503 → Spring Boot가 stats_fallback 처리.
특별모드(challenge 등)는 데이터 부족으로 학습 스킵 → stats_fallback.

### 인코딩 예시

```
card_knight     = 14   # 14레벨 기사 사용
card_hog_rider  = 0    # 덱에 없음 (레벨 0 = 미사용)
evo_knight      = 1    # 진화 기사
evo_firecracker = 0    # 진화 없음
```

LightGBM 트리는 카드 레벨 칸과 진화 스위치 칸을 동시에 참조해 "14레벨 진화 기사"를 하나의 조건으로 학습한다.

### 훈련 데이터

- `battle_type = 'pathOfLegend'` (Phase 2 기준, 추후 모드별 모델 확장)
- **Sliding Window**: 최근 **3일** 데이터 (메타 유효 기간)
- **타겟**: `result` (1=win, 0=loss)
- `ORDER BY` 제거 — LightGBM 순서 무관, leakage 방지

---

## 6. 학습 파이프라인

```
POST /train (Airflow 호출)
  │
  ├── 현재 학습 중이면 409 반환 (training_state.json 확인)
  └── training_state.json에 {status: "running", started_at} 기록
      BackgroundTasks 등록 → 즉시 202 반환
        │
        ├── 1. DB 조회
        │     match_features WHERE battle_date >= today - 3일
        │
        ├── 2. train_rows < MIN_TRAIN_ROWS(10000)이면 중단
        │     training_state.json → {status: "failed", error: "insufficient data"}
        │
        ├── 3. feature_builder.py
        │     cards 테이블 조회 → 피처 벡터 크기 동적 결정
        │     멀티-핫 레벨 벡터 + 진화 스위치 벡터 + 수치 피처 조립
        │
        ├── 4. train/val split (time-series)
        │     train: battle_date in [today-3, today-2] (2일)
        │     val:   battle_date = today-1 (1일, 미래 데이터 누출 방지)
        │
        ├── 5. LightGBM 학습
        │     min_child_samples=50
        │     early_stopping_rounds=50, verbose_eval=False
        │
        ├── 6. 모델 교체 조건
        │     matchup_current.joblib 없음 → 최초 학습, 무조건 저장
        │     있음 → accuracy >= 기존 + 0.5% AND logloss < 기존 logloss
        │     미달 시 구 모델 유지, MLflow에 실패 기록
        │
        ├── 7. MLflow 실험 기록
        │     params: window_days=3, val_days=1, min_child_samples=50, battle_type
        │     metrics: val_accuracy, val_logloss, train_rows, val_rows
        │     artifact: 모델 파일
        │
        ├── 8. 모델 핫스왑 (교체 조건 충족 시)
        │     model_store.py → matchup_backup.joblib ← 기존 current 복사
        │                    → matchup_current.joblib ← 신규 모델 저장
        │     asyncio.Lock 획득 → app.state.predictor 교체 → Lock 해제
        │
        └── 9. training_state.json 갱신
              {status: "success"/"failed", finished_at, model_version, metrics}
```

**training_state.json**: 파드 재시작 시 상태 복구용 파일 기반 영속화.
재시작 후 `status == "running"` 이면 `"failed"`로 자동 전환 (앱 시작 시 처리).

---

## 7. 패치 대응 전략

```
패치 당일
  └─ POST /train?reset_window=true 수동 호출 (Airflow or 개발자)
       → reset_window=true: 재학습 실행 없이 training_state.json의
         patch_date를 today로 기록 → 이후 Airflow 정기 트리거가
         patch_date + MIN_POST_PATCH_DAYS 이후에만 실제 학습 실행
  └─ 구 모델 계속 서빙

패치 후 3일 (워밍업)
  └─ post-patch 데이터 누적 중
  └─ 구 모델 서빙 유지

3일 후
  └─ Airflow 정기 재학습 트리거
  └─ 모델 교체 조건 충족 시 교체
```

- 워밍업 기간은 `MIN_POST_PATCH_DAYS` 환경변수로 조정 가능
- 패치 직후 유저들의 메타 탐색 데이터는 노이즈 → 안정화 대기가 맞음
- Phase 3: 카드 win-rate 분포 코사인 유사도로 패치 자동 감지

---

## 8. 모델 파일 관리

```
지금 (Phase 2):
  model_store.py → 로컬 .joblib 파일 (모드별)
  models/matchup_{battle_type}_current.joblib   ← 서빙 중
  models/matchup_{battle_type}_backup.joblib    ← 롤백용
  대상 모드: pathOfLegend, ladder (MIN_TRAIN_ROWS 미달 모드는 스킵)

Phase 3:
  model_store.py → mlflow.pyfunc.load_model() 으로 교체
  나머지 코드 변경 없음 (단일 확장 포인트)
```

---

## 9. MLflow 트래킹

- **지금**: Tracking Server (docker-compose) + 실험 기록
- **Phase 3**: Model Registry + 자동 배포

```python
with mlflow.start_run():
    mlflow.log_params({ "window_days": 3, "val_days": 1, ... })
    mlflow.log_metrics({ "val_accuracy": 0.762, "val_logloss": 0.481, ... })
    mlflow.lightgbm.log_model(model, "matchup_model")
    # Phase 3: mlflow.register_model() 추가
```

---

## 10. 인프라 / Docker

### docker-compose.yml 변경사항

```yaml
# 기존 유지: postgres, redis, api, batch
# 새로 추가:

worker:
  build: ./RoyaleLog-worker
  ports:
    - "8082:8082"
  depends_on:
    - postgres
    - mlflow

mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  ports:
    - "5000:5000"
  command: >
    mlflow server
    --backend-store-uri postgresql://royale:@postgres:5432/royalelog
    --default-artifact-root /mlruns
    --host 0.0.0.0
  volumes:
    - mlflow_artifacts:/mlruns
  depends_on:
    - postgres

volumes:
  mlflow_artifacts:
```

MLflow 테이블(`experiments`, `runs`, `metrics` 등)은 최초 실행 시 자동 생성. Flyway 마이그레이션 불필요. 기존 RoyaleLog 테이블과 이름 충돌 없음.

### .env.example

```env
DB_HOST=postgres
DB_PORT=5432
DB_NAME=royalelog
DB_USER=royale
DB_PASSWORD=

MLFLOW_TRACKING_URI=http://mlflow:5000

MODEL_DIR=./models
WINDOW_DAYS=3
VAL_DAYS=1
MIN_CHILD_SAMPLES=50
MIN_TRAIN_ROWS=10000
MIN_POST_PATCH_DAYS=3

PREDICT_TIMEOUT_MS=500
```

### Spring Boot 연동

```yaml
# application.yml (RoyaleLog-api)
ml:
  server:
    url: http://worker:8082
    timeout-ms: 500
```

---

## 11. 시스템 목표 지표

| 지표 | 목표 |
|------|------|
| 추론 latency | < 50ms |
| 모델 정확도 | val accuracy >= 75% |
| 재학습 소요 시간 | < 5분 |
| Fallback 타임아웃 | 500ms |

---

## 12. 미결 사항 (별도 설계)

- **덱 추천 기능** (`GET /api/v1/recommend/{playerTag}`): Spring Boot + Worker 협업 구조, 별도 brainstorm 예정
- **Phase 3 자동 패치 감지**: 카드 win-rate 코사인 유사도 기반
- **Phase 3 Shadow Deployment**: 새 모델 트래픽 비교 후 교체
- **Phase 3 Redis Bucket4j**: Worker Scale-out 시 글로벌 Rate Limit
