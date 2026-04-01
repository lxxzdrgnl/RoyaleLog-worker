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
    with open(path) as f:
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
        warmup_end = patch_date + timedelta(days=settings.min_post_patch_days)
        if date.today() < warmup_end:
            return TrainResponse(
                status="accepted",
                message=f"Warming up until {warmup_end}. Skipped.",
            )

    if state.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    _write_state(state_path, {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    background_tasks.add_task(_run_training, request.app)
    return TrainResponse(status="accepted", message="Training started")


def _run_training(app) -> None:
    """Sync function — FastAPI runs this in threadpool to avoid blocking the event loop."""
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
    """Train one mode. Returns None if insufficient data."""
    with get_session() as session:
        cutoff = date.today() - timedelta(days=settings.window_days)
        result = session.execute(
            text("""
                SELECT card_ids, card_evo_levels, card_levels,
                       opponent_card_ids, opponent_card_levels, opponent_card_evo_levels,
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
        card_ids      = list(row[0]) if row[0] else []
        evo_levels    = list(row[1]) if row[1] else [0] * len(card_ids)
        card_levels   = list(row[2]) if row[2] else [13] * len(card_ids)
        opp_card_ids  = list(row[3]) if row[3] else []
        opp_card_lvls = list(row[4]) if row[4] else [13] * len(opp_card_ids)
        opp_evo_lvls  = list(row[5]) if row[5] else [0] * len(opp_card_ids)
        avg_level     = float(row[8]) if row[8] else 0.0
        evo_count     = int(row[9]) if row[9] else 0
        league        = int(row[10]) if row[10] is not None else None
        trophies      = int(row[11]) if row[11] is not None else None

        vec = builder.build_matchup_vector(
            battle_type=battle_type,
            deck_card_ids=card_ids, deck_card_levels=card_levels, deck_evo_levels=evo_levels,
            opponent_card_ids=opp_card_ids, opponent_card_levels=opp_card_lvls, opponent_evo_levels=opp_evo_lvls,
            avg_level=avg_level, evolution_count=evo_count,
            league_number=league, starting_trophies=trophies,
        )
        X_list.append(vec)
        y_list.append(int(row[12]))
        dates.append(row[7])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    val_cutoff = date.today() - timedelta(days=settings.val_days)
    val_mask = np.array([d >= val_cutoff for d in dates])
    train_mask = ~val_mask
    X_sorted = np.concatenate([X[train_mask], X[val_mask]])
    y_sorted = np.concatenate([y[train_mask], y[val_mask]])
    val_start = int(train_mask.sum())

    trainer = Trainer(
        builder=builder,
        store=app.state.model_store,
        battle_type=battle_type,
        min_child_samples=settings.min_child_samples,
        early_stopping_rounds=settings.early_stopping_rounds,
        accuracy_margin=settings.accuracy_margin,
    )
    train_result = trainer.train_from_arrays(X_sorted, y_sorted, val_start)

    # MLflow tracking
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
            "train_rows": float(train_result.train_rows),
            "val_rows": float(train_result.val_rows),
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


async def _swap_model(app, battle_type, new_predictor) -> None:
    """Called from threadpool via run_coroutine_threadsafe to swap model under asyncio.Lock."""
    async with app.state.model_lock:
        app.state.predictors[battle_type] = new_predictor
