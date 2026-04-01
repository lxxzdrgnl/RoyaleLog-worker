import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.core.errors import ErrorResponse
from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

_ERROR_RESPONSES = {
    422: {
        "model": ErrorResponse,
        "description": "Validation error — deck must have 8-9 cards",
        "content": {
            "application/json": {
                "example": {
                    "timestamp": "2026-04-01T12:00:00Z",
                    "path": "/predict/matchup",
                    "status": 422,
                    "code": "VALIDATION_FAILED",
                    "message": "Request validation failed",
                    "details": {"deck_card_ids": "List should have at least 8 items after validation, not 7"}
                }
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "No model loaded for the requested battle_type",
        "content": {
            "application/json": {
                "example": {
                    "timestamp": "2026-04-01T12:00:00Z",
                    "path": "/predict/matchup",
                    "status": 503,
                    "code": "MODEL_NOT_LOADED",
                    "message": "No model loaded for battle_type=pathOfLegend",
                    "details": None
                }
            }
        },
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "timestamp": "2026-04-01T12:00:00Z",
                    "path": "/predict/matchup",
                    "status": 500,
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "details": None
                }
            }
        },
    },
}


@router.post(
    "/matchup",
    response_model=PredictResponse,
    responses=_ERROR_RESPONSES,
    summary="Predict matchup win probability",
    description=(
        "Returns the predicted win probability for the given deck vs opponent deck. "
        "Requires a trained model for the specified `battle_type`. "
        "Returns 503 if no model is loaded — Spring Boot should fall back to stats-based prediction."
    ),
)
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
        deck_card_levels=req.deck_card_levels,
        deck_evo_levels=req.deck_evo_levels,
        opponent_card_ids=req.opponent_card_ids,
        opponent_card_levels=req.opponent_card_levels,
        opponent_evo_levels=req.opponent_evo_levels,
        avg_level=0.0,
        evolution_count=sum(1 for e in req.deck_evo_levels if e > 0),
        league_number=req.league_number,
        starting_trophies=req.starting_trophies,
    )

    probs = predictor.predict(vec.reshape(1, -1))

    logger.info(
        "predict request_id=%s type=%s prob=%.4f",
        req.request_id, req.battle_type, probs[0],
    )

    return PredictResponse(
        request_id=req.request_id,
        win_probability=round(float(probs[0]), 4),
        model_version=predictor.version,
    )
