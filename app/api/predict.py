import asyncio
import logging

import numpy as np
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
        deck_card_levels=req.deck_card_levels,
        deck_evo_levels=req.deck_evo_levels,
        opponent_card_ids=req.opponent_card_ids,
        opponent_card_levels=req.opponent_card_levels,
        opponent_evo_levels=req.opponent_evo_levels,
        avg_level=0.0,  # unknown at inference time
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
