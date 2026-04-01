import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, predict, train
from app.core.config import Settings
from app.core.database import get_session
from app.core.exception_handler import register_exception_handlers
from app.core.middleware import LoggingMiddleware
from app.ml.feature_builder import FeatureBuilder
from app.ml.model_store import ModelStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.model_lock = asyncio.Lock()

    model_store = ModelStore(settings.model_dir)
    app.state.model_store = model_store

    # Load feature builder from DB
    try:
        with get_session() as session:
            builder = FeatureBuilder.from_db(session)
        app.state.feature_builder = builder
        logger.info("FeatureBuilder loaded: %d cards", builder._num_cards)
    except Exception:
        logger.warning("FeatureBuilder could not be loaded at startup (DB may be unavailable)")
        app.state.feature_builder = None

    # Load current models for each battle type
    predictors: dict = {}
    for battle_type in settings.train_battle_types:
        predictor = model_store.load_current(battle_type)
        if predictor is not None:
            predictors[battle_type] = predictor
            logger.info("Loaded model for battle_type=%s version=%s", battle_type, predictor.version)
        else:
            logger.warning("No model found for battle_type=%s", battle_type)
    app.state.predictors = predictors

    # Recover stale training state (running → failed if worker was killed mid-training)
    import json, os
    state_path = settings.state_file_path
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if state.get("status") == "running":
            from datetime import datetime, timezone
            state["status"] = "failed"
            state["error"] = "Worker restarted during training"
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            with open(state_path, "w") as f:
                json.dump(state, f, default=str)
            logger.warning("Recovered stale training state: running → failed")

    yield

    logger.info("Worker shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RoyaleLog Worker",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(LoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(train.router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.worker_port, reload=True)
