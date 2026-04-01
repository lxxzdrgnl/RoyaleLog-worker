from datetime import datetime
from pydantic import BaseModel


class TrainResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "accepted",
                "message": "Training started"
            }
        }
    }

    status: str
    message: str


class TrainStatusResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "started_at": "2026-04-01T03:00:00Z",
                "finished_at": "2026-04-01T03:05:23Z",
                "model_version": "lgbm-pathOfLegend-20260401030523",
                "train_rows": 85000,
                "val_rows": 12000,
                "val_accuracy": 0.6341,
                "val_logloss": 0.6712,
                "error": None
            }
        }
    }

    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_version: str | None = None
    train_rows: int | None = None
    val_rows: int | None = None
    val_accuracy: float | None = None
    val_logloss: float | None = None
    error: str | None = None
