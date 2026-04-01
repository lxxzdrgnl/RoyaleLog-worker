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
