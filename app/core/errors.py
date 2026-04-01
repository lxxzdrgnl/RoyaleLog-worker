from datetime import datetime, timezone
from enum import Enum

from fastapi import Request
from pydantic import BaseModel


class ErrorCode(str, Enum):
    # 400
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INVALID_QUERY_PARAM = "INVALID_QUERY_PARAM"
    # 404
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    # 409
    STATE_CONFLICT = "STATE_CONFLICT"
    TRAINING_CONFLICT = "TRAINING_CONFLICT"
    # 422
    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"
    # 429
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    # 500
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    # 503
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"


class ErrorResponse(BaseModel):
    timestamp: str
    path: str
    status: int
    code: str
    message: str
    details: dict | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-04-01T12:00:00Z",
                "path": "/predict/matchup",
                "status": 503,
                "code": "MODEL_NOT_LOADED",
                "message": "No model loaded for battle_type=pathOfLegend",
                "details": None
            }
        }
    }


def make_error(request: Request, status: int, code: ErrorCode, message: str, details: dict | None = None) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "path": request.url.path,
        "status": status,
        "code": code.value,
        "message": message,
        "details": details,
    }
