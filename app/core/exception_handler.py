import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import ErrorCode, make_error

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = {}
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
            details[field] = error["msg"]
        logger.warning("Validation error %s %s: %s", request.method, request.url.path, details)
        return JSONResponse(
            status_code=422,
            content=make_error(
                request, 422, ErrorCode.VALIDATION_FAILED,
                "Request validation failed", details
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code_map = {
            400: ErrorCode.BAD_REQUEST,
            404: ErrorCode.RESOURCE_NOT_FOUND,
            409: ErrorCode.STATE_CONFLICT,
            422: ErrorCode.UNPROCESSABLE_ENTITY,
            429: ErrorCode.TOO_MANY_REQUESTS,
            503: ErrorCode.MODEL_NOT_LOADED,
        }
        code = code_map.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
        # Use TRAINING_CONFLICT for 409 from train endpoint
        if exc.status_code == 409 and "Training" in str(exc.detail):
            code = ErrorCode.TRAINING_CONFLICT
        if exc.status_code == 503 and "No model" in str(exc.detail):
            code = ErrorCode.MODEL_NOT_LOADED

        logger.warning("HTTP %d %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error(request, exc.status_code, code, str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception %s %s\n%s",
            request.method, request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content=make_error(
                request, 500, ErrorCode.INTERNAL_SERVER_ERROR,
                "An unexpected error occurred",
            ),
        )
