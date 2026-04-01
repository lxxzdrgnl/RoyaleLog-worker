from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns ok when the worker is running.",
    response_description="Service is healthy",
)
def health():
    return {"status": "ok"}
