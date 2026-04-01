from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

_engine = None
_SessionLocal = None


def init_db(db_url: str) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(db_url, pool_size=5, max_overflow=10)
    _SessionLocal = sessionmaker(bind=_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
