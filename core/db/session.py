from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from core.db.base import Base


settings = get_settings()
engine = create_engine(settings.database_url, future=True)


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from core.db import models  # noqa: F401
    from core.runtime.langgraph_persistence import configure_langgraph_persistence
    from core.services.bootstrap import ensure_default_models

    Base.metadata.create_all(bind=engine)
    configure_langgraph_persistence(settings.database_url)
    db = SessionLocal()
    try:
        ensure_default_models(db)
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
