from __future__ import annotations

import logging
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.lifecycle import create_lifespan
from api.routers.agents import router as agents_router
from api.routers.auth import router as auth_router
from api.routers.health import create_health_router
from api.routers.knowledge import router as knowledge_router
from api.routers.memory import router as memory_router
from api.routers.models import router as models_router
from api.routers.prompt_templates import router as prompt_templates_router
from api.routers.runs import router as runs_router
from api.routers.search import router as search_router
from api.routers.sessions import router as sessions_router
from api.routers.skills import router as skills_router
from api.routers.tools import router as tools_router
from api.routers.uploads import router as uploads_router
from core.config import get_settings


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value or "").split("."):
        match = re.match(r"(\d+)", token)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts or [0])


def _validate_runtime_dependencies() -> None:
    import fastapi as fastapi_pkg
    import starlette as starlette_pkg

    starlette_version = getattr(starlette_pkg, "__version__", "0")
    if not ((0, 40, 0) <= _version_tuple(starlette_version) < (0, 42, 0)):
        raise RuntimeError(
            "Incompatible dependency set detected: "
            f"fastapi {getattr(fastapi_pkg, '__version__', 'unknown')} requires "
            f"starlette>=0.40.0,<0.42.0, but found starlette {starlette_version}. "
            "This usually happens when MCP-related dependencies upgrade Starlette transitively. "
            "Reinstall with the same interpreter used to start the server, for example: "
            "`python -m pip install -r requirements.txt` and then "
            "`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`."
        )


settings = get_settings()
_validate_runtime_dependencies()
logger = logging.getLogger(__name__)
lifecycle_state, lifespan = create_lifespan(settings, logger)
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://127.0.0.1:5174", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_health_router(lambda: lifecycle_state.startup_error))
app.include_router(agents_router)
app.include_router(auth_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(models_router)
app.include_router(prompt_templates_router)
app.include_router(runs_router)
app.include_router(search_router)
app.include_router(sessions_router)
app.include_router(skills_router)
app.include_router(tools_router)
app.include_router(uploads_router)
