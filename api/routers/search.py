from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_membership
from core.config import get_settings
from core.db.models import WorkspaceMember
from core.services.web_search import search_web, web_search_status


router = APIRouter()


@router.get("/api/search/test")
def test_web_search(
    q: str = Query(min_length=1, max_length=300),
    membership: WorkspaceMember = Depends(get_current_membership),
):
    settings = get_settings()
    try:
        return {"ok": True, **search_web(q)}
    except ValueError as exc:
        return {
            "ok": False,
            "query": q,
            "provider": web_search_status().get("provider", settings.web_search_provider),
            "items": [],
            "error_code": str(exc),
        }
