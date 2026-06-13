from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership
from api.schemas import MemoryProfileUpdateRequest
from core.db.models import Agent, WorkspaceMember
from core.db.session import get_db
from core.security.permissions import can_manage
from core.services.memory import (
    delete_memory_profile,
    get_memory_profile,
    memory_profile_payload,
    upsert_memory_profile,
)

router = APIRouter(prefix="/api/agents/{agent_id}/memory-profile", tags=["memory"])


@router.get("")
def get_agent_memory_profile(
    agent_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = _require_workspace_agent(db, membership.workspace_id, agent_id)
    _require_agent_read_access(agent, membership)
    profile = get_memory_profile(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        agent_id=agent.id,
    )
    return {"profile": memory_profile_payload(profile, agent_id=agent.id)}


@router.patch("")
def patch_agent_memory_profile(
    agent_id: int,
    request: MemoryProfileUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = _require_workspace_agent(db, membership.workspace_id, agent_id)
    _require_agent_read_access(agent, membership)
    try:
        profile = upsert_memory_profile(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            agent_id=agent.id,
            payload=request.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": memory_profile_payload(profile)}


@router.delete("")
def delete_agent_memory_profile(
    agent_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = _require_workspace_agent(db, membership.workspace_id, agent_id)
    _require_agent_read_access(agent, membership)
    delete_memory_profile(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        agent_id=agent.id,
    )
    return {"deleted": True}


def _require_workspace_agent(db: Session, workspace_id: int, agent_id: int) -> Agent:
    agent = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _require_agent_read_access(agent: Agent, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if agent.created_by == membership.user_id:
        return
    raise HTTPException(status_code=403, detail="Agent access denied")
