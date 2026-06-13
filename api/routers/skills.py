from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership
from api.schemas import SkillCreateRequest, SkillItemIdsRequest, SkillUpdateRequest
from core.db.models import Skill, WorkspaceMember
from core.db.session import get_db
from core.services.skills import (
    _replace_skill_kbs,
    _replace_skill_tools,
    create_skill,
    delete_skill,
    get_skill_detail,
    list_workspace_skills,
    update_skill,
)
from core.services.tools import validate_tool_ids


router = APIRouter()


@router.get("/api/skills")
def list_skills(
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    items = list_workspace_skills(db, workspace_id=membership.workspace_id)
    return {"items": items}


@router.post("/api/skills")
def create_skill_endpoint(
    request: SkillCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    if request.tool_ids:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.tool_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    skill = create_skill(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        payload=request.model_dump(),
    )
    return {"skill": get_skill_detail(db, skill)}


@router.get("/api/skills/{skill_id}")
def get_skill(
    skill_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    return {"skill": get_skill_detail(db, skill)}


@router.patch("/api/skills/{skill_id}")
def patch_skill(
    skill_id: int,
    request: SkillUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    if request.tool_ids is not None:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.tool_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    skill = update_skill(db, skill=skill, payload=request.model_dump(exclude_unset=True))
    return {"skill": get_skill_detail(db, skill)}


@router.delete("/api/skills/{skill_id}")
def delete_skill_endpoint(
    skill_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    delete_skill(db, skill=skill)
    return {"deleted": True}


@router.put("/api/skills/{skill_id}/tools")
def update_skill_tools(
    skill_id: int,
    request: SkillItemIdsRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    if request.ids:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    _replace_skill_tools(db, skill.id, request.ids)
    db.commit()
    return {"skill": get_skill_detail(db, skill)}


@router.put("/api/skills/{skill_id}/knowledge-bases")
def update_skill_knowledge_bases(
    skill_id: int,
    request: SkillItemIdsRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    _replace_skill_kbs(db, skill.id, request.ids)
    db.commit()
    return {"skill": get_skill_detail(db, skill)}


def require_workspace_skill(db: Session, workspace_id: int, skill_id: int) -> Skill:
    skill = db.query(Skill).filter(Skill.workspace_id == workspace_id, Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill
