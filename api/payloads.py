from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.db.models import User, Workspace, WorkspaceInvite, WorkspaceMember
from core.security.permissions import normalize_role


def user_payload(user: User) -> dict:
    return {"id": user.id, "email": user.email, "name": user.name, "avatar_url": user.avatar_url or ""}


def workspace_payload(workspace: Workspace, role: str) -> dict:
    return {"id": workspace.id, "name": workspace.name, "slug": workspace.slug, "role": normalize_role(role)}


def membership_payload(membership: WorkspaceMember) -> dict:
    return {"workspace_id": membership.workspace_id, "user_id": membership.user_id, "role": normalize_role(membership.role)}


def invite_payload(invite: WorkspaceInvite, *, include_token: bool = False) -> dict:
    payload = {
        "id": invite.id,
        "email": invite.email,
        "role": normalize_role(invite.role),
        "accepted_at": invite.accepted_at.isoformat() if invite.accepted_at else None,
    }
    if include_token:
        payload["token"] = invite.token
    return payload


def invite_workspace(db: Session, workspace_id: int) -> Workspace:
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace
