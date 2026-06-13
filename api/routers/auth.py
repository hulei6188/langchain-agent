from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership, get_current_user, require_manager
from api.payloads import invite_payload, invite_workspace, membership_payload, user_payload, workspace_payload
from api.schemas import (
    InviteAcceptRequest,
    InviteCreateRequest,
    LoginRequest,
    RegisterRequest,
    UserProfileUpdateRequest,
)
from core.config import get_settings
from core.db.models import User, WorkspaceInvite, WorkspaceMember
from core.db.session import get_db
from core.security.auth import create_access_token, hash_password, verify_password
from core.security.permissions import normalize_role
from core.services.bootstrap import create_default_workspace_user, create_first_user_workspace, has_any_user


router = APIRouter()


@router.post("/api/auth/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == request.email.lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    invite = None
    if request.invite_token:
        invite = (
            db.query(WorkspaceInvite)
            .filter(
                WorkspaceInvite.token == request.invite_token,
                WorkspaceInvite.accepted_at.is_(None),
                WorkspaceInvite.email == request.email.lower(),
            )
            .first()
        )
        if not invite:
            raise HTTPException(status_code=404, detail="Invite not found")
    if invite:
        user = User(email=request.email.lower(), name=request.name, password_hash=hash_password(request.password))
        db.add(user)
        db.flush()
        db.add(WorkspaceMember(workspace_id=invite.workspace_id, user_id=user.id, role=normalize_role(invite.role)))
        invite.accepted_at = datetime.now(timezone.utc)
        db.commit()
        workspace = invite_workspace(db, invite.workspace_id)
        role = normalize_role(invite.role)
    elif has_any_user(db):
        user, workspace = create_default_workspace_user(db, email=request.email, name=request.name, password=request.password)
        role = "user"
    else:
        user, workspace = create_first_user_workspace(db, email=request.email, name=request.name, password=request.password)
        role = "admin"
    token = create_access_token(user.id, workspace.id)
    return {"access_token": token, "token_type": "bearer", "user": user_payload(user), "workspace": workspace_payload(workspace, role)}


@router.post("/api/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    membership = db.query(WorkspaceMember).filter(WorkspaceMember.user_id == user.id).first()
    token = create_access_token(user.id, membership.workspace_id if membership else None)
    return {"access_token": token, "token_type": "bearer", "user": user_payload(user)}


@router.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user), membership: WorkspaceMember = Depends(get_current_membership)):
    return {"user": user_payload(current_user), "membership": membership_payload(membership)}


@router.patch("/api/auth/me")
def update_me(request: UserProfileUpdateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patch = request.model_dump(exclude_unset=True)
    user = db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if "name" in patch and patch["name"] is not None:
        user.name = patch["name"].strip()
    if "avatar_url" in patch:
        avatar_url = patch["avatar_url"] or ""
        if avatar_url and not avatar_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="avatar_url must be an image data URL")
        user.avatar_url = avatar_url
    db.commit()
    db.refresh(user)
    return {"user": user_payload(user)}


@router.get("/api/workspaces/current")
def current_workspace(membership: WorkspaceMember = Depends(get_current_membership)):
    return {"workspace": workspace_payload(membership.workspace, membership.role), "membership": membership_payload(membership)}


@router.get("/api/workspaces/invites")
def list_invites(membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    if not get_settings().invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    invites = db.query(WorkspaceInvite).filter(WorkspaceInvite.workspace_id == membership.workspace_id).all()
    return {"items": [invite_payload(invite, include_token=False) for invite in invites]}


@router.get("/api/workspaces/members")
def list_members(membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    rows = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == membership.workspace_id)
        .order_by(WorkspaceMember.id.asc())
        .all()
    )
    return {
        "items": [
            {
                "id": row.id,
                "role": normalize_role(row.role),
                "user": user_payload(row.user),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@router.post("/api/workspaces/invites")
def create_invite(request: InviteCreateRequest, membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    if not get_settings().invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    if normalize_role(request.role) != "user":
        raise HTTPException(status_code=400, detail="Invite role must be user")
    invite = WorkspaceInvite(
        workspace_id=membership.workspace_id,
        email=request.email.lower(),
        role="user",
        token=secrets.token_urlsafe(24),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return {"invite": invite_payload(invite, include_token=False)}


@router.post("/api/workspaces/invites/accept")
def accept_invite(request: InviteAcceptRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not get_settings().invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    invite = db.query(WorkspaceInvite).filter(WorkspaceInvite.token == request.token, WorkspaceInvite.accepted_at.is_(None)).first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    existing = db.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == invite.workspace_id, WorkspaceMember.user_id == current_user.id).first()
    if not existing:
        db.add(WorkspaceMember(workspace_id=invite.workspace_id, user_id=current_user.id, role=normalize_role(invite.role)))
    invite.accepted_at = datetime.now(timezone.utc)
    db.commit()
    return {"accepted": True}
