from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership, get_current_user
from api.schemas import FeedbackRequest, SessionUpdateRequest
from core.db.models import Agent, Feedback, Message, Session as ChatSession, User, WorkspaceMember
from core.db.session import get_db
from core.security.permissions import can_manage
from core.services.chat_sessions import (
    active_run_payload,
    chat_message_payloads,
    delete_chat_session,
    session_payload,
)

router = APIRouter(tags=["sessions"])


@router.get("/api/agents/{agent_id}/sessions")
def list_agent_sessions(
    agent_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = _require_workspace_agent(db, membership.workspace_id, agent_id)
    _require_agent_read_access(agent, membership)
    query = db.query(ChatSession).filter(
        ChatSession.workspace_id == membership.workspace_id,
        ChatSession.agent_id == agent.id,
        ChatSession.is_debug == False,
    )
    if not can_manage(membership.role):
        query = query.filter(ChatSession.user_id == membership.user_id)
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    items = []
    for session in sessions:
        payload = session_payload(session, db)
        payload["active_run"] = active_run_payload(db, session.id)
        items.append(payload)
    return {"items": items}


@router.get("/api/sessions/{session_id}")
def get_session(
    session_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    session = _require_workspace_session(db, membership, session_id)
    messages = db.query(Message).filter(Message.session_id == session.id).order_by(Message.id.asc()).all()
    return {
        "session": session_payload(session, db),
        "messages": chat_message_payloads(messages),
        "active_run": active_run_payload(db, session.id),
    }


@router.patch("/api/sessions/{session_id}")
def patch_session(
    session_id: int,
    request: SessionUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    session = _require_workspace_session(db, membership, session_id)
    session.title = request.title.strip()
    db.commit()
    db.refresh(session)
    return {"session": session_payload(session, db)}


@router.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    session = _require_workspace_session(db, membership, session_id)
    delete_chat_session(db, session)
    return {"deleted": True}


@router.post("/api/messages/{message_id}/feedback")
def create_feedback(
    message_id: int,
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    message = db.get(Message, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    session = db.get(ChatSession, message.session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Message not found")
    _require_session_access(session, membership)
    feedback = Feedback(message_id=message.id, user_id=current_user.id, rating=request.rating, comment=request.comment)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"feedback": {"id": feedback.id, "rating": feedback.rating, "comment": feedback.comment}}


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


def _require_workspace_session(db: Session, membership: WorkspaceMember, session_id: int) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_access(session, membership)
    return session


def _require_session_access(session: ChatSession, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if session.user_id == membership.user_id:
        return
    raise HTTPException(status_code=404, detail="Session not found")
