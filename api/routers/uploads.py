from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership, get_current_user
from api.schemas import UploadCreateRequest
from core.db.models import User, WorkspaceMember
from core.db.session import get_db
from core.services.uploads import create_upload, upload_payload


router = APIRouter()


@router.post("/api/uploads")
def upload_file(
    request: UploadCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        upload = create_upload(
            db,
            workspace_id=membership.workspace_id,
            user_id=current_user.id,
            filename=request.filename,
            content_type=request.content_type,
            content_base64=request.content_base64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"upload": upload_payload(upload)}
