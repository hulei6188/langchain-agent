from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_current_membership
from api.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeDocumentBatchCreateRequest,
    KnowledgeDocumentCreateRequest,
)
from core.db.models import KnowledgeBase, KnowledgeChunk, KnowledgeDocument, WorkspaceMember
from core.db.session import get_db
from core.security.permissions import can_manage
from core.services.knowledge import (
    KnowledgeDocumentError,
    add_document,
    clear_knowledge_base_documents,
    create_knowledge_base,
    delete_document,
    delete_knowledge_base,
    document_payload,
    index_document,
    knowledge_base_summary,
    list_document_chunks,
    reindex_knowledge_base,
    split_by_hierarchy,
    split_parent_child,
)
from core.services.rag_cache import redis_store
from core.services.run_streams import sanitize_public_error

router = APIRouter(tags=["knowledge"])
logger = logging.getLogger(__name__)


class ResegmentRequest(BaseModel):
    parse_mode: str = "precise"
    segment_mode: str = "auto"
    delimiter: str | None = "##"
    max_chunk_len: int = 5000
    overlap_pct: int = 10
    hierarchy_level: int = 3
    keep_hierarchy_info: bool = True


@router.get("/api/knowledge-bases")
def list_knowledge_bases(
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == membership.workspace_id).order_by(KnowledgeBase.id.desc()).all()
    items = []
    for kb in kbs:
        count = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).count()
        items.append(knowledge_base_summary(kb, count))
    return {"items": items}


@router.post("/api/knowledge-bases")
def create_kb(
    request: KnowledgeBaseCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = create_knowledge_base(db, workspace_id=membership.workspace_id, user_id=membership.user_id, name=request.name, description=request.description)
    return {"knowledge_base": knowledge_base_summary(kb)}


@router.post("/api/knowledge-bases/{kb_id}/documents")
def upload_document(
    kb_id: int,
    request: KnowledgeDocumentCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    logger.info(
        "Knowledge document upload request: schema=%s kb_id=%s workspace_id=%s filename=%s source_type=%s",
        request.__class__.__name__,
        kb.id,
        membership.workspace_id,
        request.filename or request.title or "",
        request.source_type,
    )
    try:
        document, payload = _add_knowledge_document_from_request(db, workspace_id=membership.workspace_id, kb=kb, request=request)
    except KnowledgeDocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        logger.exception("Knowledge document indexing failed")
        raise HTTPException(status_code=502, detail={"message": sanitize_public_error(str(exc)), "error_code": "knowledge_index_failed"}) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("Knowledge document upload failed")
        raise HTTPException(status_code=500, detail={"message": "Knowledge document upload failed.", "error_code": "knowledge_upload_failed"}) from exc
    if document.status == "failed":
        raise HTTPException(status_code=422, detail={"message": document.error_message or "Document text extraction failed", "document": payload})
    return {"document": payload}


@router.post("/api/knowledge-bases/{kb_id}/documents/batch")
def upload_documents_batch(
    kb_id: int,
    request: KnowledgeDocumentBatchCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    logger.info(
        "Knowledge document upload request: schema=%s kb_id=%s workspace_id=%s count=%s filenames=%s",
        request.__class__.__name__,
        kb.id,
        membership.workspace_id,
        len(request.documents),
        [item.filename or item.title or f"document-{index + 1}" for index, item in enumerate(request.documents)],
    )
    documents = []
    errors = []

    for index, item in enumerate(request.documents):
        filename = item.filename or item.title or f"document-{index + 1}"
        try:
            document, payload = _add_knowledge_document_from_request(db, workspace_id=membership.workspace_id, kb=kb, request=item)
            if document.status == "failed":
                errors.append(
                    {
                        "index": index,
                        "filename": filename,
                        "message": document.error_message or "Document text extraction failed",
                        "document": payload,
                    }
                )
            else:
                documents.append(payload)
        except KnowledgeDocumentError as exc:
            db.rollback()
            errors.append({"index": index, "filename": filename, "message": str(exc), "status_code": exc.status_code})
        except RuntimeError as exc:
            db.rollback()
            logger.exception("Knowledge document batch indexing failed")
            errors.append(
                {
                    "index": index,
                    "filename": filename,
                    "message": sanitize_public_error(str(exc)),
                    "error_code": "knowledge_index_failed",
                }
            )
        except Exception:
            db.rollback()
            logger.exception("Knowledge document batch upload failed")
            errors.append(
                {
                    "index": index,
                    "filename": filename,
                    "message": "Knowledge document upload failed.",
                    "error_code": "knowledge_upload_failed",
                }
            )

    return {
        "documents": documents,
        "errors": errors,
        "total": len(request.documents),
        "succeeded": len(documents),
        "failed": len(errors),
    }


@router.get("/api/knowledge-bases/{kb_id}/documents")
def list_documents(
    kb_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    documents = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).order_by(KnowledgeDocument.id.desc()).all()
    return {
        "items": [
            document_payload(
                document,
                db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document.id).count(),
            )
            for document in documents
        ]
    }


@router.delete("/api/knowledge-bases/{kb_id}/documents")
def clear_documents(
    kb_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    try:
        summary = clear_knowledge_base_documents(db, workspace_id=membership.workspace_id, kb=kb)
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Knowledge base documents clear failed: kb_id=%s workspace_id=%s",
            kb.id,
            membership.workspace_id,
        )
        raise HTTPException(status_code=500, detail={"message": "Knowledge base documents clear failed.", "error_code": "knowledge_documents_clear_failed"}) from exc
    logger.info(
        "Knowledge base documents cleared: kb_id=%s workspace_id=%s documents_deleted=%s chunks_deleted=%s vectors_delete_requested=%s",
        kb.id,
        membership.workspace_id,
        summary.get("documents_deleted", 0),
        summary.get("chunks_deleted", 0),
        summary.get("vectors_delete_requested", False),
    )
    return {"cleared": True, **summary}


@router.delete("/api/knowledge-bases/{kb_id}/documents/{document_id}")
def remove_document(
    kb_id: int,
    document_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    document = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id, KnowledgeDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        summary = delete_document(db, workspace_id=membership.workspace_id, document=document)
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Knowledge document delete failed: kb_id=%s document_id=%s workspace_id=%s",
            kb.id,
            document_id,
            membership.workspace_id,
        )
        raise HTTPException(status_code=500, detail={"message": "Knowledge document delete failed.", "error_code": "knowledge_document_delete_failed"}) from exc
    logger.info(
        "Knowledge document deleted: kb_id=%s document_id=%s workspace_id=%s chunks_deleted=%s vectors_delete_requested=%s",
        kb.id,
        document_id,
        membership.workspace_id,
        summary.get("chunks_deleted", 0),
        summary.get("vectors_delete_requested", False),
    )
    return {"deleted": True, **summary}


@router.post("/api/knowledge-bases/{kb_id}/index")
def index_kb(
    kb_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    job_id = f"kb-{kb.id}-sync"
    summary = reindex_knowledge_base(db, workspace_id=membership.workspace_id, kb=kb)
    status = "failed" if summary["documents_failed"] and not summary["documents_indexed"] else "succeeded"
    payload = {
        "job_id": job_id,
        "knowledge_base_id": kb.id,
        "status": status,
        "message": (
            f"Rebuilt {summary['chunks_indexed']} chunks for {summary['documents_indexed']} documents."
            if status == "succeeded"
            else "Knowledge base reindex failed for all documents."
        ),
        **summary,
    }
    redis_store.set_job(job_id, payload)
    return payload


@router.get("/api/knowledge/jobs/{job_id}")
def get_knowledge_job(
    job_id: str,
    _: WorkspaceMember = Depends(get_current_membership),
):
    lookup = redis_store.get_job(job_id)
    if lookup.hit and lookup.value:
        return lookup.value
    return {"job_id": job_id, "status": "unknown", "message": "Job state is not available or Redis is not configured."}


@router.delete("/api/knowledge-bases/{kb_id}")
def delete_kb(
    kb_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    delete_knowledge_base(db, workspace_id=membership.workspace_id, kb=kb)
    return {"deleted": True}


@router.patch("/api/knowledge-bases/{kb_id}")
def update_kb(
    kb_id: int,
    request: KnowledgeBaseCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    kb.name = request.name
    kb.description = request.description
    db.commit()
    db.refresh(kb)
    return {"knowledge_base": knowledge_base_summary(kb)}


@router.get("/api/knowledge-bases/{kb_id}/documents/{document_id}/chunks")
def get_document_chunks(
    kb_id: int,
    document_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = list_document_chunks(db, document_id=document.id)
    return {"document": document_payload(document, len(chunks)), "chunks": chunks}


@router.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/preview")
def preview_document_chunks(
    kb_id: int,
    document_id: int,
    request: ResegmentRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    cfg = request.model_dump()
    seg_mode = cfg.get("segment_mode", "auto")
    if seg_mode == "hierarchy":
        chunks = split_by_hierarchy(
            document.text,
            kb_id=kb.id,
            document_id=document.id,
            max_level=cfg.get("hierarchy_level", 3),
            keep_hierarchy_info=cfg.get("keep_hierarchy_info", True),
        )
    elif seg_mode == "custom":
        chunks = split_parent_child(
            document.text,
            kb_id=kb.id,
            document_id=document.id,
            parent_size=cfg.get("max_chunk_len", 1600),
            child_size=int(cfg.get("max_chunk_len", 1600) * 0.35),
            overlap=int(cfg.get("max_chunk_len", 1600) * cfg.get("overlap_pct", 10) / 100),
        )
    else:
        chunks = split_parent_child(document.text, kb_id=kb.id, document_id=document.id)

    return {
        "chunks_count": len(chunks),
        "preview_items": [
            {
                "chunk_index": idx,
                "text": chunk.get("text", ""),
                "hierarchy_path": chunk.get("section", ""),
            }
            for idx, chunk in enumerate(chunks)
        ],
    }


@router.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/resegment")
def resegment_document_chunks(
    kb_id: int,
    document_id: int,
    request: ResegmentRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    kb = _require_workspace_kb(db, membership.workspace_id, kb_id)
    _require_kb_write_access(kb, membership)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    document.segment_config = request.model_dump()
    db.commit()

    try:
        chunk_count = index_document(
            db,
            workspace_id=membership.workspace_id,
            kb=kb,
            document=document,
            clear_existing=True,
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail={"message": f"Resegment index failed: {str(exc)}"}) from exc

    return {"document": document_payload(document, chunk_count)}


def _add_knowledge_document_from_request(
    db: Session,
    *,
    workspace_id: int,
    kb: KnowledgeBase,
    request: KnowledgeDocumentCreateRequest,
) -> tuple[KnowledgeDocument, dict]:
    document = add_document(
        db,
        workspace_id=workspace_id,
        kb=kb,
        filename=request.filename,
        title=request.title,
        text=request.text,
        content=request.content,
        content_type=request.content_type,
        content_base64=request.content_base64,
        source_type=request.source_type,
    )
    payload = document_payload(document, db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document.id).count())
    return document, payload


def _require_workspace_kb(db: Session, workspace_id: int, kb_id: int) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == workspace_id, KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _require_kb_write_access(kb: KnowledgeBase, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if kb.created_by == membership.user_id:
        return
    raise HTTPException(status_code=403, detail="Knowledge base edit denied")
