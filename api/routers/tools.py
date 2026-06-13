from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_membership
from api.schemas import MCPToolDiscoverRequest, ToolRequest, ToolTestRequest, ToolUpdateRequest
from core.db.models import WorkspaceMember
from core.db.session import get_db
from core.services.bootstrap import ensure_builtin_tools
from core.services.tools import (
    create_tool,
    delete_tool,
    discover_mcp_tools,
    get_accessible_tool,
    list_available_tools,
    test_tool,
    tool_payload,
    update_tool,
)

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
def list_tools(
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    ensure_builtin_tools(db)
    tools = list_available_tools(db, workspace_id=membership.workspace_id, user_id=membership.user_id)
    return {"items": [tool_payload(tool) for tool in tools]}


@router.post("")
def create_tool_endpoint(
    request: ToolRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    try:
        tool = create_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, payload=request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": tool_payload(tool)}


@router.post("/mcp/discover")
def discover_mcp_tools_endpoint(
    request: MCPToolDiscoverRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    existing_tool = None
    if request.tool_id:
        existing_tool = get_accessible_tool(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            tool_id=request.tool_id,
        )
        if not existing_tool:
            raise HTTPException(status_code=404, detail="Tool not found")
    try:
        items = discover_mcp_tools(request.model_dump(), existing_tool=existing_tool)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items}


@router.patch("/{tool_id}")
def patch_tool_endpoint(
    tool_id: int,
    request: ToolUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    try:
        tool = update_tool(db, tool=tool, payload=request.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": tool_payload(tool)}


@router.delete("/{tool_id}")
def delete_tool_endpoint(
    tool_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    try:
        delete_tool(db, tool=tool)
    except ValueError as exc:
        status_code = 409 if "in use" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return {"deleted": True}


@router.post("/{tool_id}/test")
def test_tool_endpoint(
    tool_id: int,
    request: ToolTestRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return test_tool(tool, input_data=request.input, body=request.body)
