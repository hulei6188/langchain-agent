from __future__ import annotations

from typing import TypedDict


class WorkflowGraphSpec(TypedDict):
    nodes: list[dict]
    edges: list[dict]
    conditional_edges: list[dict]
    entrypoint: str
    checkpointer_config: dict


def default_workflow_nodes() -> list[dict]:
    return [
        {"id": "start", "type": "Start", "name": "接收用户输入", "config": {}},
        {"id": "knowledge", "type": "Knowledge", "name": "检索绑定知识库", "config": {"top_k": 4}},
        {"id": "tool", "type": "Tool", "name": "调用绑定工具", "config": {"tools": []}},
        {"id": "llm", "type": "LLM", "name": "生成候选回答", "config": {}},
        {"id": "answer", "type": "Answer", "name": "输出最终回答", "config": {}},
    ]


def default_workflow() -> WorkflowGraphSpec:
    return workflow_graph_spec({"nodes": default_workflow_nodes()})


def workflow_graph_spec(definition: list[dict] | dict | None) -> WorkflowGraphSpec:
    if isinstance(definition, dict) and isinstance(definition.get("nodes"), list):
        nodes = [dict(node) for node in definition.get("nodes") or []]
        raw_edges = definition.get("edges") if isinstance(definition.get("edges"), list) else []
        raw_conditional_edges = definition.get("conditional_edges") if isinstance(definition.get("conditional_edges"), list) else []
        checkpointer_config = dict(definition.get("checkpointer_config") or definition.get("checkpoint") or {})
        entrypoint = str(definition.get("entrypoint") or "")
    else:
        nodes = [dict(node) for node in (definition or default_workflow_nodes())]
        raw_edges = []
        raw_conditional_edges = []
        checkpointer_config = {}
        entrypoint = ""
    if not nodes:
        nodes = [dict(node) for node in default_workflow_nodes()]
    node_ids = [str(node.get("id") or f"node_{index}") for index, node in enumerate(nodes)]
    valid_ids = set(node_ids)
    if entrypoint not in valid_ids:
        entrypoint = node_ids[0]
    edges = []
    for edge in raw_edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in valid_ids and target in valid_ids:
            edges.append({**edge, "source": source, "target": target})
    if not edges:
        edges = [
            {"source": node_ids[index], "target": node_ids[index + 1], "type": "linear"}
            for index in range(len(node_ids) - 1)
        ]
    conditional_edges = []
    for edge in raw_conditional_edges:
        source = str(edge.get("source") or "")
        path_map = edge.get("path_map") if isinstance(edge.get("path_map"), dict) else {}
        valid_path_map = {str(key): str(value) for key, value in path_map.items() if str(value) in valid_ids}
        if source in valid_ids and valid_path_map:
            conditional_edges.append({**edge, "source": source, "path_map": valid_path_map})
    return {
        "nodes": nodes,
        "edges": edges,
        "conditional_edges": conditional_edges,
        "entrypoint": entrypoint,
        "checkpointer_config": checkpointer_config,
    }
