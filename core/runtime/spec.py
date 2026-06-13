from __future__ import annotations

from typing import TypedDict


class WorkflowGraphSpec(TypedDict):
    nodes: list[dict]
    edges: list[dict]
    checkpoint: dict


def default_workflow() -> list[dict]:
    return [
        {"id": "start", "type": "Start", "name": "接收用户输入", "config": {}},
        {"id": "knowledge", "type": "Knowledge", "name": "检索绑定知识库", "config": {"top_k": 4}},
        {"id": "tool", "type": "Tool", "name": "调用绑定工具", "config": {"tools": []}},
        {"id": "llm", "type": "LLM", "name": "生成候选回答", "config": {}},
        {"id": "answer", "type": "Answer", "name": "输出最终回答", "config": {}},
    ]


def workflow_graph_spec(definition: list[dict] | dict | None) -> WorkflowGraphSpec:
    if isinstance(definition, dict) and isinstance(definition.get("nodes"), list):
        nodes = [dict(node) for node in definition.get("nodes") or []]
        raw_edges = definition.get("edges") if isinstance(definition.get("edges"), list) else []
        checkpoint = dict(definition.get("checkpoint") or {})
    else:
        nodes = [dict(node) for node in (definition or default_workflow())]
        raw_edges = []
        checkpoint = {}
    if not nodes:
        nodes = [dict(node) for node in default_workflow()]
    edges = []
    node_ids = [str(node.get("id") or f"node_{index}") for index, node in enumerate(nodes)]
    valid_ids = set(node_ids)
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
    return {"nodes": nodes, "edges": edges, "checkpoint": checkpoint}
