from __future__ import annotations

import json


def merge_stream_tool_call_chunks(builders: dict[int, dict], chunks: list[dict]) -> None:
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        try:
            index = int(chunk.get("index")) if chunk.get("index") is not None else len(builders)
        except (TypeError, ValueError):
            index = len(builders)
        call = builders.setdefault(index, {"id": "", "name": "", "args": ""})
        if chunk.get("id"):
            call["id"] = str(chunk.get("id"))
        name_part = chunk.get("name")
        if name_part:
            name_part = str(name_part)
            current_name = call.get("name") or ""
            if not current_name or name_part.startswith(current_name):
                call["name"] = name_part
            elif not current_name.endswith(name_part):
                call["name"] = current_name + name_part
        if chunk.get("args") is not None:
            call["args"] = (call.get("args") or "") + str(chunk.get("args"))


def finalize_stream_tool_calls(builders: dict[int, dict]) -> list[dict]:
    calls = []
    for index in sorted(builders):
        call = builders[index]
        name = call.get("name") or ""
        if not name:
            continue
        raw_args = call.get("args") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(args, dict):
                args = {"input": args}
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {"input": str(raw_args)}
        calls.append(
            {
                "id": call.get("id") or f"call_{index}",
                "name": name,
                "args": args,
            }
        )
    return calls
