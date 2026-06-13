from __future__ import annotations

import re


def skill_manifest(item: dict) -> dict:
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return {
        "id": item["id"],
        "name": item["name"],
        "description": item.get("description") or "",
        "category": item.get("category") or "general",
        "tags": tags,
        "activation_mode": item.get("activation_mode") or "auto",
        "priority": item.get("priority", 0),
    }


def dedupe_skill_bindings(items: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for item in items:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        result.append(item)
    return result


def skill_selection_text(
    runtime,
    context: dict,
    chat_session,
    active_skill_names: list[str],
    *,
    history_limit: int,
) -> str:
    parts = [context.get("input") or ""]
    if chat_session.title and chat_session.title != "新对话":
        parts.append(f"会话标题：{chat_session.title}")
    history = context.get("history_messages") or []
    for item in history[-history_limit:]:
        role = item.get("role") or ""
        content = str(item.get("content") or "")
        if role in {"user", "assistant"} and content.strip():
            parts.append(f"{role}: {content[:1200]}")
    if context.get("memory_summary"):
        parts.append(f"会话记忆：{str(context.get('memory_summary'))[:2000]}")
    if active_skill_names:
        parts.append("已加载技能：" + "、".join(active_skill_names))
    return "\n".join(part for part in parts if str(part).strip())


def skill_explicitly_requested(item: dict, user_text: str) -> bool:
    text = str(user_text or "").lower()
    name = str(item.get("name") or "").strip().lower()
    if not name:
        return False
    return name in text or f"@{name}" in text or f"#{name}" in text


def score_runtime_skills(selection_text: str, skills: list[dict]) -> list[tuple[dict, float]]:
    text = normalize_skill_text(selection_text)
    scored = [(item, skill_match_score(text, item)) for item in skills]
    return sorted(scored, key=lambda pair: (pair[1], pair[0].get("priority", 0)), reverse=True)


def skill_match_score(normalized_text: str, item: dict) -> float:
    if not normalized_text.strip():
        return 0.0
    score = 0.0
    name = normalize_skill_text(item.get("name") or "")
    if name and name in normalized_text:
        score += 0.65
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        tag_text = normalize_skill_text(tag)
        if len(tag_text) >= 2 and tag_text in normalized_text:
            score += 0.18
    category = normalize_skill_text(item.get("category") or "")
    if len(category) >= 2 and category in normalized_text:
        score += 0.08

    terms = skill_terms(item)
    if terms:
        matched_weight = sum(weight for term, weight in terms if term in normalized_text)
        total_weight = sum(weight for _, weight in terms)
        if total_weight > 0:
            score += 0.55 * min(matched_weight / total_weight, 1.0)
    return min(score, 1.0)


def normalize_skill_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def skill_terms(item: dict) -> list[tuple[str, float]]:
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    weighted_fields = [
        (item.get("name") or "", 1.0),
        (" ".join(str(tag) for tag in tags), 0.9),
        (item.get("category") or "", 0.35),
        (item.get("description") or "", 0.65),
    ]
    terms: dict[str, float] = {}
    for text, weight in weighted_fields:
        normalized = normalize_skill_text(text)
        for token in re.findall(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", normalized):
            if len(token) < 2:
                continue
            terms[token] = max(terms.get(token, 0.0), weight)
            if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
                for index in range(0, len(token) - 1):
                    terms[token[index:index + 2]] = max(terms.get(token[index:index + 2], 0.0), weight * 0.35)
    return list(terms.items())


def skill_manifest_text(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = [
        "可用技能清单（渐进式披露）：",
        "只有“本轮已加载技能”的完整规则已进入上下文；auto 技能需达阈值才加载，manual 技能需用户明确点名或通过 load_skill 加载，always 技能每轮加载。",
    ]
    for skill in skills[:80]:
        tags = "、".join(skill.get("tags") or [])
        desc = skill.get("description") or "无描述"
        lines.append(
            f"- [{skill.get('activation_mode')}] #{skill.get('id')} {skill.get('name')}: {desc}"
            + (f"；标签：{tags}" if tags else "")
        )
    if len(skills) > 80:
        lines.append(f"... 还有 {len(skills) - 80} 个技能未列出")
    return "\n".join(lines)


def loaded_skill_text(skills: list[dict]) -> str:
    if not skills:
        return "本轮已加载技能：无"
    lines = ["本轮已加载技能："]
    for skill in skills:
        mode = skill.get("activation_mode") or "auto"
        score = skill.get("score")
        score_text = f"，score={score}" if score is not None else ""
        lines.append(f"- [{mode}] {skill.get('name')}#{skill.get('id')}{score_text}")
    return "\n".join(lines)
