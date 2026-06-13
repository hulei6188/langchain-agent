from __future__ import annotations

import json
import urllib.parse


def dict_value(value) -> dict:
    return value if isinstance(value, dict) else {}


def safe_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def preview(value, limit: int = 500) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return value[:limit]


def url_with_query(url: str, params: dict) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in params.items() if value is not None})
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))
