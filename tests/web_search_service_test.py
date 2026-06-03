from core.config import get_settings
from core.services import web_search


def test_tavily_search_provider(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    get_settings.cache_clear()

    def fake_fetch(query, *, limit, timeout_seconds):
        assert query == "langchain agent"
        assert limit == 2
        assert timeout_seconds == 5
        return {
            "results": [
                {"title": "LangChain", "url": "https://example.com/a", "content": "Agent framework"},
                {"title": "Duplicate", "url": "https://example.com/a", "content": "Skip duplicate"},
                {"title": "Docs", "url": "https://example.com/docs", "content": "Reference"},
            ]
        }

    monkeypatch.setattr(web_search, "_fetch_tavily_json", fake_fetch)

    result = web_search.search_web("  langchain   agent  ", top_k=2, timeout_seconds=5)

    assert result["provider"] == "tavily"
    assert result["query"] == "langchain agent"
    assert result["items"] == [
        {"title": "LangChain", "url": "https://example.com/a", "snippet": "Agent framework"},
        {"title": "Docs", "url": "https://example.com/docs", "snippet": "Reference"},
    ]


def test_serpapi_search_provider(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-test")
    get_settings.cache_clear()

    def fake_fetch(query, *, limit, timeout_seconds):
        assert query == "python search"
        assert limit == 1
        assert timeout_seconds == 7
        return {
            "organic_results": [
                {"title": "Python", "link": "https://example.com/python", "snippet": "Programming language"},
                {"title": "Ignored", "link": "https://example.com/ignored", "snippet": "Over limit"},
            ]
        }

    monkeypatch.setattr(web_search, "_fetch_serpapi_json", fake_fetch)

    result = web_search.search_web("python search", top_k=1, timeout_seconds=7)

    assert result["provider"] == "serpapi"
    assert result["items"] == [
        {"title": "Python", "url": "https://example.com/python", "snippet": "Programming language"},
    ]


def test_web_search_status_requires_api_key(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    get_settings.cache_clear()

    status = web_search.web_search_status()

    assert status["provider"] == "serpapi"
    assert status["requires_api_key"] is True
    assert status["configured"] is False
