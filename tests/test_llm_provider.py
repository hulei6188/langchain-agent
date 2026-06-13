from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import StructuredTool

from core.config import get_settings
from core.integrations import chat_models as chat_models_module
from core.integrations.llm import OpenAICompatibleProvider
from core.integrations.model_clients import OpenAICompatibleEmbeddings, OpenAICompatibleReranker


class FakeChatOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.bound_tools = None
        self.tool_choice = None
        self.invoke_calls = []
        self.stream_calls = []
        FakeChatOpenAI.instances.append(self)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages, *, stop=None):
        self.invoke_calls.append({"messages": messages, "stop": stop})
        return AIMessage(content="native response")

    def stream(self, messages, *, stop=None):
        self.stream_calls.append({"messages": messages, "stop": stop})
        yield AIMessageChunk(content="native ")
        yield AIMessageChunk(content="stream")


def _sample_tool(query: str) -> str:
    return query


def test_provider_delegates_chat_to_chat_openai(monkeypatch):
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(chat_models_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    get_settings.cache_clear()

    try:
        provider = OpenAICompatibleProvider()
        response = provider.invoke(
            [HumanMessage(content="hello")],
            model="gpt-5-mini",
            runtime_config={"base_url": "https://gateway.example/v1", "api_key": "runtime-key"},
            temperature=0.2,
            thinking_enabled=True,
            tools=[
                StructuredTool.from_function(
                    _sample_tool,
                    name="sample_tool",
                    description="sample tool",
                )
            ],
        )
    finally:
        get_settings.cache_clear()

    assert response.content == "native response"
    instance = FakeChatOpenAI.instances[0]
    assert instance.kwargs["api_key"] == "runtime-key"
    assert instance.kwargs["base_url"] == "https://gateway.example/v1"
    assert instance.kwargs["model"] == "gpt-5-mini"
    assert instance.kwargs["temperature"] == 0.2
    assert instance.kwargs["reasoning_effort"] == "high"
    assert instance.tool_choice == "auto"
    assert instance.bound_tools[0].name == "sample_tool"


def test_provider_delegates_stream_to_chat_openai(monkeypatch):
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(chat_models_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    get_settings.cache_clear()

    try:
        provider = OpenAICompatibleProvider()
        chunks = list(
            provider.stream(
                [HumanMessage(content="hello")],
                model="qwen-plus",
                runtime_config={"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
                thinking_enabled=False,
            )
        )
    finally:
        get_settings.cache_clear()

    assert "".join(chunk.content for chunk in chunks) == "native stream"
    instance = FakeChatOpenAI.instances[0]
    assert instance.kwargs["streaming"] is True
    assert instance.kwargs["extra_body"] == {"enable_thinking": False}


def test_chat_model_factory_reasoning_kwargs():
    assert chat_models_module.chat_model_kwargs(
        api_base="https://gateway.example/v1",
        model="gpt-5-mini",
        thinking_enabled=True,
    ) == {"reasoning_effort": "high"}
    assert chat_models_module.chat_model_kwargs(
        api_base="https://api.deepseek.com",
        model="deepseek-reasoner",
        thinking_enabled=False,
    ) == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert chat_models_module.requires_reasoning_replay(
        api_base="https://api.deepseek.com",
        model="deepseek-chat",
    ) is True


def test_embedding_client_mock_mode_is_deterministic(monkeypatch):
    monkeypatch.setenv("AGENTBASE_MOCK_LLM", "true")
    get_settings.cache_clear()
    try:
        embeddings = OpenAICompatibleEmbeddings()
        first = embeddings.embed_query("hello")
        second = embeddings.embed_query("hello")
    finally:
        get_settings.cache_clear()

    assert first == second
    assert len(first) == 32
    assert embeddings.last_mock is True


def test_reranker_mock_mode_scores_query_terms(monkeypatch):
    monkeypatch.setenv("AGENTBASE_MOCK_LLM", "true")
    get_settings.cache_clear()
    try:
        ranked = OpenAICompatibleReranker().rerank(
            "alpha",
            ["beta text", "alpha text"],
            top_n=1,
        )
    finally:
        get_settings.cache_clear()

    assert ranked == [{"index": 1, "relevance_score": 1.0}]
