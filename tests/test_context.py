"""Tests for AgentContext initialization and LLM/embeddings client creation."""

from pathlib import Path

import pytest

from sage_sanctum.context import AgentContext
from sage_sanctum.errors import ConfigurationError
from sage_sanctum.gateway.client import DirectProviderClient
from sage_sanctum.gateway.http import GatewayHttpClient
from sage_sanctum.llm.gateway_chat import GatewayChatModel
from sage_sanctum.llm.gateway_embeddings import GatewayEmbeddings
from sage_sanctum.llm.model_category import ModelCategory
from sage_sanctum.llm.model_ref import ModelRef
from sage_sanctum.llm.model_selector import ModelSelector, StaticModelSelector
from sage_sanctum.testing.mocks import MockGatewayClient


class TestAgentContext:
    def test_create_with_defaults(self, tmp_path):
        ctx = AgentContext(
            run_id="test-123",
            org_id="org-1",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=MockGatewayClient(),
            model_selector=StaticModelSelector("gpt-4o"),
        )
        assert ctx.run_id == "test-123"
        assert ctx.org_id == "org-1"

    def test_load_input(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setenv("REPO_PATH", str(repo_dir))

        ctx = AgentContext(
            run_id="test",
            org_id="org",
            work_dir=tmp_path,
            output_dir=tmp_path / "out",
            gateway_client=MockGatewayClient(),
            model_selector=StaticModelSelector("gpt-4o"),
        )
        repo_input = ctx.load_input()
        assert repo_input.path == Path(str(repo_dir))


class TestFromEnvironment:
    def test_missing_run_id(self, monkeypatch):
        monkeypatch.delenv("RUN_ID", raising=False)
        monkeypatch.setenv("ORG_ID", "test")
        with pytest.raises(ConfigurationError, match="RUN_ID"):
            AgentContext.from_environment()

    def test_missing_org_id(self, monkeypatch):
        monkeypatch.setenv("RUN_ID", "test")
        monkeypatch.delenv("ORG_ID", raising=False)
        with pytest.raises(ConfigurationError, match="ORG_ID"):
            AgentContext.from_environment()

    def test_direct_mode(self, monkeypatch):
        monkeypatch.setenv("RUN_ID", "run-1")
        monkeypatch.setenv("ORG_ID", "org-1")
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        ctx = AgentContext.from_environment()
        assert ctx.run_id == "run-1"
        assert ctx.org_id == "org-1"
        assert isinstance(ctx.gateway_client, DirectProviderClient)

    def test_no_spiffe_no_direct_raises(self, monkeypatch):
        monkeypatch.setenv("RUN_ID", "run-1")
        monkeypatch.setenv("ORG_ID", "org-1")
        monkeypatch.delenv("SAGE_SANCTUM_ALLOW_DIRECT", raising=False)
        monkeypatch.delenv("SPIFFE_JWT_PATH", raising=False)

        with pytest.raises(ConfigurationError, match="SPIFFE_JWT_PATH"):
            AgentContext.from_environment()


class TestForLocalDevelopment:
    def test_creates_direct_context(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        ctx = AgentContext.for_local_development(
            work_dir="/tmp/work",
            output_dir="/tmp/out",
            model="gpt-4o-mini",
        )
        assert ctx.run_id == "local"
        assert ctx.org_id == "local"
        assert not ctx.gateway_client.is_gateway_mode

    def test_model_selector_uses_specified_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        ctx = AgentContext.for_local_development(model="claude-3-5-sonnet")
        ref = ctx.model_selector.select(ModelCategory.ANALYSIS)
        assert ref == ModelRef(provider="anthropic", model="claude-3-5-sonnet")


# ---------------------------------------------------------------------------
# create_llm_client
# ---------------------------------------------------------------------------


class TestCreateLlmClient:
    @pytest.fixture
    def gateway_context(self, tmp_path):
        return AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=MockGatewayClient(
                is_gateway=True,
                endpoints={"openai": "unix:///run/sage/llm.sock"},
            ),
            model_selector=StaticModelSelector("openai:gpt-4o"),
        )

    def test_returns_gateway_chat_model(self, gateway_context):
        llm = gateway_context.create_llm_client(ModelCategory.ANALYSIS)

        assert isinstance(llm, GatewayChatModel)

    def test_model_ref_from_selector(self, gateway_context):
        llm = gateway_context.create_llm_client(ModelCategory.ANALYSIS)

        assert isinstance(llm, GatewayChatModel)
        assert llm.model_ref == ModelRef(provider="openai", model="gpt-4o")

    def test_uses_unix_socket_http_client(self, gateway_context):
        llm = gateway_context.create_llm_client(ModelCategory.ANALYSIS)

        assert isinstance(llm, GatewayChatModel)
        assert llm.http_client.is_unix_socket

    def test_different_categories_select_different_models(self, tmp_path):
        selector = ModelSelector({
            "triage": ["openai:gpt-4o-mini"],
            "analysis": ["anthropic:claude-3-5-sonnet-latest"],
            "reasoning": ["openai:o1"],
            "embeddings": ["openai:text-embedding-3-small"],
        })
        context = AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=MockGatewayClient(
                is_gateway=True,
                endpoints={
                    "openai": "unix:///run/sage/llm.sock",
                    "anthropic": "unix:///run/sage/llm.sock",
                },
            ),
            model_selector=selector,
        )

        triage = context.create_llm_client(ModelCategory.TRIAGE)
        analysis = context.create_llm_client(ModelCategory.ANALYSIS)

        assert isinstance(triage, GatewayChatModel)
        assert triage.model_ref.model == "gpt-4o-mini"

        assert isinstance(analysis, GatewayChatModel)
        assert analysis.model_ref.model == "claude-3-5-sonnet-latest"

    def test_direct_mode_returns_litellm(self, tmp_path, monkeypatch):
        from langchain_litellm import ChatLiteLLM

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        context = AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=DirectProviderClient(),
            model_selector=StaticModelSelector("openai:gpt-4o"),
        )

        llm = context.create_llm_client(ModelCategory.ANALYSIS)
        assert isinstance(llm, ChatLiteLLM)


# ---------------------------------------------------------------------------
# create_embeddings_client
# ---------------------------------------------------------------------------


class TestCreateEmbeddingsClient:
    @pytest.fixture
    def gateway_context(self, tmp_path):
        return AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=MockGatewayClient(
                is_gateway=True,
                endpoints={"openai": "unix:///run/sage/llm.sock"},
            ),
            model_selector=StaticModelSelector("openai:gpt-4o"),
        )

    def test_returns_gateway_embeddings(self, gateway_context):
        emb = gateway_context.create_embeddings_client()

        assert isinstance(emb, GatewayEmbeddings)

    def test_uses_unix_socket(self, gateway_context):
        emb = gateway_context.create_embeddings_client()

        assert isinstance(emb, GatewayEmbeddings)
        assert emb._http_client.is_unix_socket

    def test_default_model(self, gateway_context):
        emb = gateway_context.create_embeddings_client()

        assert emb._model == "text-embedding-3-small"

    def test_custom_model(self, gateway_context):
        emb = gateway_context.create_embeddings_client(model="text-embedding-3-large")

        assert emb._model == "text-embedding-3-large"

    def test_tcp_endpoint(self, tmp_path):
        context = AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=MockGatewayClient(
                is_gateway=True,
                endpoints={"openai": "http://gateway:8080/v1"},
            ),
            model_selector=StaticModelSelector("openai:gpt-4o"),
        )

        emb = context.create_embeddings_client()

        assert isinstance(emb, GatewayEmbeddings)
        assert not emb._http_client.is_unix_socket

    def test_direct_mode_returns_openai_embeddings(self, tmp_path, monkeypatch):
        pytest.importorskip("langchain_openai")
        from langchain_openai import OpenAIEmbeddings

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        context = AgentContext(
            run_id="run_test",
            org_id="org_test",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            gateway_client=DirectProviderClient(),
            model_selector=StaticModelSelector("openai:gpt-4o"),
        )

        emb = context.create_embeddings_client()
        assert isinstance(emb, OpenAIEmbeddings)
