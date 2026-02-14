"""Tests for GatewayEmbeddings and create_embeddings_for_gateway."""

import json
from unittest.mock import MagicMock

import pytest

from sage_sanctum.auth.credentials import GatewayCredentials
from sage_sanctum.errors import GatewayError
from sage_sanctum.gateway.http import GatewayHttpClient, HttpResponse
from sage_sanctum.llm.gateway_embeddings import (
    GatewayEmbeddings,
    create_embeddings_for_gateway,
)


def _make_embedding_response(embeddings: list[list[float]]) -> str:
    """Build an OpenAI-format embeddings JSON response."""
    return json.dumps({
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    })


class TestGatewayEmbeddings:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    @pytest.fixture
    def mock_http_client(self):
        client = MagicMock()
        client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=_make_embedding_response([[0.1, 0.2, 0.3]]),
        )
        return client

    def test_embed_query_returns_vector(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = emb.embed_query("hello world")

        assert result == [0.1, 0.2, 0.3]

    def test_embed_documents_returns_vectors(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=_make_embedding_response([[0.1, 0.2], [0.3, 0.4]]),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = emb.embed_documents(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    def test_embed_documents_empty_list(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = emb.embed_documents([])

        assert result == []
        mock_http_client.request.assert_not_called()

    def test_injects_auth_headers(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        emb.embed_query("test")

        call_args = mock_http_client.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer test-svid"
        assert headers["Txn-Token"] == "test-trat"
        assert headers["X-Provider"] == "openai"

    def test_posts_to_v1_embeddings(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        emb.embed_query("test")

        call_args = mock_http_client.request.call_args
        assert call_args.kwargs.get("method") == "POST"
        assert call_args.kwargs.get("path") == "/v1/embeddings"

    def test_sends_model_and_input_in_body(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=_make_embedding_response([[0.1, 0.2], [0.3, 0.4]]),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        emb.embed_documents(["hello", "world"])

        call_args = mock_http_client.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["hello", "world"]

    def test_refreshes_credentials_each_call(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        emb.embed_query("first")
        emb.embed_query("second")

        assert mock_gateway_client.get_credentials.call_count == 2

    def test_gateway_error_on_non_200(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=401,
            headers={},
            data='{"error": "unauthorized"}',
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="401"):
            emb.embed_query("test")

    def test_gateway_error_on_invalid_json(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={},
            data="not json",
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="Invalid JSON"):
            emb.embed_query("test")

    def test_gateway_error_on_empty_data(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={},
            data=json.dumps({"data": []}),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="no embeddings"):
            emb.embed_query("test")

    def test_gateway_error_on_count_mismatch(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={},
            data=_make_embedding_response([[0.1, 0.2]]),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="Expected 3.*got 1"):
            emb.embed_documents(["a", "b", "c"])

    def test_request_exception_wrapped(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.side_effect = RuntimeError("socket reset")

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="Embeddings request failed"):
            emb.embed_query("test")

    def test_large_batch_embeddings(self, mock_gateway_client, mock_http_client):
        """Embedding 150 texts (typical knowledge base chunk count) in a single request."""
        n = 150
        texts = [f"security document chunk {i}" for i in range(n)]
        embeddings = [[float(i) / n, float(i + 1) / n] for i in range(n)]

        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "data": [
                    {"embedding": emb, "index": i}
                    for i, emb in enumerate(embeddings)
                ],
                "model": "text-embedding-3-small",
            }),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = emb.embed_documents(texts)

        assert len(result) == n
        assert result[0] == embeddings[0]
        assert result[n - 1] == embeddings[n - 1]

        # Verify all texts were sent in a single request
        call_args = mock_http_client.request.call_args
        body = call_args.kwargs.get("body", {})
        assert len(body["input"]) == n
        assert mock_http_client.request.call_count == 1

    def test_out_of_order_indices_sorted(self, mock_gateway_client, mock_http_client):
        """Embeddings returned out-of-order by index are sorted correctly."""
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "data": [
                    {"embedding": [0.9, 0.8], "index": 1},
                    {"embedding": [0.1, 0.2], "index": 0},
                ],
                "model": "text-embedding-3-small",
            }),
        )

        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = emb.embed_documents(["first", "second"])

        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.9, 0.8]


class TestProviderParameter:
    """Tests for the configurable provider parameter."""

    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    @pytest.fixture
    def mock_http_client(self):
        client = MagicMock()
        client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=_make_embedding_response([[0.1, 0.2, 0.3]]),
        )
        return client

    def test_default_provider_is_openai(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="text-embedding-3-small",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        emb.embed_query("test")

        call_args = mock_http_client.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["X-Provider"] == "openai"

    def test_custom_provider(self, mock_gateway_client, mock_http_client):
        emb = GatewayEmbeddings(
            model="voyage-3",
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            provider="voyage",
        )
        emb.embed_query("test")

        call_args = mock_http_client.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["X-Provider"] == "voyage"


class TestCreateEmbeddingsForGateway:
    def test_gateway_mode_unix_socket(self):
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "unix:///run/sage/llm.sock"

        result = create_embeddings_for_gateway("text-embedding-3-small", client)

        assert isinstance(result, GatewayEmbeddings)
        assert result._http_client.is_unix_socket

    def test_gateway_mode_tcp_endpoint(self):
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "http://gateway:8080/v1"

        result = create_embeddings_for_gateway("text-embedding-3-small", client)

        assert isinstance(result, GatewayEmbeddings)
        assert not result._http_client.is_unix_socket

    def test_gateway_mode_custom_provider(self):
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "unix:///run/sage/llm.sock"

        result = create_embeddings_for_gateway(
            "voyage-3", client, provider="voyage"
        )

        assert isinstance(result, GatewayEmbeddings)
        assert result._provider == "voyage"
        client.get_endpoint.assert_called_once_with("voyage")

    def test_direct_mode_returns_openai_embeddings(self, monkeypatch):
        pytest.importorskip("langchain_openai")
        from langchain_openai import OpenAIEmbeddings

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = MagicMock()
        client.is_gateway_mode = False

        result = create_embeddings_for_gateway("text-embedding-3-small", client)

        assert isinstance(result, OpenAIEmbeddings)


class TestMockEmbeddings:
    def test_embed_query_returns_vector(self):
        from sage_sanctum.testing.mocks import MockEmbeddings

        emb = MockEmbeddings(dimension=4)
        result = emb.embed_query("hello")

        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_returns_list(self):
        from sage_sanctum.testing.mocks import MockEmbeddings

        emb = MockEmbeddings(dimension=4)
        result = emb.embed_documents(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 4

    def test_deterministic(self):
        from sage_sanctum.testing.mocks import MockEmbeddings

        emb = MockEmbeddings(dimension=4)
        r1 = emb.embed_query("hello")
        r2 = emb.embed_query("hello")

        assert r1 == r2

    def test_records_calls(self):
        from sage_sanctum.testing.mocks import MockEmbeddings

        emb = MockEmbeddings()
        emb.embed_documents(["a", "b"])
        emb.embed_query("c")

        assert len(emb.calls) == 2
        assert emb.calls[0] == ["a", "b"]
        assert emb.calls[1] == ["c"]
