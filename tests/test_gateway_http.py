"""Tests for GatewayHttpClient HTTP parsing and socket communication."""

import json
import random
import socket
import threading
import time
from pathlib import Path

import pytest

from sage_sanctum.errors import GatewayError, GatewayUnavailableError, RateLimitError
from sage_sanctum.gateway.http import GatewayHttpClient

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestGatewayHttpClientInit:
    def test_with_socket_path(self, tmp_path):
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")
        assert client.is_unix_socket

    def test_with_host_and_port(self):
        client = GatewayHttpClient(host="localhost", port=8080)
        assert not client.is_unix_socket

    def test_neither_socket_nor_host_raises(self):
        with pytest.raises(GatewayError, match="Either socket_path or host"):
            GatewayHttpClient()

    def test_custom_timeout(self, tmp_path):
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock", timeout=30)
        assert client._timeout == 30

    def test_string_socket_path_converted_to_path(self):
        client = GatewayHttpClient(socket_path="/tmp/test.sock")
        assert client._socket_path is not None
        assert str(client._socket_path) == "/tmp/test.sock"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _make_client(self, tmp_path):
        return GatewayHttpClient(socket_path=tmp_path / "gw.sock")

    def test_valid_200_response(self, tmp_path):
        client = self._make_client(tmp_path)
        raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}"
        resp = client._parse_response(raw)

        assert resp.status == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.data == '{"ok":true}'

    def test_status_429(self, tmp_path):
        client = self._make_client(tmp_path)
        raw = b"HTTP/1.1 429 Too Many Requests\r\n\r\nrate limited"
        resp = client._parse_response(raw)

        assert resp.status == 429
        assert resp.data == "rate limited"

    def test_status_502(self, tmp_path):
        client = self._make_client(tmp_path)
        raw = b"HTTP/1.1 502 Bad Gateway\r\n\r\n"
        resp = client._parse_response(raw)

        assert resp.status == 502
        assert resp.data == ""

    def test_multiple_headers(self, tmp_path):
        client = self._make_client(tmp_path)
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"X-Request-Id: abc123\r\n"
            b"Cache-Control: no-cache\r\n"
            b"\r\n"
            b"body"
        )
        resp = client._parse_response(raw)

        assert resp.status == 200
        assert resp.headers["content-type"] == "text/plain"
        assert resp.headers["x-request-id"] == "abc123"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.data == "body"

    def test_header_keys_lowercased(self, tmp_path):
        client = self._make_client(tmp_path)
        raw = b"HTTP/1.1 200 OK\r\nX-CUSTOM-Header: value\r\n\r\n"
        resp = client._parse_response(raw)

        assert "x-custom-header" in resp.headers
        assert resp.headers["x-custom-header"] == "value"

    def test_header_value_with_colon(self, tmp_path):
        """Header values can contain colons (e.g. timestamps)."""
        client = self._make_client(tmp_path)
        raw = b"HTTP/1.1 200 OK\r\nDate: Mon, 01 Jan 2024 12:00:00 GMT\r\n\r\n"
        resp = client._parse_response(raw)

        assert resp.headers["date"] == "Mon, 01 Jan 2024 12:00:00 GMT"

    def test_empty_response_raises(self, tmp_path):
        client = self._make_client(tmp_path)
        with pytest.raises(GatewayError, match="Empty response"):
            client._parse_response(b"")

    def test_no_separator_raises(self, tmp_path):
        client = self._make_client(tmp_path)
        with pytest.raises(GatewayError, match="no header/body separator"):
            client._parse_response(b"HTTP/1.1 200 OK\r\nContent-Type: text")

    def test_malformed_status_line_raises(self, tmp_path):
        client = self._make_client(tmp_path)
        with pytest.raises(GatewayError, match="Malformed status line"):
            client._parse_response(b"GARBAGE\r\n\r\nbody")

    def test_json_body_preserved(self, tmp_path):
        """Verify JSON body is preserved exactly."""
        client = self._make_client(tmp_path)
        body = json.dumps({"choices": [{"message": {"content": "hello"}}]})
        raw = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{body}".encode()
        resp = client._parse_response(raw)

        parsed = json.loads(resp.data)
        assert parsed["choices"][0]["message"]["content"] == "hello"


# ---------------------------------------------------------------------------
# _is_response_complete
# ---------------------------------------------------------------------------


class TestIsResponseComplete:
    def _make_client(self, tmp_path):
        return GatewayHttpClient(socket_path=tmp_path / "gw.sock")

    def test_incomplete_no_separator(self, tmp_path):
        client = self._make_client(tmp_path)
        assert client._is_response_complete(b"HTTP/1.1 200 OK\r\n") is False

    def test_complete_with_content_length_exact(self, tmp_path):
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nbody"
        assert client._is_response_complete(response) is True

    def test_incomplete_with_content_length_short(self, tmp_path):
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial"
        assert client._is_response_complete(response) is False

    def test_complete_with_content_length_excess(self, tmp_path):
        """More data than Content-Length is still considered complete."""
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nbody"
        assert client._is_response_complete(response) is True

    def test_complete_with_zero_content_length(self, tmp_path):
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
        assert client._is_response_complete(response) is True

    def test_chunked_incomplete(self, tmp_path):
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nbody\r\n"
        assert client._is_response_complete(response) is False

    def test_chunked_complete(self, tmp_path):
        client = self._make_client(tmp_path)
        response = (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"4\r\nbody\r\n0\r\n\r\n"
        )
        assert client._is_response_complete(response) is True

    def test_no_content_length_no_chunked_is_complete(self, tmp_path):
        """Without Content-Length or chunked, rely on EOF (return True)."""
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nsome data"
        assert client._is_response_complete(response) is True

    def test_content_length_header_case_insensitive(self, tmp_path):
        client = self._make_client(tmp_path)
        response = b"HTTP/1.1 200 OK\r\ncontent-length: 3\r\n\r\nabc"
        assert client._is_response_complete(response) is True


# ---------------------------------------------------------------------------
# request (integration with mocked _send_raw)
# ---------------------------------------------------------------------------


class TestRequest:
    def _make_client(self, tmp_path):
        return GatewayHttpClient(socket_path=tmp_path / "gw.sock")

    def test_get_request(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        captured_data = {}

        def mock_send_raw(data: bytes) -> bytes:
            captured_data["raw"] = data
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/trat")

        assert resp.status == 200
        assert resp.data == "ok"
        raw = captured_data["raw"].decode()
        assert raw.startswith("GET /trat HTTP/1.1\r\n")
        assert "Host: gateway.local" in raw

    def test_post_with_dict_body(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        captured_data = {}

        def mock_send_raw(data: bytes) -> bytes:
            captured_data["raw"] = data
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("POST", "/v1/chat", body={"model": "gpt-4o"})

        assert resp.status == 200
        raw = captured_data["raw"].decode()
        assert "Content-Type: application/json" in raw
        assert "Content-Length:" in raw
        assert '"model": "gpt-4o"' in raw

    def test_post_with_string_body(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("POST", "/v1/chat", body="raw body text")

        assert resp.status == 200

    def test_custom_headers_merged(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        captured_data = {}

        def mock_send_raw(data: bytes) -> bytes:
            captured_data["raw"] = data
            return b"HTTP/1.1 200 OK\r\n\r\n"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        client.request("GET", "/test", headers={"Authorization": "Bearer tok"})

        raw = captured_data["raw"].decode()
        assert "Authorization: Bearer tok" in raw
        assert "Host: gateway.local" in raw

    def test_gateway_error_passthrough(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            raise GatewayUnavailableError("socket gone")

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        with pytest.raises(GatewayUnavailableError, match="socket gone"):
            client.request("GET", "/test")

    def test_unexpected_error_wrapped(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            raise RuntimeError("boom")

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        with pytest.raises(GatewayError, match="Gateway request failed"):
            client.request("GET", "/test")

    def test_no_body_omits_content_length(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        captured_data = {}

        def mock_send_raw(data: bytes) -> bytes:
            captured_data["raw"] = data
            return b"HTTP/1.1 200 OK\r\n\r\n"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        client.request("GET", "/test")

        raw = captured_data["raw"].decode()
        assert "Content-Length" not in raw
        assert "Content-Type" not in raw


# ---------------------------------------------------------------------------
# _send_raw with real Unix socket
# ---------------------------------------------------------------------------


class TestSendRawUnixSocket:
    def _serve_once(self, sock_path, response_bytes):
        """Start a Unix socket server that serves one response then exits."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        server.settimeout(5)

        def handler():
            conn, _ = server.accept()
            conn.recv(65536)
            conn.sendall(response_bytes)
            conn.close()
            server.close()

        t = threading.Thread(target=handler, daemon=True)
        t.start()
        return t

    def test_roundtrip_over_unix_socket(self):
        # Use a short path to avoid AF_UNIX 104-byte limit on macOS
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
            sock_path = Path(short_dir) / "t.sock"
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"status\":\"ok\"}"

            self._serve_once(sock_path, response)

            client = GatewayHttpClient(socket_path=sock_path, timeout=5)
            resp = client.request("GET", "/health")

            assert resp.status == 200
            assert json.loads(resp.data) == {"status": "ok"}

    def test_post_json_roundtrip_over_unix_socket(self):
        """POST with JSON body and JSON response over a real AF_UNIX socket."""
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
            sock_path = Path(short_dir) / "t.sock"

            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(sock_path))
            server.listen(1)
            server.settimeout(5)

            captured = {}

            def handler():
                conn, _ = server.accept()
                raw_request = conn.recv(65536)
                captured["request"] = raw_request

                # Build response
                body = json.dumps({
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                    "model": "text-embedding-3-small",
                }).encode()
                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"\r\n" + body
                )
                conn.sendall(response)
                conn.close()
                server.close()

            t = threading.Thread(target=handler, daemon=True)
            t.start()

            client = GatewayHttpClient(socket_path=sock_path, timeout=5)
            resp = client.request(
                method="POST",
                path="/v1/embeddings",
                headers={"Authorization": "Bearer test-jwt", "Txn-Token": "test-trat"},
                body={"model": "text-embedding-3-small", "input": ["hello"]},
            )

            t.join(timeout=5)

            # Verify response
            assert resp.status == 200
            parsed = json.loads(resp.data)
            assert parsed["data"][0]["embedding"] == [0.1, 0.2, 0.3]

            # Verify request was well-formed
            raw = captured["request"].decode()
            assert raw.startswith("POST /v1/embeddings HTTP/1.1\r\n")
            assert "Content-Type: application/json" in raw
            assert "Authorization: Bearer test-jwt" in raw
            assert "Txn-Token: test-trat" in raw
            # Body is after the header separator
            body_start = raw.index("\r\n\r\n") + 4
            request_body = json.loads(raw[body_start:])
            assert request_body["model"] == "text-embedding-3-small"
            assert request_body["input"] == ["hello"]

    def test_connection_refused_raises(self, tmp_path):
        sock_path = tmp_path / "nonexistent.sock"
        client = GatewayHttpClient(socket_path=sock_path, timeout=1)

        with pytest.raises(GatewayUnavailableError, match="Cannot connect"):
            client.request("GET", "/health")

    def test_tcp_connection_refused_raises(self):
        client = GatewayHttpClient(host="127.0.0.1", port=19999, timeout=1)
        with pytest.raises(GatewayUnavailableError, match="Cannot connect"):
            client.request("GET", "/health")


# ---------------------------------------------------------------------------
# _send_raw read loop with chunked recv() (integration)
# ---------------------------------------------------------------------------


class TestSendRawChunkedRecv:
    """Verify _send_raw assembles multi-chunk responses via the read loop."""

    def test_content_length_response_in_multiple_chunks(self, tmp_path, monkeypatch):
        """Response arriving in multiple recv() calls is assembled correctly."""
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")

        full_response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"status\":\"ok\"}"
        # Split into 3 chunks: headers partial, headers+body partial, body rest
        chunks = [full_response[:20], full_response[20:40], full_response[40:]]

        class FakeSocket:
            def __init__(self):
                self._chunk_iter = iter(chunks)

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return next(self._chunk_iter, b"")

            def close(self):
                pass

        fake_sock = FakeSocket()
        monkeypatch.setattr(socket, "socket", lambda *a, **kw: fake_sock)

        raw = client._send_raw(b"GET /health HTTP/1.1\r\nHost: gateway.local\r\n\r\n")
        assert raw == full_response

    def test_recv_assembles_until_content_length_satisfied(self, tmp_path, monkeypatch):
        """Read loop keeps reading until Content-Length bytes are received."""
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")

        headers = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n"
        # Send headers first, then body byte-by-byte
        chunks = [headers, b"h", b"e", b"l", b"l", b"o"]

        class FakeSocket:
            def __init__(self):
                self._chunk_iter = iter(chunks)

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return next(self._chunk_iter, b"")

            def close(self):
                pass

        fake_sock = FakeSocket()
        monkeypatch.setattr(socket, "socket", lambda *a, **kw: fake_sock)

        raw = client._send_raw(b"GET / HTTP/1.1\r\n\r\n")
        parsed = client._parse_response(raw)
        assert parsed.status == 200
        assert parsed.data == "hello"

    def test_eof_before_content_length_returns_partial(self, tmp_path, monkeypatch):
        """If connection closes before Content-Length is satisfied, return what we have."""
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")

        # Content-Length says 100 but we only get 3 bytes before EOF
        chunks = [b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nabc"]

        class FakeSocket:
            def __init__(self):
                self._chunk_iter = iter(chunks)

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return next(self._chunk_iter, b"")

            def close(self):
                pass

        fake_sock = FakeSocket()
        monkeypatch.setattr(socket, "socket", lambda *a, **kw: fake_sock)

        raw = client._send_raw(b"GET / HTTP/1.1\r\n\r\n")
        # The read loop exits on empty recv (EOF), returns what it got
        assert b"abc" in raw


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def _make_client(self, tmp_path, max_retries=3, retry_backoff=0.5):
        return GatewayHttpClient(
            socket_path=tmp_path / "gw.sock",
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def test_retries_on_502(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert resp.data == "ok"
        assert call_count == 3

    def test_retries_on_503(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert call_count == 2

    def test_retries_on_504(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"HTTP/1.1 504 Gateway Timeout\r\nContent-Length: 0\r\n\r\n"
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert call_count == 2

    def test_returns_last_response_when_retries_exhausted_on_502(self, tmp_path, monkeypatch):
        """When all retries are exhausted on a non-429 retryable status, return the response."""
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 11\r\n\r\nbad gateway"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 502
        assert resp.data == "bad gateway"

    def test_exponential_backoff_delays(self, tmp_path, monkeypatch):
        """Verify retry delays follow exponential backoff pattern."""
        client = self._make_client(tmp_path, retry_backoff=1.0)

        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda d: sleep_calls.append(d))
        # Suppress random jitter for predictable testing
        monkeypatch.setattr(random, "random", lambda: 0.0)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        client.request("GET", "/test")

        # With retry_backoff=1.0 and random()=0.0:
        # attempt 0 -> 1.0 * (2^0) + 0.0 = 1.0
        # attempt 1 -> 1.0 * (2^1) + 0.0 = 2.0
        # attempt 2 -> 1.0 * (2^2) + 0.0 = 4.0
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0

    def test_max_retries_zero_disables_retries(self, tmp_path, monkeypatch):
        """With max_retries=0, no retries occur."""
        client = self._make_client(tmp_path, max_retries=0)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            return b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 502
        assert call_count == 1

    def test_retries_on_gateway_unavailable_error(self, tmp_path, monkeypatch):
        """Connection errors (GatewayUnavailableError) trigger retries too."""
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise GatewayUnavailableError("Cannot connect")
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert call_count == 3

    def test_gateway_unavailable_raises_after_retries_exhausted(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def mock_send_raw(data: bytes) -> bytes:
            raise GatewayUnavailableError("Cannot connect")

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        with pytest.raises(GatewayUnavailableError, match="Cannot connect"):
            client.request("GET", "/test")

    def test_non_retryable_status_returns_immediately(self, tmp_path, monkeypatch):
        """Status codes outside the retryable set are returned without retrying."""
        client = self._make_client(tmp_path)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            return b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nbad request"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 400
        assert call_count == 1


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def _make_client(self, tmp_path, max_retries=3, retry_backoff=0.5):
        return GatewayHttpClient(
            socket_path=tmp_path / "gw.sock",
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def test_429_raises_rate_limit_error_after_retries(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 12\r\n\r\nrate limited"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        with pytest.raises(RateLimitError, match="Rate limited.*429.*retries"):
            client.request("GET", "/test")

    def test_429_with_max_retries_zero_raises_immediately(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path, max_retries=0)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            return b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 12\r\n\r\nrate limited"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        with pytest.raises(RateLimitError):
            client.request("GET", "/test")

        assert call_count == 1

    def test_retry_after_header_respected(self, tmp_path, monkeypatch):
        """When 429 includes Retry-After header, the delay uses that value."""
        client = self._make_client(tmp_path, retry_backoff=0.5)

        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda d: sleep_calls.append(d))

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Retry-After: 5\r\n"
                    b"Content-Length: 12\r\n"
                    b"\r\n"
                    b"rate limited"
                )
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 5.0

    def test_429_then_success(self, tmp_path, monkeypatch):
        """A transient 429 followed by a 200 succeeds."""
        client = self._make_client(tmp_path)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        call_count = 0

        def mock_send_raw(data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\n\r\n"
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        resp = client.request("GET", "/test")

        assert resp.status == 200
        assert call_count == 2


# ---------------------------------------------------------------------------
# Timeout handling in _send_raw
# ---------------------------------------------------------------------------


class TestSendRawTimeout:
    def test_timeout_raises_gateway_error(self, tmp_path, monkeypatch):
        """TimeoutError during recv raises GatewayError, not silently returning partial data."""
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")

        recv_count = 0

        class FakeSocket:
            def __init__(self):
                pass

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                nonlocal recv_count
                recv_count += 1
                if recv_count == 1:
                    return b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial"
                raise TimeoutError("timed out")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(GatewayError, match="timed out"):
            client._send_raw(b"GET / HTTP/1.1\r\n\r\n")

    def test_timeout_on_first_recv_raises_gateway_error(self, tmp_path, monkeypatch):
        """TimeoutError on the very first recv raises GatewayError."""
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")

        class FakeSocket:
            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                raise TimeoutError("timed out")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(GatewayError, match="timed out"):
            client._send_raw(b"GET / HTTP/1.1\r\n\r\n")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def _make_client(self, tmp_path):
        return GatewayHttpClient(
            socket_path=tmp_path / "gw.sock", max_retries=0,
        )

    def test_returns_true_on_200(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        assert client.health_check() is True

    def test_returns_false_on_non_200(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            return b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        assert client.health_check() is False

    def test_returns_false_when_gateway_unreachable(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            raise GatewayUnavailableError("Cannot connect")

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        assert client.health_check() is False

    def test_returns_false_on_gateway_error(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        def mock_send_raw(data: bytes) -> bytes:
            raise GatewayError("something broke")

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        assert client.health_check() is False

    def test_sends_get_health(self, tmp_path, monkeypatch):
        """Verify health_check sends GET /health."""
        client = self._make_client(tmp_path)

        captured_data = {}

        def mock_send_raw(data: bytes) -> bytes:
            captured_data["raw"] = data
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        monkeypatch.setattr(client, "_send_raw", mock_send_raw)
        client.health_check()

        raw = captured_data["raw"].decode()
        assert raw.startswith("GET /health HTTP/1.1\r\n")


# ---------------------------------------------------------------------------
# Streaming (request_stream)
# ---------------------------------------------------------------------------


class TestRequestStream:
    def _make_client(self, tmp_path):
        return GatewayHttpClient(socket_path=tmp_path / "gw.sock")

    def test_yields_lines_from_sse_response(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        sse_body = b"data: hello\ndata: world\n\n"
        response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" + sse_body

        class FakeSocket:
            def __init__(self):
                self._sent = False

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                if not self._sent:
                    self._sent = True
                    return response
                return b""

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        lines = list(client.request_stream("GET", "/stream"))

        assert b"data: hello\n" in lines
        assert b"data: world\n" in lines

    def test_yields_lines_arriving_in_multiple_chunks(self, tmp_path, monkeypatch):
        """SSE data arriving across multiple recv calls is yielded line-by-line."""
        client = self._make_client(tmp_path)

        header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        chunks = [
            header + b"data: first",  # incomplete line
            b"\ndata: second\n",       # completes first, includes second
            b"data: third\n",          # third line
            b"",                        # EOF
        ]

        class FakeSocket:
            def __init__(self):
                self._iter = iter(chunks)

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return next(self._iter, b"")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        lines = list(client.request_stream("GET", "/stream"))

        assert lines == [b"data: first\n", b"data: second\n", b"data: third\n"]

    def test_stream_non_200_raises_gateway_error(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        chunks = [b"HTTP/1.1 500 Internal Server Error\r\n\r\nfailed", b""]

        class FakeSocket:
            def __init__(self):
                self._iter = iter(chunks)

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return next(self._iter, b"")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(GatewayError, match="status 500"):
            list(client.request_stream("GET", "/stream"))

    def test_stream_429_raises_rate_limit_error(self, tmp_path, monkeypatch):
        client = self._make_client(tmp_path)

        response = b"HTTP/1.1 429 Too Many Requests\r\n\r\nrate limited"
        sent = False

        class FakeSocket:
            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                nonlocal sent
                if not sent:
                    sent = True
                    return response
                return b""

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(RateLimitError, match="429"):
            list(client.request_stream("GET", "/stream"))

    def test_stream_connection_refused_raises_gateway_unavailable(self, tmp_path):
        client = GatewayHttpClient(socket_path=tmp_path / "nonexistent.sock")

        with pytest.raises(GatewayUnavailableError, match="Cannot connect"):
            list(client.request_stream("GET", "/stream"))

    def test_stream_timeout_raises_gateway_error(self, tmp_path, monkeypatch):
        """Timeout during streaming body raises GatewayError."""
        client = self._make_client(tmp_path)

        header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        recv_count = 0

        class FakeSocket:
            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                nonlocal recv_count
                recv_count += 1
                if recv_count == 1:
                    return header + b"data: first\n"
                raise TimeoutError("timed out")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(GatewayError, match="timed out"):
            list(client.request_stream("GET", "/stream"))

    def test_stream_unexpected_error_wrapped_in_gateway_error(self, tmp_path, monkeypatch):
        """Non-gateway exceptions during streaming are wrapped in GatewayError."""
        client = self._make_client(tmp_path)

        class FakeSocket:
            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                raise RuntimeError("unexpected failure")

            def recv(self, bufsize):
                return b""

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        with pytest.raises(GatewayError, match="Gateway stream failed"):
            list(client.request_stream("GET", "/stream"))

    def test_stream_yields_remaining_data_without_trailing_newline(self, tmp_path, monkeypatch):
        """Data remaining in the buffer after EOF (no trailing newline) is yielded."""
        client = self._make_client(tmp_path)

        header = b"HTTP/1.1 200 OK\r\n\r\n"

        class FakeSocket:
            def __init__(self):
                self._sent = False

            def settimeout(self, t):
                pass

            def connect(self, addr):
                pass

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                if not self._sent:
                    self._sent = True
                    return header + b"data: partial"
                return b""

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSocket())

        lines = list(client.request_stream("GET", "/stream"))

        assert lines == [b"data: partial"]


# ---------------------------------------------------------------------------
# __init__ with retry parameters
# ---------------------------------------------------------------------------


class TestInitRetryParams:
    def test_default_retry_params(self, tmp_path):
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock")
        assert client._max_retries == 3
        assert client._retry_backoff == 0.5

    def test_custom_retry_params(self, tmp_path):
        client = GatewayHttpClient(
            socket_path=tmp_path / "gw.sock",
            max_retries=5,
            retry_backoff=1.0,
        )
        assert client._max_retries == 5
        assert client._retry_backoff == 1.0

    def test_max_retries_zero(self, tmp_path):
        client = GatewayHttpClient(socket_path=tmp_path / "gw.sock", max_retries=0)
        assert client._max_retries == 0
