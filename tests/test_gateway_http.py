"""Tests for GatewayHttpClient HTTP parsing and socket communication."""

import json
import socket
import threading
from pathlib import Path

import pytest

from sage_sanctum.errors import GatewayError, GatewayUnavailableError
from sage_sanctum.gateway.http import GatewayHttpClient, HttpResponse


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
