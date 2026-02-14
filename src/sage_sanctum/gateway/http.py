"""HTTP client with Unix socket and TCP support for gateway communication."""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

from ..errors import GatewayError, GatewayUnavailableError

logger = logging.getLogger(__name__)

_BUFFER_SIZE = 65536
_DEFAULT_TIMEOUT = 120  # seconds


@dataclass
class HttpResponse:
    """Simple HTTP response container.

    Attributes:
        status: HTTP status code (e.g. ``200``, ``429``, ``502``).
        headers: Response headers (keys are lower-cased).
        data: Response body as a decoded string.
    """

    status: int
    headers: dict[str, str]
    data: str


class GatewayHttpClient:
    """HTTP client for gateway communication over Unix socket or TCP.

    In production, agents communicate with gateways via Unix sockets
    (AF_UNIX) because AF_INET is blocked by seccomp. For local development,
    TCP connections are also supported.
    """

    def __init__(
        self,
        socket_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the HTTP client.

        Exactly one of ``socket_path`` or ``host`` must be provided.

        Args:
            socket_path: Path to a Unix domain socket (``AF_UNIX``).
            host: TCP hostname for ``AF_INET`` connections.
            port: TCP port number (required when ``host`` is set).
            timeout: Socket timeout in seconds. Defaults to 120.

        Raises:
            GatewayError: If neither ``socket_path`` nor ``host`` is provided.
        """
        self._socket_path = Path(socket_path) if socket_path else None
        self._host = host
        self._port = port
        self._timeout = timeout

        if not self._socket_path and not self._host:
            raise GatewayError("Either socket_path or host must be provided")

    @property
    def is_unix_socket(self) -> bool:
        """Whether this client connects via Unix domain socket."""
        return self._socket_path is not None

    def request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: str | dict | None = None,
    ) -> HttpResponse:
        """Send an HTTP request to the gateway.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g., '/v1/chat/completions')
            headers: Additional HTTP headers
            body: Request body (str or dict to be JSON-encoded)

        Returns:
            HttpResponse with status, headers, and body data.

        Raises:
            GatewayUnavailableError: If the gateway is unreachable.
            GatewayError: On other communication errors.
        """
        all_headers = {"Host": "gateway.local"}
        if headers:
            all_headers.update(headers)

        body_bytes = b""
        if body is not None:
            if isinstance(body, dict):
                body_bytes = json.dumps(body).encode("utf-8")
                all_headers["Content-Type"] = "application/json"
            else:
                body_bytes = body.encode("utf-8")
            all_headers["Content-Length"] = str(len(body_bytes))

        # Build raw HTTP request
        request_line = f"{method} {path} HTTP/1.1\r\n"
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in all_headers.items())
        raw_request = (request_line + header_lines + "\r\n").encode("utf-8") + body_bytes

        try:
            raw_response = self._send_raw(raw_request)
            return self._parse_response(raw_response)
        except (GatewayError, GatewayUnavailableError):
            raise
        except Exception as e:
            raise GatewayError(f"Gateway request failed: {e}") from e

    def _send_raw(self, data: bytes) -> bytes:
        """Send raw bytes over socket and read response."""
        try:
            if self._socket_path:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.connect(str(self._socket_path))
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.connect((self._host, self._port))
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            target = str(self._socket_path) if self._socket_path else f"{self._host}:{self._port}"
            raise GatewayUnavailableError(
                f"Cannot connect to gateway at {target}: {e}"
            ) from e

        try:
            sock.sendall(data)

            # Read response
            chunks = []
            while True:
                try:
                    chunk = sock.recv(_BUFFER_SIZE)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    # Check if we've received the full response
                    response = b"".join(chunks)
                    if self._is_response_complete(response):
                        break
                except TimeoutError:
                    break
            return b"".join(chunks)
        finally:
            sock.close()

    def _is_response_complete(self, response: bytes) -> bool:
        """Check if HTTP response is complete."""
        # Look for headers/body separator
        separator = b"\r\n\r\n"
        sep_idx = response.find(separator)
        if sep_idx == -1:
            return False

        headers_part = response[:sep_idx].decode("utf-8", errors="replace")
        body_start = sep_idx + len(separator)

        # Check Content-Length
        for line in headers_part.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                return len(response) >= body_start + content_length

        # Check for chunked transfer encoding end
        if b"transfer-encoding: chunked" in response.lower():
            return response.endswith(b"0\r\n\r\n")

        # For responses without Content-Length (e.g., connection: close), we rely on EOF
        return True

    def _parse_response(self, raw: bytes) -> HttpResponse:
        """Parse raw HTTP response bytes."""
        if not raw:
            raise GatewayError("Empty response from gateway")

        # Split headers and body
        separator = b"\r\n\r\n"
        sep_idx = raw.find(separator)
        if sep_idx == -1:
            raise GatewayError("Malformed HTTP response: no header/body separator")

        header_bytes = raw[:sep_idx]
        body_bytes = raw[sep_idx + len(separator):]

        # Parse status line
        header_text = header_bytes.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        if not lines:
            raise GatewayError("Malformed HTTP response: no status line")

        status_parts = lines[0].split(" ", 2)
        if len(status_parts) < 2:
            raise GatewayError(f"Malformed status line: {lines[0]!r}")
        status = int(status_parts[1])

        # Parse headers
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        # Decode chunked transfer encoding if present
        if headers.get("transfer-encoding", "").lower() == "chunked":
            body_bytes = self._decode_chunked(body_bytes)

        body = body_bytes.decode("utf-8", errors="replace")

        return HttpResponse(status=status, headers=headers, data=body)

    @staticmethod
    def _decode_chunked(data: bytes) -> bytes:
        """Decode HTTP chunked transfer encoding."""
        decoded = bytearray()
        pos = 0
        while pos < len(data):
            # Find end of chunk size line
            crlf = data.find(b"\r\n", pos)
            if crlf == -1:
                break
            # Parse hex chunk size (ignore extensions after semicolon)
            size_str = data[pos:crlf].split(b";")[0].strip()
            if not size_str:
                break
            try:
                chunk_size = int(size_str, 16)
            except ValueError:
                break
            if chunk_size == 0:
                break  # Terminal chunk
            # Extract chunk data
            chunk_start = crlf + 2
            chunk_end = chunk_start + chunk_size
            if chunk_end > len(data):
                # Incomplete chunk — take what we have
                decoded.extend(data[chunk_start:])
                break
            decoded.extend(data[chunk_start:chunk_end])
            # Skip trailing CRLF after chunk data
            pos = chunk_end + 2
        return bytes(decoded)
