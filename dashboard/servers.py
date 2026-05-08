"""WSGI server bootstrap used to run the dashboard locally.

Extracted from ``workflow_dashboard.py`` without behavior changes. Provides the
threaded WSGI server class plus the helpers that build a server and bind to a
free port.

``waitress_available`` and ``resolve_server_mode`` deliberately remain in
``workflow_dashboard`` so the existing tests can monkey-patch
``workflow_web.waitress_available`` and have ``resolve_server_mode`` honor the
patched binding. Moving them here would require the tests to also patch
``dashboard.servers.waitress_available``.
"""
from __future__ import annotations

import errno
import gzip
import io
from pathlib import Path
from socketserver import ThreadingMixIn


# Content types that compress well. Audio/video/images are already compressed
# so we skip them — gzip would just burn CPU for ~0% savings.
_GZIP_MIN_BYTES = 1024
_GZIP_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-yaml",
    "image/svg+xml",
)


def _accepts_gzip(environ) -> bool:
    accept = environ.get("HTTP_ACCEPT_ENCODING", "")
    return "gzip" in accept.lower()


def _is_gzippable(content_type: str) -> bool:
    if not content_type:
        return False
    lowered = content_type.lower()
    return any(lowered.startswith(prefix) for prefix in _GZIP_CONTENT_TYPES)


class GzipMiddleware:
    """WSGI middleware that compresses text/JSON responses with gzip.

    The dashboard ships a multi-megabyte JSON state blob inside the page HTML —
    on a slow link the browser stalls waiting for it before LCP can fire.
    Most of that blob is repetitive text, so gzip cuts it ~10x. We skip
    binary/already-compressed content types and very small responses to avoid
    wasting CPU on cases where compression doesn't pay off.
    """

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        if not _accepts_gzip(environ):
            return self.application(environ, start_response)

        # We need to inspect the inner app's Content-Type before deciding what
        # to do, but we must NOT buffer the body for non-gzippable responses —
        # otherwise large media downloads (audio_in/*.wav, 70+ MB) get slurped
        # into memory and break range/streaming behavior. Capture status +
        # headers via a wrapped start_response, then branch.
        captured: dict[str, object] = {}

        def capture_start(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = list(headers)
            captured["exc_info"] = exc_info
            return lambda chunk: None  # discard write() escape hatch

        body_iter = self.application(environ, capture_start)

        headers = captured.get("headers") or []
        header_map = {name.lower(): value for name, value in headers}
        content_type = header_map.get("content-type", "")
        already_encoded = header_map.get("content-encoding", "")
        status = captured.get("status", "200 OK")
        exc_info = captured.get("exc_info")

        if already_encoded or not _is_gzippable(content_type):
            # Pass through untouched — preserves chunked streaming and range
            # responses (Content-Range / 206) for audio + binary assets.
            start_response(status, headers, exc_info)
            return body_iter

        # Gzippable: buffer the (small) text/JSON body so we can compress.
        try:
            chunks = list(body_iter)
        finally:
            close = getattr(body_iter, "close", None)
            if callable(close):
                close()
        body = b"".join(chunks)

        if len(body) < _GZIP_MIN_BYTES:
            new_headers = [(name, value) for name, value in headers if name.lower() != "content-length"]
            new_headers.append(("Content-Length", str(len(body))))
            start_response(status, new_headers, exc_info)
            return [body]

        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as fh:
            fh.write(body)
        compressed = buffer.getvalue()

        new_headers = [
            (name, value)
            for name, value in headers
            if name.lower() not in {"content-length", "content-encoding"}
        ]
        new_headers.append(("Content-Encoding", "gzip"))
        new_headers.append(("Content-Length", str(len(compressed))))
        # Vary: Accept-Encoding lets caches store both the gzip and identity
        # variants without serving the wrong one to a non-gzip client.
        if not any(name.lower() == "vary" for name, _ in new_headers):
            new_headers.append(("Vary", "Accept-Encoding"))
        start_response(status, new_headers, exc_info)
        return [compressed]


class ThreadedWSGIServer(ThreadingMixIn):
    """Allow the built-in WSGI server to handle concurrent browser requests."""

    daemon_threads = True


def built_in_make_server(host: str, port: int, application, *, threaded: bool):
    """Create one of the built-in WSGI server variants."""

    from wsgiref.simple_server import WSGIServer, make_server

    server_class = type(
        "ThreadedBuiltinWSGIServer" if threaded else "BuiltinWSGIServer",
        (ThreadedWSGIServer, WSGIServer) if threaded else (WSGIServer,),
        {},
    )
    return make_server(host, port, application, server_class=server_class)


def waitress_make_server(host: str, port: int, application, *, threads: int):
    """Create a Waitress server when the optional dependency is available."""

    from waitress.server import create_server

    return create_server(application, host=host, port=port, threads=threads)


def server_port(server, requested_port: int) -> int:
    """Recover the bound port from the server object when possible."""

    for attribute in ("server_port", "effective_port", "port"):
        value = getattr(server, attribute, None)
        if isinstance(value, int) and value > 0:
            return value

    socket_obj = getattr(server, "socket", None)
    getsockname = getattr(socket_obj, "getsockname", None)
    if callable(getsockname):
        try:
            return int(getsockname()[1])
        except (OSError, TypeError, ValueError):
            pass

    return requested_port


def write_url_file(destination: str | None, url: str) -> None:
    """Persist the resolved local dashboard URL when requested."""

    if not destination:
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{url}\n", encoding="utf-8")


def bind_server(host: str, port: int, application, *, max_port_tries: int = 25, make_server_fn=None):
    """Bind the workflow web server, retrying nearby ports when one is already busy."""

    make_server_fn = make_server_fn or (
        lambda bind_host, bind_port, bind_app: built_in_make_server(
            bind_host,
            bind_port,
            bind_app,
            threaded=False,
        )
    )
    attempts = 1 if port == 0 else max(max_port_tries, 1)
    last_exc: OSError | None = None
    for offset in range(attempts):
        candidate_port = port if port == 0 else port + offset
        try:
            server = make_server_fn(host, candidate_port, application)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or port == 0:
                raise
            last_exc = exc
            continue
        return server, server_port(server, candidate_port)

    upper_bound = port + attempts - 1
    raise OSError(
        errno.EADDRINUSE,
        f"No free port found between {port} and {upper_bound}.",
    ) from last_exc
