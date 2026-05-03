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
from pathlib import Path
from socketserver import ThreadingMixIn


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
