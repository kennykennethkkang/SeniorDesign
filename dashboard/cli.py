"""Command-line entry point for the dashboard server.

Extracted from ``workflow_dashboard.py``. ``main`` lazily imports the
WorkflowWebApp instance to avoid a circular import; it can also receive a
custom ``app`` if a caller wants to inject one for testing.
"""
from __future__ import annotations

import argparse
import errno
import socket
import sys
import threading

from dashboard.constants import DEFAULT_SERVER_MODE, DEFAULT_SERVER_THREADS
from dashboard.servers import (
    GzipMiddleware,
    bind_server,
    built_in_make_server,
    waitress_make_server,
    write_url_file,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose a minimal CLI for serving the dashboard locally."""

    parser = argparse.ArgumentParser(description="Serve the ML Speech Diarization dashboard.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Address to bind. Defaults to 127.0.0.1 (local only), which is what "
            "an SSH tunnel connects to. Pass --host 0.0.0.0 to expose the "
            "dashboard on the network, but only on a trusted network: the "
            "dashboard has no authentication."
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--server",
        choices=["auto", "threaded", "waitress", "wsgiref"],
        default=DEFAULT_SERVER_MODE,
        help="Runtime server to use. 'auto' prefers Waitress when installed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_SERVER_THREADS,
        help="Worker thread count for Waitress. Retained for auto mode selection.",
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help="Optional path that receives the resolved local URL after startup.",
    )
    return parser


def main(argv: list[str] | None = None, app=None) -> int:
    """Start the dashboard with the selected WSGI runtime."""

    # Lazy imports to avoid circular import (workflow_dashboard imports from dashboard.cli)
    # and to honor monkey-patched waitress_available (kept in workflow_dashboard).
    if app is None:
        from workflow_dashboard import app as _default_app
        app = _default_app
    from workflow_dashboard import resolve_server_mode

    args = build_parser().parse_args(argv)
    host = args.host
    requested_port = args.port
    if args.threads < 1:
        print("Error: --threads must be at least 1.", file=sys.stderr)
        return 1
    try:
        server_mode = resolve_server_mode(args.server)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    display_host = host
    if host in {"0.0.0.0", "::"}:
        display_host = "127.0.0.1"
    hostname = socket.gethostname()
    make_server_fn = None
    runtime_label = "built-in WSGI"
    if server_mode == "threaded":
        make_server_fn = lambda bind_host, bind_port, bind_app: built_in_make_server(
            bind_host,
            bind_port,
            bind_app,
            threaded=True,
        )
        runtime_label = "threaded built-in WSGI"
    elif server_mode == "waitress":
        make_server_fn = lambda bind_host, bind_port, bind_app: waitress_make_server(
            bind_host,
            bind_port,
            bind_app,
            threads=args.threads,
        )
        runtime_label = f"Waitress ({args.threads} worker threads)"
    elif server_mode == "wsgiref":
        runtime_label = "standard-library WSGI"

    # Wrap with gzip compression at the server boundary only; keep tests
    # talking to the bare WSGI app so they see uncompressed responses.
    served_app = GzipMiddleware(app)

    try:
        server, actual_port = bind_server(host, requested_port, served_app, make_server_fn=make_server_fn)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        raise
    local_url = f"http://{display_host}:{actual_port}"
    try:
        write_url_file(args.url_file, local_url)
        if actual_port != requested_port:
            print(
                f"Port {requested_port} is already in use. "
                f"Serving workflow hub on port {actual_port} instead."
            )
        print(f"Server runtime: {runtime_label}")
        if host in {"0.0.0.0", "::"}:
            print(f"Serving workflow hub on all interfaces, port {actual_port}")
            print(f"Open locally: {local_url}")
            print(
                "Open remotely if your HPC/network setup allows it: "
                f"http://{hostname}:{actual_port}"
            )
        else:
            print(f"Serving workflow hub on {local_url}")

        # Warm the dashboard caches in the background so the first real
        # page render isn't a cold rglob/list_projects/750 KB-JSON parse
        # all at once. Daemon thread so it never blocks shutdown.
        warmup = getattr(app, "warmup_caches", None)
        if callable(warmup):
            threading.Thread(target=warmup, name="dashboard-warmup", daemon=True).start()

        if server_mode == "waitress":
            server.run()
        else:
            with server:
                server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping workflow hub.")
    finally:
        if server_mode == "waitress":
            close = getattr(server, "close", None)
            if callable(close):
                close()
    return 0
