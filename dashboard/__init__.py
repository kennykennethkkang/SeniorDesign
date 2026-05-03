"""Dashboard package.

This package was extracted from the monolithic ``workflow_dashboard.py`` to
separate concerns (constants, slurm helpers, server bootstrap, CLI). The
single-class WSGI app currently still lives in ``workflow_dashboard``; later
phases of the refactor will split it into per-domain mixins under
``dashboard.app``.

Importing this package has no side effects on its own. Callers should import
``workflow_dashboard`` (or the specific submodule they need).
"""
from __future__ import annotations
