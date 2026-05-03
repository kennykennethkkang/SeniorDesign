#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash tests/run_all_tests.sh
