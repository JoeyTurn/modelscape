#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v pytest >/dev/null 2>&1; then
  python -m pytest -q "$TEST_DIR"
else
  python -m unittest discover -s "$TEST_DIR" -p "test_*.py"
fi
