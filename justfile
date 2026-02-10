# Run all default tasks for local development
default: format check pytest mypy

# -------------------------------------------------
# Local venv + deps (explicit step)
# -------------------------------------------------

# Create/refresh the main venv and install dev deps.
venv:
	uv venv --clear
	uv sync --dev

# -------------------------------------------------
# Formatting / linting
# -------------------------------------------------

# Tool-only execution (does not require resolving project deps).
format:
	uvx ruff format --preview

check:
	uvx ruff check --preview --fix

# -------------------------------------------------
# Provider venv + deps (explicit step)
# -------------------------------------------------

# Create/refresh the provider venv and install all deps (including dev group).
provider-sync:
	cd flepimop2-op_system && uv venv --clear
	cd flepimop2-op_system && uv sync --dev

# -------------------------------------------------
# Tests
# -------------------------------------------------

pytest-core:
	uv run pytest --doctest-modules

# Ensure provider venv exists before running provider tests.
pytest-provider: provider-sync
	cd flepimop2-op_system && .venv/bin/python -m pytest --doctest-modules

pytest:
	just pytest-core
	just pytest-provider

# -------------------------------------------------
# Type checking
# -------------------------------------------------

mypy-core:
	uv run mypy --strict src/op_system

# Ensure provider venv exists before running provider mypy.
mypy-provider: provider-sync
	cd flepimop2-op_system && .venv/bin/python -m mypy --strict src/flepimop2

mypy:
	just mypy-core
	just mypy-provider

# -------------------------------------------------
# CI aggregate
# -------------------------------------------------

ci:
	uvx ruff format --preview --check
	uvx ruff check --preview --no-fix
	just provider-sync
	just pytest
	just mypy

# -------------------------------------------------
# Utilities
# -------------------------------------------------

clean:
	rm -f uv.lock
	rm -rf .*_cache
	rm -rf .venv
	rm -rf flepimop2-op_system/.venv
	rm -f flepimop2-op_system/uv.lock

docs:
	uv run mkdocs build --verbose --strict

serve:
	uv run mkdocs serve
