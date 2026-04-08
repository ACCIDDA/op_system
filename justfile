run := "uv run"
provider_dir := "flepimop2-op_system"
provider_run := run + " --project " + provider_dir + " --directory " + provider_dir

# Run all default tasks for local development
default: dev docs

# Run all default dev tasks
dev: ruff mypy test

# Run all default CI tasks
ci: quality test docs

# Format code in both packages
[group('dev')]
fmt: fmt-root fmt-provider

# Run local Ruff formatting and lint fixes in both packages
[group('dev')]
ruff: fmt lint

# Run local lint fixes in both packages
[group('dev')]
lint: lint-root lint-provider

# Run CI Ruff checks in both packages
[group('ci')]
ci-ruff: ci-ruff-root ci-ruff-provider

# Format the root package
[group('dev')]
fmt-root:
	{{run}} ruff format

# Run Ruff lint fixes for the root package
[group('dev')]
lint-root:
	{{run}} ruff check --fix

# Run CI Ruff checks for the root package
[group('ci')]
ci-ruff-root:
	{{run}} ruff format --check
	{{run}} ruff check --no-fix

# Format the provider package
[group('dev')]
fmt-provider:
	{{provider_run}} ruff format

# Run Ruff lint fixes for the provider package
[group('dev')]
lint-provider:
	{{provider_run}} ruff check --fix

# Run CI Ruff checks for the provider package
[group('ci')]
ci-ruff-provider:
	{{provider_run}} ruff format --check
	{{provider_run}} ruff check --no-fix

# Run all tests in both packages
[group('dev')]
[group('ci')]
test: test-root test-provider

# Run the root package test suite
[group('dev')]
[group('ci')]
test-root:
	{{run}} pytest

# Run the provider package test suite
[group('dev')]
[group('ci')]
test-provider: provider-sync
	cd {{provider_dir}} && .venv/bin/python -m pytest

# Create/refresh the provider venv and install all deps (including dev group).
# Run this once before running provider pytest/mypy if you aren't using `ci`.
[group('dev')]
provider-sync:
	cd {{provider_dir}} && uv venv --clear
	cd {{provider_dir}} && uv sync --dev

# Run type checks in both packages
[group('dev')]
[group('ci')]
mypy: mypy-root mypy-provider

# Run mypy for the root package
[group('dev')]
[group('ci')]
mypy-root:
	{{run}} mypy

# Run mypy for the provider package
[group('dev')]
[group('ci')]
mypy-provider: provider-sync
	cd {{provider_dir}} && .venv/bin/python -m mypy --strict --namespace-packages --explicit-package-bases src/flepimop2

# Run all CI quality checks
[group('ci')]
quality: ci-ruff mypy

# Build API reference for the documentation using `mkdocstrings`
[group('docs')]
api-reference:
	{{run}} scripts/api-reference.py

# Build the documentation using `mkdocs`
[group('docs')]
docs: api-reference
	{{run}} mkdocs build --verbose --strict

# Serve the documentation locally using `mkdocs`
[group('docs')]
serve: api-reference
	{{run}} mkdocs serve

# Clean generated venvs, lockfiles, caches, and built docs
[group('dev')]
[unix]
clean:
	rm -rf site
	rm -f uv.lock
	rm -rf .venv
	rm -rf .*_cache
	rm -f flepimop2-op_system/uv.lock
	rm -rf flepimop2-op_system/.venv
	rm -rf flepimop2-op_system/.*_cache
