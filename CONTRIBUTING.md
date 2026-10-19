# Contributing

`op_system` is the standalone specification/normalization/RHS compiler in
`src/op_system/`. The `flepimop2-op_system/` distribution adapts compiled models
to flepimop2. Put compiler behavior in the core and provider integration in the
adapter; a core-only change still needs provider compatibility checks. Runtime
JAX/Torch extras remain optional for users.

## Clean development setup

Install Git, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/man/en/). Use Python 3.12 for the development/docs
commands below. Core CI also tests Python 3.11 and 3.13; Python 3.14 is outside
the core package's declared range.

```shell
git clone https://github.com/ACCIDDA/op_system.git
cd op_system
uv python install 3.12
export UV_MANAGED_PYTHON=true
export UV_PYTHON=3.12
export UV_PYTHON_VERSION="$UV_PYTHON"
uv sync --dev --python "$UV_PYTHON"
just provider-sync
```

The setup uses a complete uv-managed Python, including the `ensurepip` support
needed by isolated wheel builds. The root and provider have separate virtual
environments. `UV_PYTHON` selects uv interpreters; `UV_PYTHON_VERSION` selects the
clean-wheel test interpreter. `provider-sync`
refreshes the provider environment and installs the local core plus its declared
flepimop2 dependency. It clears the provider's `.venv`; keep personal packages
in a separate environment. `uv.lock` files are generated locally and ignored.
The docs API generator runs as a uv script with its own declared dependencies.

## Prepare a focused change

Start a branch from current `main` and keep each PR focused on one issue or
behavior. Read [the code conventions](docs/development/code-style.md): default
to `NamedTuple` for small immutable records, use private helper names, and raise
specific errors where failure occurs. Add behavior tests that would fail before
the fix; assert independently expected equations or invariants rather than only
copying generated expressions. Explain changed public behavior in the docs and
`CHANGELOG.md` under `Unreleased`. Include migration guidance for a compatibility
change. Do not bump the shared release version for an ordinary fix.

Run these checks locally **before pushing or opening a PR**:

```shell
just fmt
just ci
just qualify
just build-all
```

`just fmt` formats both packages; review its changes. `just ci` checks Ruff and
strict typing for both distributions, runs core/provider tests, and builds docs
with strict link/warning checks. `just build-all` builds both wheels, validates
metadata, and tests installed wheels in clean environments. When iterating, use
focused tests, then run the full checks above once the change is ready.

`just qualify` installs and tests NumPy, JAX, and CPU Torch explicitly. It rejects
missing backends and skipped selected tests; do not count baseline
`importorskip` results as backend qualification. These commands refresh the
root environment between backends, so use the target appropriate to a rerun or
`uv run --no-sync` if retaining the final environment is intentional. The CI
jobs `compiler (numpy, 3.12)`, `compiler (jax, 3.12)`, and
`compiler (torch, 3.12)` are required alongside baseline docs/quality/build and
Python 3.11–3.13 test jobs. Provider tests remain a separate compatibility layer.

The `provider-typing` job verifies the provider's installed wheel, including its
`py.typed` marker and strict downstream typing of public state/stepper interfaces.
It also requires an invalid state argument to be rejected by the type checker.
The root mypy exclusion for `flepimop2-op_system/` is intentional: the adapter
runs strict checks in its own dependency environment through `mypy-provider`.
Run `just build-all` to include the installed consumer check locally.

Optional GPU checks require suitable hardware and are outside the required CPU
matrix. Large timing runs are opt-in; benchmark correctness is checked in CI,
but there are no timing thresholds. See
[backend qualification](docs/guides/backend-qualification.md) and
[numerical qualification](docs/guides/numerical-qualification.md) for exact
capability boundaries and fixture/reference checks. A backend change must
preserve dtype, shape, and namespace and exercise its claimed tracing/AD path.

The PR description should state the trigger/problem, resulting behavior, issue
link, tests/checks run, and material limitations. Use focused commits that a
reviewer can inspect. Follow repository review and branch protection rules;
merge only after every current-head CI job is green. A new commit invalidates
the previous CI result for that head.

## Support and bug reports

Use [GitHub issues](https://github.com/ACCIDDA/op_system/issues) for support,
bugs, or proposed behavior, searching existing issues first. Include a minimal
specification, state/parameter shapes and dtypes, Python/package/backend
versions, evaluator layout, expected equations/result, and traceback. Use
synthetic fixtures when real data cannot be shared. For provider problems,
include both distribution versions and the flepimop2 version and distinguish
compiler evaluation from the integration failure. Numerical disagreement should
include an independent reference and tolerances where possible.

## Releases, documentation, and citation

The two distributions share one version and one `vX.Y.Z` GitHub release. The
release workflow publishes core before provider. See
[creating a release](docs/development/creating-a-release.md) for validation and
publishing; publishing is a separate maintainer action from merging a PR.
Core uses MIT; the provider uses GPLv3 or later. Preserve the applicable license and
existing attribution when moving or reusing code.

Use the [documentation version selector](https://accidda.github.io/op_system/)
for released documentation. `main`/the development docs can describe unreleased
behavior. Find migration notes in the
[changelog](CHANGELOG.md) and relevant guide; use the exact tag's README and
source when reproducing a release. The documentation deployment groups releases
by major/minor, so use the exact source tag for patch-specific behavior.

[CITATION.cff](CITATION.cff) records the shared released software and the authors
already attributed in both package metadata files. Maintainers should review
author order/names and release details when these change. Validate it with
`just citation`; `just ci` and release validation include the same syntax check.
See [citation and archiving](docs/development/citation-and-archiving.md) for
version-specific citations, the current identifier audit, and the software-paper
archive plan. Add a preferred paper citation only once its bibliographic record
exists and has been verified by the maintainers.

## Executable documentation

Keep capability claims, API docstrings and complete examples in sync. Add a
small deterministic `Examples:` doctest for a changed public workflow where
useful; assert independently known equations, shapes or invariants. Larger and
optional workflows belong in dedicated tests that require their dependencies.
Run `just doctest` during iteration and the full checks above before a PR.
See [capabilities and example checks](docs/guides/capabilities.md).
