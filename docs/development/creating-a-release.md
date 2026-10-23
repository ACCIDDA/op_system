# Creating a core release

This workflow releases only the independent MIT-licensed `op_system` core.
It cannot build or publish `flepimop2-op-system`. The provider has a separate
version, changelog and release lifecycle; a future provider release requires a
separately reviewed workflow and namespaced tag.

## Prerequisites

- You have write access to
  [`ACCIDDA/op_system`](https://github.com/ACCIDDA/op_system).
- The core version in `pyproject.toml` is a canonical PEP 440 release version.
- `CITATION.cff` has the matching core version, release date, MIT license and
  `v<version>` artifact URL.
- `CHANGELOG.md` has a dated section for the core version.
- The release commit is the current default-branch head.
- Independent review approved that exact commit and required CI is green.

The provider version does not need to match the core. Core release validation
deliberately ignores it, while provider validation requires a matching version
and exact core dependency.

## 1. Run the local preflight

```shell
just release-validate-core
```

This validates core release and citation metadata, builds both the wheel and
source archive, rejects provider files in either artifact, applies strict
metadata checks and runs the installed-core test suite. Provider compatibility
checks remain part of ordinary repository CI, but provider artifacts are not
release artifacts.

## 2. Run a non-publishing workflow check

Before merging a release PR, dispatch the workflow on its branch:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_system \
  --ref <branch-name> \
  --field publish-target=none \
  --field create-github-release=false \
  --field deploy-docs=false
```

The run validates and uploads only `op-system` artifacts. It performs no
package publication, GitHub release creation or documentation deployment.

## 3. Publish from authoritative main

After exact-commit review, green CI, successful dry run and merge, confirm that
`main` points to the reviewed merge commit. Then dispatch:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_system \
  --field publish-target=pypi \
  --field create-github-release=true \
  --field deploy-docs=true
```

Production actions refuse a branch or stale default-branch checkout. The
workflow publishes `op-system`, creates the exact `v<version>` GitHub release
and deploys versioned core documentation. A zero major version is not itself a
prerelease; only explicit alpha, beta and release-candidate versions receive
the prerelease flag.

To test only package-index acceptance, use `publish-target=testpypi` with
release creation and documentation deployment disabled. Branch TestPyPI runs
remain non-authoritative and must not create releases.

## 4. Documentation behavior

Push deployments require the current `main` head and update only mutable
`dev` documentation. Release deployments require the exact tag, refuse to
overwrite an existing exact documentation version, and move `latest` for
final releases or `pre` for explicit prereleases. The deployment helper
prevents Mike from receiving the duplicated `dev dev` arguments.

## 5. Verify the result

Confirm all of the following:

- PyPI serves the intended `op-system` version and its wheel and source archive.
- No provider version was published or changed.
- The GitHub release tag resolves to the reviewed main commit.
- Versioned documentation resolves and the appropriate alias points to it.
- The published source archive contains no `flepimop2-op_system` tree or GPL
  provider license.

GitHub release and package publishing do not themselves prove archival
ingestion. Follow the [citation and archiving plan](citation-and-archiving.md),
verify any existing archive integration without exposing credentials, and
record only identifiers that resolve to the exact release.
