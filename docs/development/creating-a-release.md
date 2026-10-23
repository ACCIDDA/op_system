# Creating A Release

This guide covers the `op_system` release process for this repository's two Python distributions:

- `op_system`
- `flepimop2-op_system`

The release workflow treats them as a single release unit. Both package versions must match, and the workflow publishes them sequentially under one GitHub release tag.

## Prerequisites

- You have write access to the [`ACCIDDA/op_system`](https://github.com/ACCIDDA/op_system) repository.
- The release version has already been updated everywhere it is declared:
  - `pyproject.toml`
  - `flepimop2-op_system/pyproject.toml`
- `CITATION.cff` has matching version, release date, and artifact URL, with
  maintainer-reviewed attribution from the package metadata.
- The `CHANGELOG.md` has a section for the new version.
- Your local environment is synced with a supported Python version.
- You have [GitHub's `gh` CLI](https://cli.github.com/) installed and authenticated if you plan to dispatch workflows from the command line.

## 1. Confirm The Shared Version

The release workflow validates that all version declarations match before it builds or publishes anything.

Today that means these two files must contain the same canonical PEP 440
version. Production releases use `X.Y.Z` or an explicit `X.Y.ZaN`, `X.Y.ZbN`,
or `X.Y.ZrcN` version:

- `pyproject.toml`
- `flepimop2-op_system/pyproject.toml`

If they differ, the `validate` job fails immediately. Both public module
versions are read dynamically from installed package metadata, using the
distribution names `op_system` and `flepimop2-op-system`, so neither needs a
separate manual edit. The provider reports `0+unknown` only when imported
directly from an uninstalled source tree where distribution metadata is absent.

## 2. Run The Local Release Preflight

Use the local pre-release target before dispatching the release workflow:

```shell
just release-validate
```

That command performs these checks:

- Validates CFF syntax and checks citation attribution, shared version, date,
  and release URL against package metadata and the dated changelog.

- Validates that the release version matches in:
  - `pyproject.toml`
  - `flepimop2-op_system/pyproject.toml`
- Runs `just build-all` to execute the clean-room build and install tests for both packages.

The release workflow runs the same validation again on GitHub Actions before publishing.

## 3. Run The Release Workflow

Releases are created through the manual GitHub Actions workflow in `.github/workflows/release.yaml`.

If you are testing the workflow before merging, add `--ref <branch-name>` to the `gh workflow run` command. Without `--ref`, GitHub dispatches the workflow definition from the repository's default branch.

### Dry Run

Use this to validate shared versioning, build both packages, and upload release artifacts without publishing anything:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_system \
  --ref <branch-name> \
  --field publish-target=none \
  --field create-github-release=false \
  --field deploy-docs=false
```

This runs the `validate`, `publish-core`, and `publish-provider` jobs, but the publish jobs become no-ops when `publish-target=none`.

### TestPyPI

Use this to publish both packages to TestPyPI without creating the GitHub release or deploying docs:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_system \
  --ref <branch-name> \
  --field publish-target=testpypi \
  --field create-github-release=false \
  --field deploy-docs=false
```

The workflow always publishes in dependency order:

1. `op_system`
2. `flepimop2-op_system`

When testing from a branch, keep `create-github-release=false` and
`deploy-docs=false`. Release documentation is built only from the exact
`v<version>` tag.

### PyPI

Use this to publish both packages, create a GitHub release, and deploy the versioned documentation:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_system \
  --field publish-target=pypi \
  --field create-github-release=true \
  --field deploy-docs=true
```

When `create-github-release=true`, the workflow creates a single
`v<version>` GitHub release for the shared package version and uses GitHub's
generated release notes. A zero major version is not a prerelease: only explicit
alpha, beta, and release-candidate versions receive GitHub's prerelease flag.

## 4. Documentation Deployment

The release workflow calls `.github/workflows/gh-pages.yaml` as a reusable workflow when `deploy-docs=true`.

That workflow supports two deployment modes:

- `push`: the exact `main` commit is deployed only to mutable `dev`
- `release`: the exact tag is deployed to its full immutable version; final
  releases move `latest`, and alpha, beta, or release-candidate versions move
  `pre`

The workflow refuses to overwrite an existing exact version. A manual release
deployment must provide the corresponding `v<version>` tag in `release-ref`.

## 5. Trusted Publishing Setup

The publish jobs use PyPI Trusted Publishing rather than a stored API token.

For `publish-target=testpypi`, configure the TestPyPI trusted publisher entry for:

- Owner: `ACCIDDA`
- Repository: `op_system`
- Workflow file: `release.yaml`

For `publish-target=pypi`, configure the PyPI trusted publisher entry for the same repository and workflow file.

Because this repository publishes two distributions from one workflow, the same trusted publisher setup must be allowed to publish both package names on the selected index.

## 6. Current Packaging Note

The release workflow now handles the mechanics of validating, building, and sequencing both distributions together. Package-index acceptance still depends on the metadata inside each built distribution, so publishing `flepimop2-op_system` to a public index may still require follow-up packaging changes outside the workflow itself.

## 7. Software-Paper Archive

GitHub release and package publishing do not themselves confirm an archival
deposit. Follow the [citation and archiving plan](citation-and-archiving.md):
verify existing arrangements before creating a new record, archive the exact
shared release, and record a verified version DOI and any distinct concept DOI.
The paper should identify the tagged software and its independently retained
benchmark/environment evidence.

After Zenodo ingests a new release, verify its tag, version, creators, licenses,
source archive, exact DOI, and concept DOI. Submit the published record to the
ACCIDDA community for curator review. Account linking, repository enablement,
community submission, and acceptance remain manual operations.
