# Contributing

The repository's [contributor guide](https://github.com/ACCIDDA/op_system/blob/main/CONTRIBUTING.md)
covers clean setup, compiler/provider boundaries, focused behavior tests,
required local checks, wheel verification, support, and releases.

Run baseline `just ci`, mandatory CPU backend `just qualify`, and installed-wheel
`just build-all` checks before pushing a PR. Optional GPU and large benchmark
runs are separate from that required matrix. The
[code conventions](code-style.md),
[backend qualification guide](../guides/backend-qualification.md), and
[numerical qualification guide](../guides/numerical-qualification.md) give the
contracts those checks protect.

For versioned citations and the software-paper archive plan, see
[citation and archiving](citation-and-archiving.md). The
[release workflow](creating-a-release.md) publishes only the core; the provider
has a separate lifecycle.
