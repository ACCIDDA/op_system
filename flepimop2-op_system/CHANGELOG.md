# Changelog

Current and future changes to `flepimop2-op-system` are recorded here. Core
`op-system` releases have an independent lifecycle and do not publish this
provider.

## [Unreleased]

- Forward immutable `impulses`, `impulse_times` and map-only parameter requests.
  Expose `impulse_rhs` for engines that explicitly validate and execute the
  schedule. Ordinary steppers reject impulse models, and structured callbacks
  are withheld, so older engines cannot silently ignore these state maps.

- Replace issue breadcrumbs in the sparse-table provider documentation with
  current supported behavior.

## [0.9.0] - 2026-10-19

This was the last shared core/provider release. Its full history remains in the
[repository changelog](../CHANGELOG.md#090---2026-10-19).
