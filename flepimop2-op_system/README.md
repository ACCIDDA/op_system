# flepimop2-op_system

`flepimop2-op_system` provides the `flepimop2` system adapter for `op_system`.

It packages the `flepimop2.system.op_system` provider so `flepimop2` can load and execute RHS specifications compiled by the core `op_system` package.

## Compatibility

This version (`0.3.0`) requires `flepimop2 >= 0.3.0.dev0` (the consolidated
`ModuleBase` API).

## Changelog

### 0.3.0

- Adopted the consolidated `flepimop2.module.ModuleBase` API. The
  `OpSystemSystem` connector now declares its `module` discriminator
  explicitly as a `Literal[...]` field rather than via the
  `module="..."` class-keyword shortcut.
- Switched the connector's `model_config` to `extra="forbid"`. Unknown
  top-level keys are rejected; engine-specific knobs continue to live
  inside `spec`.
- Verified compatibility with the new `ModuleBase.patch(...)` method and
  the `flepimop2 patch` CLI; added regression tests for both the
  `REPLACE` mode (recompiles the RHS) and the type-mismatch guard.
- Bumped `flepimop2` floor from `>=0.2.0` to `>=0.3.0.dev0`.
