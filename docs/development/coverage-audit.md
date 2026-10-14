# Readiness coverage audit

Issue #29's original inventory predates the provider suite and the typed-IR
compiler. This audit maps its requested behavior to current regression tests.
The tests protect observable contracts; the table does not claim 100% line or
branch coverage or numerical validity of every user model.

Paths below are relative to the repository root. Source and installed-wheel
suites exercise these contracts; mandatory backend jobs additionally reject
missing or skipped selected tests. See [contributing](contributing.md) for the
local commands and [backend qualification](../guides/backend-qualification.md)
for the qualified CPU precision, layout, and AD boundaries.

| Requested area | Evidence |
| --- | --- |
| Mixing kernel generation | `flepimop2-op_system/tests/test_mixing_kernels.py`: independent scalar formulas for all five forms, rectangular source/target grids, one-axis square grids, custom asymmetric matrix orientation, scalar no-axis policy, flat/structured injection with independent nonuniform quadrature. `test_erfc_kernel.py` checks fixed numerical values and positive tails (#274). |
| Multi-axis and categorical/ordinal/continuous axes | `tests/op_system/test_op_system_specs.py`, `test_op_system_templates.py`, `test_time_interpolation.py`: generated linear/log grids, coordinate validation, multi-axis selectors, pinned coordinates and singleton time axes. `test_axis_edge_contracts.py` rejects empty coordinate lists for each type while retaining scalar `axes: []`. |
| Sum and integration helpers | `test_op_system_apply_along.py`, `test_op_system_ir_lower.py`, `test_renewal_births.py`: helper composition, filters, categorical sums, trapezoidal weights, mismatched axis-type errors, and births. Required `test_backend_qualification.py` checks nested mixed categorical/continuous reductions against an analytic bilinear integral on nonuniform grids, including precision and JAX/Torch derivatives (#283). |
| Hazard-rate transitions and conservation | `test_routing_transitions.py`, `test_axis_kernel.py`, `test_jump_integral.py`, `test_renewal_births.py`: independently computed cell flows, batch balance, source/target orientation and open/closed-system invariants. Required backend cases compare independent block, mixed-selector and global-total flows (#170, #113, #260). |
| Provider construction and metadata | `flepimop2-op_system/tests/test_system.py`: bind/step interfaces, flat/structured/block parity, flatten/unflatten helpers, axis order/sizes/coordinates/types/labels, parameter requests, time tables, operators and reactions. `test_sparse_table.py` covers support assembly. `scripts/qualify_provider_types.py` gates the installed wheel's public types (#60). |
| Failure paths | `test_op_system_specs.py`, `test_op_system_compile.py`, `test_bare_state_references.py`, `test_mixed_template_selectors.py`: malformed declarations, missing/indexed states, allowlisted ASTs and legacy-selector diagnostics. Provider kernel tests check unknown axes/forms, missing form parameters and missing custom values. |

The audit exposed and separately fixed unavailable `numpy.erfc` (#274), untyped
literal precision (#277), and incompatible/promoting continuous weights (#283).
The broader inventory therefore checks actual matrix values and derivative
behavior rather than accepting compilation or metadata shape alone.

Gamma kernels retain their existing **unnormalized** shape with distance clipped
to `1e-12` at zero. No-axis analytic declarations retain scalar `params.value`
(default zero); callers must declare axes to obtain a generated matrix.
Cross-axis `constraints` are preserved as reserved metadata by public
normalization; private parser tests do not establish public validation or
masking. Filtering and reduction support remain tracked in #37/#38. Optional GPU paths and large timing runs are
not qualified by the required CPU matrix.
