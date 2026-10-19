# Scaling gauges

This feature is unreleased; examples require a source checkout.

A positive scaling action can change state/parameter units while leaving the
model's observed outputs unchanged. Declare it next to the equations, verify
it at compilation, and optionally bind scalar parameters to select one
representative. This analysis concerns deterministic equations and outputs;
it does not establish structural identifiability.

```python
--8 < --"examples/scaling_gauges.py"
```

For this example, replacing `x` and `K` by `s*x` and `s*K`, with `s > 0`,
multiplies `dx/dt` by `s` and preserves the output `x/K`. Choosing `K = 1`
uses `x/K` as the state in the fixed representative.

| Quantity | Original values | Fixed representative |
| --- | ---: | ---: |
| State `x` | 6 | 0.5 |
| Scale parameter `K` | 12 | 1, bound at compilation |
| Rate | 0.5 | 0.5 |
| Derivative | 1.5 | 0.125 |
| Observed fraction | 0.5 | 0.5 |

Use `gauge_fix.encode(initial_state, original_parameters)` for each original
parameter draw and check its native `valid` flag. Integrate the returned state
using the fixed model and remaining parameters. The model rejects runtime
overrides of its anchors. Its flat, PyTree, block-PyTree and observable
interfaces bind the same constants. Scalar anchors may occur only in the
observation model. `param_names` and observable parameter shapes omit bound
anchors; available parameter declarations mark them fixed.

`encode` requires real floating states and scalar, positive anchors. It has
no hidden floor or clamp. Separate draws can be mapped with JAX `vmap`; the
map and evaluators also have NumPy and CPU PyTorch checks, with native AD
checks where supported. Remaining shaped parameters retain their arrays.
One scale is shared across every coordinate of a named state/parameter base.
This does not introduce one free generator per cell.

A quotient alone cannot reconstruct the discarded scale. `decode` takes the
full `GaugeCoordinates`, including the original group coordinates, to recover
that original state and parameter point. Retain these coordinates if an inverse
map is needed. Pickling a compiled model retains its analysis/fixing options
and rebuilds the evaluators.

## Verification and detection

Compiling declared gauges probes RHS covariance and **every declared observable's
invariance**. A failed comparison raises a diagnostic naming the gauge/output.
`CompiledRhs.gauges.verification` records the seed (19471), four points per
generator, positive scales (0.5 and 2), relative tolerance `1e-9`, absolute
tolerance `1e-11`, checked outputs and largest absolute residual. Points use
positive uniform state/parameter values in `[0.6, 1.4]` and times in `[0.1, 0.9]`;
unscaled bounded parameters are sampled inside their support. Finite probes are
not a proof across a continuous domain.

`detect_gauges=True` additionally constructs rational homogeneity constraints
from supported expression structure and computes an exact integer nullspace.
It handles supported monomials, additive degree matching, ratios, powers and
homogeneous reductions, along with specified native math rules. The report
separates detected directions from declarations and records whether those
rules also verify the declarations. Detection is conservative: identities
requiring algebraic cancellation can hide additional directions. Unsupported
operations fail an explicit detection request rather than reporting an empty
group. Ordinary undeclared compilation does not run this analysis; `gauges`
is `None`, which differs from an analyzed rank-zero action.

`rank` is the reported action's rank on states and parameters together;
`parameter_rank` is its parameter projection. Duplicate generators do not
increase rank. `invariant_coordinates` contains integer-exponent monomials
in shared parameter bases. These monomials need not be a complete coordinate
basis for shaped arrays, whose within-array ratios can carry more invariants.
The anchored map transforms the complete arrays and retains their entries.
Neither these ranks nor the monomial count is an identifiable-dimension estimate.

## Assumptions and limits

The action must transform initial states along with parameters. Known initial
conditions, priors or other observations can remove a freedom even if the RHS
is covariant. Preset `initial_state` specifications are rejected during fixing:
remove that preset and explicitly encode it with its original parameter values.
A fixed or bounded parameter cannot participate in the reported global positive
scaling action. Anchors must be rank-many independent scalar parameters with
finite positive targets; invariant declared outputs are required for fixing.

Named-axis fields and expression-based nonlocal reductions can participate in
the shared action when their operations have supported homogeneity rules.
Separate operator descriptors, history providers and time tables require
additional scaling contracts and are rejected by this analysis. This prevents an instantaneous
RHS check from being presented as verification of omitted dynamics. Ordinary
compilation of those model classes remains available.

Gauge-fixed models omit stochastic reaction exports and report
`gauge_fix_requires_jump_contract` gaps. Deterministic drift covariance alone
does not preserve a jump process's distribution; changing state units also
changes the meaning of a unit jump. A stochastic consumer must reject these
gaps. A declared but unfixed model retains its original natural-state reactions.

## Validation reports

```shell
python -m op_system.validate model.yml --detect-gauges --json
python -m op_system.validate model.yml --gauge-fix K=1
```

Repeat `--gauge-fix` for independent anchors. CLI options apply to each selected
specification. `validate_spec` accepts the same keyword options and reports
counts under `gauges`. Parameter-coordinate counts after removing the reported
action describe a generic nonzero parameter point. Missing symmetries, fixed
initial conditions and observation assumptions still require separate analysis;
the report explicitly leaves identifiability unassessed. Counts cover equation
and declared-observable parameters; free initial-condition seeds or noise-model
parameters supplied elsewhere are outside this count.

The report's `parameters` mapping continues to describe equation/operator
inputs. Observable-only arguments are listed in each compiled observable's
parameter shapes; the gauge parameter count includes them.
