# flepimop2-op_system

`flepimop2-op_system` provides the `flepimop2` system adapter for `op_system`.

It packages the `flepimop2.system.op_system` provider so `flepimop2` can load and execute RHS specifications compiled by the core `op_system` package.

It also provides the `sparse_table` parameter module, which assembles a dense
array for a routing transition's matrix parameter (for example
`eta[time, imm:i, imm:j]`) from a declared support. Each entry is a number or
any nested parameter configuration, sampled with the request's leading axes:

```yaml
parameter:
  eta:
    module: sparse_table
    indices: [imm, imm]
    entries:
      - index: [x0, x3]
        value: {module: fixed, value: 0.2}
      - index: [x5, x7]
        value: 1.0
```

Off-support entries are zero. `SparseTableParameter.support(axes)` returns
the support positions for inference.
