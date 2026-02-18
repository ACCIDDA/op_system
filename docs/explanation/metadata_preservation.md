## Metadata Preservation

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMwTP8Qiq9b-CRgKE77YNgNLaZxVvbiDgs3A&s" width="300" alt="mkdocs"> 
</p>

---
The `NormalizedRhs` object (returned by `normalize_rhs`) includes a `meta` field that contains:

- `axes`: Normalized axis definitions with resolved coords and sizes
- `state_axes`: Mapping of state variables to their axis dependencies
- `mixing`: Normalized mixing kernel specifications
- `operators`: Operator metadata for future IMEX integration
- Reserved blocks: `sources`, `couplings`, `constraints` (passthrough for future use)

This metadata is preserved for consumption by adapters and orchestration layers.

### Example: Accessing Metadata

```python
from op_system import normalize_rhs

spec = {
    "kind": "expr",
    "state": ["S", "I"],
    "equations": {"S": "-beta * S * I", "I": "beta * S * I - gamma * I"},
    "axes": [
        {"name": "space", "type": "continuous", "domain": {"lb": 0, "ub": 10}, "size": 5},
    ],
    "mixing": [
        {"name": "K", "axes": ["space", "space"], "form": "gaussian", "params": {"sigma": 1.5, "scale": 1.0}},
    ],
}

rhs = normalize_rhs(spec)

# Access normalized metadata
axes_meta = rhs.meta["axes"]        # List of normalized axis definitions
mixing_meta = rhs.meta["mixing"]    # List of normalized mixing kernels
```

---