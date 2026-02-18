## Design Principles

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAPUxG5Hza7gFsxIqfjC2MpZ3LYbvzRQ7k2w&s" width="400" alt="mkdocs"> 
</p>

### Domain Agnostic

`op_system`:

- Does NOT import flepimop2
- Does NOT depend on op_engine
- Does NOT impose solver semantics

It only defines RHS structure and compilation.

---

### Backend Friendly

Compiled RHS functions are compatible with:

- op_engine
- scipy.integrate
- custom solvers
- GPU backends (via wrapping)

---

### Forward Compatible

Reserved fields are preserved during normalization and exposed via `NormalizedRhs.meta` so future multiphysics extensions and adapters can read axes, mixing kernels, and operator metadata without breaking the API.

### Adapters

- The flepimop2 system adapter can consume inline specs, builds `mixing_kernels` from `NormalizedRhs.meta["mixing"]`, and exposes them for downstream engines (e.g., op_engine adapter). Operator metadata is preserved in meta but not yet wired into op_engine IMEX.

---