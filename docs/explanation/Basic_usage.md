## Basic Usage

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTf4FpVvzboK03mfLytPcOjb1kFOxzgpeoa6w&s" width="300" alt="mkdocs"> 
</p>

### Option A — One-step API 

```python
from op_system import compile_spec

compiled = compile_spec(spec)

dydt = compiled.eval_fn(
    t=0.0,
    y=[999.0, 1.0, 0.0],
    beta=0.3,
    gamma=0.1,
)
```

---

### Option B — Two-step API (advanced control)

```python
from op_system import normalize_rhs, compile_rhs

rhs = normalize_rhs(spec)
compiled = compile_rhs(rhs)
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---

### Option C — Working with YAML specifications

While `compile_spec` doesn't directly load files, you can easily load YAML/JSON files using standard Python libraries:

```yaml
# sir.yaml
kind: expr
state: [S, I, R]
aliases:
  N: "S + I + R"
equations:
  S: "-beta * S * I / N"
  I: "beta * S * I / N - gamma * I"
  R: "gamma * I"
```

```python
import yaml
from op_system import compile_spec

with open("sir.yaml") as f:
    spec = yaml.safe_load(f)

compiled = compile_spec(spec)
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---