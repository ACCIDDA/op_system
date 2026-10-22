"""Optional bounded NumPy CPU lowering with explicit, reusable packed inputs.

Importing this module does not import Numba. Compilation is eager and never
falls back to Python. Generated functions have one homogeneous float signature;
normal compiler evaluators and other array namespaces remain independent.
"""

from __future__ import annotations

import ast
import copy
import importlib
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from op_system._vectorize import (
    _BufferTemplate,
    _compile_vector_expr,
    _CompiledExpression,
    _EqGroup,
    _VectorPlan,
    build_vector_plan,
)
from op_system.identity import StructuralModelIdentity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import CodeType

    import numpy.typing as npt

    from op_system.compile import CompiledRhs

    _Array = npt.NDArray[np.float32 | np.float64]

__all__ = ["BufferLayout", "CompilationSeconds", "NumbaRhs", "compile_numba"]

_FUNCTIONS = frozenset({
    "abs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tanh",
    "maximum",
    "minimum",
    "where",
    "sum",
    "permute_dims",
})
_NODES = (
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Tuple,
    ast.Slice,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BitAnd,
    ast.BitOr,
    ast.Invert,
)


@dataclass(frozen=True, slots=True)
class BufferLayout:
    """A named C-order slice in a homogeneous flat input array."""

    name: str
    shape: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        """Number of scalar entries, including one for scalar buffers."""
        return math.prod(self.shape)


def _input(value: object, shape: tuple[int, ...], dtype: str) -> _Array:
    """Refuse foreign arrays and implicit casts of explicitly typed inputs.

    Returns:
        A checked NumPy buffer or scalar converted to the declared precision.

    Raises:
        TypeError: Input type or explicit dtype differs.
        ValueError: Input shape differs.
    """
    if type(value) not in {np.ndarray, np.float32, np.float64, int, float}:
        msg = "Packed inputs require plain NumPy arrays or real Python scalars"
        raise TypeError(msg)
    if isinstance(value, (np.ndarray, np.floating)) and value.dtype != np.dtype(dtype):
        msg = f"Packed input dtype must be {dtype}; convert explicitly before packing"
        raise TypeError(msg)
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        msg = f"Packed input shape must be {shape}, got {result.shape}"
        raise ValueError(msg)
    return result


def _pack(
    layout: tuple[BufferLayout, ...], values: Mapping[str, object], dtype: str
) -> _Array:
    if set(values) != {buffer.name for buffer in layout}:
        msg = "Packed input names must match the declared buffer layout exactly"
        raise ValueError(msg)
    result = np.empty(sum(buffer.size for buffer in layout), dtype=dtype)
    for buffer in layout:
        result[buffer.offset : buffer.offset + buffer.size] = _input(
            values[buffer.name], buffer.shape, dtype
        ).reshape(-1)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class CompilationSeconds:
    """Wall times for source generation, optional import and eager compilation."""

    generation: float
    optional_import: float
    native_compilation: float


class _NativeKernel(Protocol):
    @property
    def nopython_signatures(self) -> Sequence[object]:
        """The one eagerly compiled native signature."""
        ...

    def __call__(self, t: object, y: object, p: object) -> _Array:
        """Execute using native packed NumPy inputs."""
        ...


@dataclass(frozen=True, slots=True)
class NumbaRhs:
    """One eagerly compiled, homogeneous, NumPy CPU RHS signature.

    ``kernel(t, y, p)`` accepts C-contiguous one-dimensional NumPy arrays in the
    layouts below. The signature also accepts read-only arrays. It checks lengths,
    does not mutate inputs and returns a fresh flat array. Call ``evaluate`` for
    strict Python-side time/dtype/layout validation, or use ``kernel`` from a
    Numba consumer after validating the same contract. Each call allocates its
    result and expression temporaries. No process-global or on-disk cache is used.

    ``pack_state`` and ``pack_parameters`` explicitly copy once into owned,
    read-only arrays. Reuse packed parameters while values stay fixed. No input
    arrays or training data are retained by this artifact.
    """

    model: StructuralModelIdentity
    dtype: Literal["float32", "float64"]
    state_names: tuple[str, ...]
    state_layout: tuple[BufferLayout, ...]
    parameter_layout: tuple[BufferLayout, ...]
    source: str = field(repr=False)
    compilation_seconds: CompilationSeconds
    kernel: _NativeKernel = field(repr=False, compare=False)

    def pack_state(self, values: Mapping[str, object]) -> _Array:
        """Copy structured state buffers into the declared C-order flat layout.

        Returns:
            Owned read-only NumPy state array.
        """
        return _pack(self.state_layout, values, self.dtype)

    def pack_parameters(self, values: Mapping[str, object]) -> _Array:
        """Copy scalar and shaped parameters into one reusable read-only array.

        Returns:
            Owned read-only NumPy parameter array; empty for parameter-free models.
        """
        return _pack(self.parameter_layout, values, self.dtype)

    def evaluate(self, t: object, y: object, parameters: object) -> _Array:
        """Validate packed inputs and execute the single compiled signature.

        Returns:
            Fresh flat NumPy derivative array in ``state_names`` order.

        Raises:
            TypeError: Inputs are foreign, differently typed or noncontiguous.
        """
        state = _input(y, (len(self.state_names),), self.dtype)
        params = _input(
            parameters, (sum(item.size for item in self.parameter_layout),), self.dtype
        )
        if not state.flags.c_contiguous or not params.flags.c_contiguous:
            msg = "Numba inputs must be C-contiguous; pack or copy explicitly"
            raise TypeError(msg)
        time = _input(t, (), self.dtype)[()]
        return self.kernel(time, state, params)


class _Rewrite(ast.NodeTransformer):
    """Rename bound symbols and retain homogeneous arithmetic precision."""

    def __init__(self, names: Mapping[str, str], dtype: str) -> None:
        self.names = names
        self.dtype = dtype

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if not isinstance(node, _NODES):
            msg = f"Numba lowering does not support {type(node).__name__}"
            raise NotImplementedError(msg)
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id not in self.names:
            msg = f"Numba lowering does not support binding {node.id!r}"
            raise NotImplementedError(msg)
        return ast.Name(id=self.names[node.id], ctx=ast.Load())

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(node.value):
                msg = "Numba arithmetic constants must be finite"
                raise NotImplementedError(msg)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr=self.dtype,
                    ctx=ast.Load(),
                ),
                args=[node],
                keywords=[],
            )
        if node.value is None or type(node.value) is bool:
            return node
        msg = "Numba lowering requires real numeric arithmetic constants"
        raise NotImplementedError(msg)

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        # Compiler-generated indices/slices are static configuration, not floats.
        if any(
            not isinstance(
                item,
                (ast.Tuple, ast.Slice, ast.Constant, ast.Load, ast.UnaryOp, ast.USub),
            )
            for item in ast.walk(node.slice)
        ):
            msg = "Numba lowering requires static array indices"
            raise NotImplementedError(msg)
        return ast.Subscript(
            value=self.visit(node.value), slice=node.slice, ctx=ast.Load()
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in _FUNCTIONS
        ):
            msg = "Numba lowering supports only the documented NumPy call subset"
            raise NotImplementedError(msg)
        function = node.func.attr
        if function == "sum":
            return self._sum(node)
        if function == "permute_dims":
            if len(node.args) != 2 or node.keywords:
                msg = "Numba transpose requires a static permutation"
                raise NotImplementedError(msg)
            permutation = ast.literal_eval(node.args[1])
            if not isinstance(permutation, tuple) or any(
                type(a) is not int for a in permutation
            ):
                msg = "Numba transpose requires a static permutation"
                raise NotImplementedError(msg)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="transpose",
                    ctx=ast.Load(),
                ),
                args=[self.visit(node.args[0]), node.args[1]],
                keywords=[],
            )
        if node.keywords:
            msg = "Numba elementwise calls do not accept keyword arguments"
            raise NotImplementedError(msg)
        return ast.Call(
            func=node.func, args=[self.visit(arg) for arg in node.args], keywords=[]
        )

    def _sum(self, node: ast.Call) -> ast.AST:
        if len(node.args) != 1 or any(
            k.arg not in {"axis", "keepdims"} for k in node.keywords
        ):
            msg = "Numba sum requires one input and static axis/keepdims options"
            raise NotImplementedError(msg)
        options = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        axis = options.get("axis")
        keep = options.get("keepdims", False)
        valid_axis = axis is None or (type(axis) is int and axis >= 0)
        if type(keep) is not bool or not valid_axis or (keep and axis is None):
            msg = "Numba sum requires one nonnegative axis; keepdims needs an axis"
            raise NotImplementedError(msg)
        value = ast.Call(
            func=node.func,
            args=[self.visit(node.args[0])],
            keywords=[]
            if axis is None
            else [ast.keyword(arg="axis", value=ast.Constant(axis))],
        )
        if keep:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="expand_dims",
                    ctx=ast.Load(),
                ),
                args=[value, ast.Constant(axis)],
                keywords=[],
            )
        return value


class _Expressions:
    """Capture bounded expression ASTs only during an explicit native request."""

    def __init__(self) -> None:
        self.sources: dict[CodeType, ast.expr] = {}

    def __call__(self, tree: ast.Expression, *, filename: str) -> _CompiledExpression:
        stack: list[tuple[ast.AST, int]] = [(tree, 0)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if count > 10_000 or depth > 64:
                msg = "Numba expression exceeds the node/depth limit"
                raise ValueError(msg)
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        result = _compile_vector_expr(tree, filename=filename)
        self.sources[result.code] = copy.deepcopy(tree.body)
        return result


def _scalar_plan(compiled: CompiledRhs, expressions: _Expressions) -> _VectorPlan:
    """Lower bounded scalar source IR when the tensor planner is inapplicable.

    Returns:
        Scalar aliases and equations with no tensor expansion.

    Raises:
        NotImplementedError: Source has shaped states or parameters.
    """
    from op_system._ir import unparse_ir  # ruff: ignore[import-outside-top-level]

    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if (
        rhs is None
        or len(compiled.state_names) > 128
        or rhs.shaped_params
        or any(t.axes for t in rhs.state_templates)
    ):
        msg = "Numba requires a supported tensor plan or at most 128 scalar states"
        raise NotImplementedError(msg)
    templates = tuple(
        _BufferTemplate(name, (), (), (name,), ({},), i)
        for i, name in enumerate(compiled.state_names)
    )
    aliases = tuple(
        (
            name,
            expressions(
                ast.parse(unparse_ir(expr), mode="eval"),
                filename="<numba_scalar_alias>",
            ),
            (),
        )
        for name, expr in rhs.aliases_ir.items()
    )
    groups = tuple(
        _EqGroup(
            name,
            (
                expressions(
                    ast.parse(unparse_ir(expr), mode="eval"),
                    filename="<numba_scalar_equation>",
                ),
            ),
            (),
            (),
            (),
            (),
            (),
            (),
        )
        for name, expr in zip(compiled.state_names, rhs.equations_ir, strict=True)
        if expr is not None
    )
    if len(groups) != len(templates):
        msg = "Numba requires complete scalar expression IR"
        raise NotImplementedError(msg)
    return _VectorPlan(templates, (), (), aliases, groups, len(templates))


class _Source:
    """Emit one straight-line function, retaining no runtime dictionaries."""

    def __init__(self, dtype: str, expressions: Mapping[CodeType, ast.expr]) -> None:
        self.dtype = dtype
        self.expressions = expressions
        self.names = {"t": "t"}
        self.counter = 0
        self.lines = ["def _kernel(t, y, p):"]

    def bind(self, name: str, expression: str) -> str:
        variable = f"_v{self.counter}"
        self.counter += 1
        self.names[name] = variable
        self.lines.append(f"    {variable} = {expression}")
        return variable

    def expression(self, code: _CompiledExpression) -> str:
        body = _Rewrite(self.names, self.dtype).visit(
            copy.deepcopy(self.expressions[code.code])
        )
        return ast.unparse(ast.fix_missing_locations(body))

    def generate(self, plan: _VectorPlan, parameters: tuple[BufferLayout, ...]) -> str:
        count = sum(p.size for p in parameters)
        self.lines.extend([
            f"    if y.size != {plan.n_state} or p.size != {count}:",
            "        raise ValueError('Numba input lengths differ from layout')",
        ])
        for buffer in parameters:
            value = (
                f"p[{buffer.offset}]"
                if not buffer.shape
                else (
                    f"p[{buffer.offset}:{buffer.offset + buffer.size}]"
                    f".reshape({buffer.shape!r})"
                )
            )
            self.bind(buffer.name + ("_buf" if buffer.shape else ""), value)
        for template in plan.state_templates:
            variable = self.bind(
                template.base + "_buf",
                (
                    f"y[{template.offset}:"
                    f"{template.offset + math.prod(template.shape)}]"
                    f".reshape({template.shape!r})"
                ),
            )
            if not template.shape:
                self.names[template.base] = variable
        for name, code, shape in plan.alias_codes:
            variable = self.bind(
                name + "_buf",
                f"np.broadcast_to(np.asarray({self.expression(code)}), {shape!r})",
            )
            self.names[name] = variable
        for name, code in plan.cse_codes:
            self.bind(name, self.expression(code))
        pieces = []
        for index, group in enumerate(plan.eq_groups):
            if group.unroll_axes or len(group.codes) != 1:
                msg = "Numba requires fully vectorized equations without unrolling"
                raise NotImplementedError(msg)
            expression = self.expression(group.codes[0])
            variable = f"_out{index}"
            self.lines.append(
                f"    {variable} = np.ascontiguousarray(np.broadcast_to("
                f"np.asarray({expression}), {group.full_shape!r})).reshape(-1)"
            )
            pieces.append(variable)
        self.lines.append(f"    return np.concatenate(({', '.join(pieces)},))")
        source = "\n".join(self.lines) + "\n"
        if len(source) > 100_000:
            msg = "Numba generated source exceeds the 100000-character limit"
            raise ValueError(msg)
        return source


def compile_numba(  # ruff: ignore[too-many-locals]
    compiled: CompiledRhs, *, dtype: Literal["float32", "float64"] = "float64"
) -> NumbaRhs:
    """Eagerly lower a bounded scalar/fully vectorized model to NumPy CPU code.

    Arithmetic, static indexing, elementwise abs/exp/log/sqrt/sin/cos/tanh,
    maximum/minimum/where, transposes and single-axis sums are supported. No
    fastmath, dynamic evaluator closure, automatic accelerator selection or fallback
    is used. State/parameter scalar counts are capped at one million each and
    generated source at 100000 characters. Runtime input values are not captured.

    Args:
        compiled: Natural-coordinate compiler artifact with retained source IR.
        dtype: One homogeneous native floating precision.

    Returns:
        Artifact with one eagerly compiled signature, source and explicit layouts.

    Raises:
        ImportError: Install ``op_system[numba]`` to enable the optional compiler.
        ValueError: Precision/size bounds differ or Numba JIT is disabled.
        NotImplementedError: The model needs an unsupported operation or layout.
    """
    started = time.perf_counter()
    if dtype not in {"float32", "float64"}:
        msg = "Numba dtype must be float32 or float64"
        raise ValueError(msg)
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if (
        rhs is None
        or compiled.operators
        or compiled.history_requirements
        or compiled.gauge_fix is not None
        or rhs.time_varying_params
    ):
        msg = (
            "Numba requires natural source IR without operators, history, "
            "gauge fixing or time tables"
        )
        raise NotImplementedError(msg)
    if not 0 < len(compiled.state_names) <= 1_000_000:
        msg = "Numba state count must be between 1 and 1000000"
        raise ValueError(msg)
    expressions = _Expressions()
    plan = build_vector_plan(rhs, compile_expression=expressions)
    if plan is None:
        plan = _scalar_plan(compiled, expressions)
    if plan.param_templates:
        msg = "Numba requires a supported scalar or fully vectorized template plan"
        raise NotImplementedError(msg)
    layouts = [BufferLayout(name, (), i) for i, name in enumerate(compiled.param_names)]
    offset = len(layouts)
    for name, _, shape in plan.extra_param_buffers:
        layouts.append(BufferLayout(name, shape, offset))
        offset += math.prod(shape)
    if offset > 1_000_000:
        msg = "Numba packed parameter count exceeds 1000000"
        raise ValueError(msg)
    parameters = tuple(layouts)
    source = _Source(dtype, expressions.sources).generate(plan, parameters)
    model = StructuralModelIdentity.from_compiled(compiled)
    generated = time.perf_counter()
    try:
        numba = importlib.import_module("numba")
    except ImportError as exc:
        msg = "Optional Numba lowering requires installation with op_system[numba]"
        raise ImportError(msg) from exc
    imported = time.perf_counter()
    if numba.config.DISABLE_JIT:
        msg = "Numba lowering requires JIT; NUMBA_DISABLE_JIT is enabled"
        raise ValueError(msg)
    namespace: dict[str, object] = {
        "np": np,
        "__builtins__": {"ValueError": ValueError},
    }
    exec(compile(source, "<op_system_numba>", "exec"), namespace)  # ruff: ignore[exec-builtin]
    scalar = getattr(numba, dtype)
    array = numba.types.Array(scalar, 1, "C", readonly=True)
    signature = scalar[::1](scalar, array, array)
    kernel = numba.njit(
        signature, nogil=True, fastmath=False, error_model="numpy", boundscheck=True
    )(namespace["_kernel"])
    return NumbaRhs(
        model,
        dtype,
        compiled.state_names,
        tuple(BufferLayout(t.base, t.shape, t.offset) for t in plan.state_templates),
        parameters,
        source,
        CompilationSeconds(
            generated - started, imported - generated, time.perf_counter() - imported
        ),
        cast("_NativeKernel", kernel),
    )
