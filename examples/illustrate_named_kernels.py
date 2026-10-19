"""Illustrate a rectangular named-axis kernel and its compiled derivatives.

Regenerate with ``uv run --with matplotlib==3.11.2 python
examples/illustrate_named_kernels.py``; ``--check`` needs no plotting dependency.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from illustration_artifacts import IllustrationData, build_artifacts
from named_kernel_products import model_spec, reference

from op_system import compile_spec

if TYPE_CHECKING:
    from collections.abc import Sequence


def _render(output: Path, kernel: np.ndarray, drift: np.ndarray) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"svg.hashsalt": "op-system-named-kernels", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
    image = axes[0].imshow(kernel, cmap="Blues", vmin=0, vmax=0.5)
    for row, column in np.ndindex(kernel.shape):
        axes[0].text(
            column,
            row,
            f"{kernel[row, column]:g}",
            ha="center",
            va="center",
            color="white" if kernel[row, column] > 0.3 else "#243746",
        )
    axes[0].set(
        xticks=range(3),
        xticklabels=["r0", "r1", "r2"],
        yticks=range(2),
        yticklabels=["c0", "c1"],
        xlabel="Resource axis",
        ylabel="Consumer axis",
        title="K[consumer, resource]",
    )
    figure.colorbar(image, ax=axes[0], shrink=0.65)
    axes[1].bar(range(5), drift, color=["#0d6b6e"] * 3 + ["#2471a3"] * 2)
    axes[1].axhline(0, color="#526674", lw=0.7)
    axes[1].set(
        xticks=range(5),
        xticklabels=["P[r0]", "P[r1]", "P[r2]", "Z[c0]", "Z[c1]"],
        ylabel="Derivative (arbitrary units / time)",
        title="Evaluation at one fixed state",
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "figure.svg", metadata={"Date": None})
    svg = output / "figure.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    figure.savefig(
        output / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/named-kernels",
    )
    args = parser.parse_args(argv)
    # Dyadic inputs make this tiny reference exact in binary floating point.
    state = np.asarray([1.0, 2.0, 4.0, 0.5, 1.5])
    params = {
        "K": np.asarray([[0.5, 0.25, 0.0], [0.125, 0.25, 0.5]]),
        "g": np.asarray([0.5, 0.75]),
        "efficiency": np.asarray([0.25, 0.5]),
        "loss": np.asarray([0.125, 0.25]),
        "linear": np.asarray([1.0, 0.0]),
    }
    compiled = compile_spec(model_spec())
    drift = compiled.eval_fn(0.0, state, **params)
    np.testing.assert_array_equal(drift, [-0.265625, -0.6875, -2.25, 0.0625, 1.40625])
    np.testing.assert_array_equal(drift, reference(state, params))
    labels = ["P[r0]", "P[r1]", "P[r2]", "Z[c0]", "Z[c1]"]
    lines = ["| State | Value | Compiled derivative |", "| --- | ---: | ---: |"]
    lines.extend(
        f"| {name} | {value:g} | {change:g} |"
        for name, value, change in zip(labels, state, drift, strict=True)
    )
    build_artifacts(
        args.output,
        IllustrationData(
            "\n".join(lines) + "\n",
            {
                "state": state.tolist(),
                "parameters": {name: value.tolist() for name, value in params.items()},
                "drift": drift.tolist(),
            },
        ),
        [Path(__file__), Path(__file__).with_name("named_kernel_products.py")],
        lambda output: _render(output, params["K"], drift),
        check=args.check,
    )


if __name__ == "__main__":
    _main()
