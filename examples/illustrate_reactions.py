"""Illustrate the checked frozen-state compiler example, without integrating it.

Regenerate with Python 3.12 and Matplotlib: uv run --with matplotlib==3.11.2
python examples/illustrate_reactions.py. Use --check without plotting dependencies.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from illustration_artifacts import IllustrationData, build_artifacts
from independent_reactions import run_example

if TYPE_CHECKING:
    from collections.abc import Sequence


def _render(output: Path, before: list[float], after: list[float]) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"svg.hashsalt": "op-system-transfer", "font.size": 10})
    figure, axis = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
    positions = np.arange(2)
    axis.bar(
        positions - 0.18,
        -np.asarray(before[:2]),
        0.36,
        color="#24678e",
        label="t = 0.5, before change",
    )
    axis.bar(
        positions + 0.18,
        -np.asarray(after[:2]),
        0.36,
        color="#0d6b6e",
        label="t = 1, at change",
    )
    axis.set(
        xticks=positions,
        xticklabels=["replicate a", "replicate b"],
        ylabel="Transfer propensity (counts / time)",
        title="Compiler evaluation at one fixed state",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "figure.svg", metadata={"Date": None})
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
        default=Path(__file__).resolve().parents[1] / "docs/assets/compiled-transfer",
    )
    args = parser.parse_args(argv)
    result = run_example()
    before = cast("list[float]", result["drift_before"])
    after = cast("list[float]", result["drift_after"])
    population = cast("list[float]", result["population"])
    lines = [
        "| Evaluation | Replicate | dS/dt | dI/dt | Population |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, drift in [("t = 0.5", before), ("t = 1", after)]:
        lines.extend(
            f"| {label} | {name} | {drift[i]:zg} | {drift[i + 2]:zg} | "
            f"{population[i]:g} |"
            for i, name in enumerate(["a", "b"])
        )
    summary = "\n".join(lines) + "\n"
    build_artifacts(
        args.output,
        IllustrationData(
            summary, {"before": before, "after": after, "population": population}
        ),
        [Path(__file__), Path(__file__).with_name("independent_reactions.py")],
        lambda output: _render(output, before, after),
        check=args.check,
    )


if __name__ == "__main__":
    _main()
