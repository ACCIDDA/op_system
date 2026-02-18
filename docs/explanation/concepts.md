## Core Concept

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSN7IbPo3a1hORzrbNH-oq7d15rY2RmpYocHg&s" width="400" alt="mkdocs"> 
</p>


# Comprehensive Explanation to `op_system` Core Concepts

## 

The `op_system` library is designed as a domain-agnostic layer for defining right-hand-side (RHS) specifications for ordinary and partial differential equations (ODEs/PDEs). Its primary goal is to allow researchers to define complex biological or physical systems in a human-readable format (YAML/JSON-friendly dictionaries) which are then normalized and compiled into optimized evaluation functions.

## 1\. RHS Specification: The Logic of System Dynamics

## 

At the heart of any simulation is the definition of how state variables change over time. `op_system` provides two distinct "dialects" for this purpose. This duality ensures that the library can cater to both mathematicians who think in terms of pure equations and epidemiologists or ecologists who think in terms of transitions between compartments.

### A. Expression Style (`kind: expr`)

## 

The **Expression Style** is the most direct way to define a system. It maps one-to-one to the mathematical representation of a derivative.

#### Mathematical Foundation

## 

In a system of differential equations, we define:

$$\frac{dy}{dt} = f(t, y, \theta)$$

Where $y$ is the state vector and $\\theta$ represents parameters. In the `expr` style, you are explicitly defining the function $f$.

#### Implementation Logic

## 

When using `kind: expr`, you provide a mapping of state variables to their respective derivative expressions.

*   **State List**: Defines the order and names of variables (e.g., `["S", "I", "R"]`).
    
*   **Equations**: A dictionary where keys are state variables and values are strings representing the math.
    

#### Example Deep Dive:

## Python

    spec = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "equations": {
            "S": "-beta * S * I / N",
            "I": "beta * S * I / N - gamma * I",
            "R": "gamma * I",
        },
    }

In this configuration:

1.  **Aliases** act as intermediate variables to simplify equations and reduce redundant calculations.
    
2.  **Safety**: These strings are not simply "evaled." They are parsed using Python's Abstract Syntax Tree (`ast`) to ensure only safe arithmetic and permitted NumPy functions are executed.
    

* * *

### B. Transition Style (`kind: transitions`)

## 

The **Transition Style** is a higher-level abstraction. Instead of defining the net change for a variable, you define the "flow" between variables. This is particularly useful for compartmental models (like SIR or SEIR) where mass must be conserved.

#### The Conservation Law Logic

## 

In the transition style, you describe "hazards" or "rates." If a transition is defined from state $A$ to state $B$ with rate $r$:

* State $A$ loses mass: $\frac{dA}{dt} = -r \cdot A$

* State $B$ gains mass: $\frac{dB}{dt} = +r \cdot A$
    

#### Implementation Logic

## 

The user provides a list of `transitions`. Each transition object identifies a `from` state, a `to` state, and the `rate` of the transition.

#### Internal Conversion Example:

## 

Consider the following specification:

Python

    "transitions": [
        {"from": "S", "to": "I", "rate": "beta * I / N"},
        {"from": "I", "to": "R", "rate": "gamma"},
    ]

The `op_system` engine internally converts this into the following conservation-law equations to be passed to the numerical solver:

*   `dS/dt -= (beta * I / N) * S`
    
*   `dI/dt += (beta * I / N) * S - gamma * I`
    
*   `dR/dt += gamma * I`
    

This approach drastically reduces human error. In a complex model with 20+ compartments, manually balancing `expr` equations is prone to "leaky" models where mass is accidentally created or destroyed. The `transitions` style guarantees a closed system.

* * *

## 2\. Axes: Managing Dimensionality

## 

Modern scientific simulations are rarely just a few scalar variables. They are often "vectorized" across multiple dimensions such as age groups, geographic locations, or risk categories. `op_system` handles this through **Axes**.

### Categorical vs. Continuous

## 

*   **Categorical Axes**: These represent discrete groups (e.g., `age_groups: [child, adult, senior]`).
    
*   **Continuous Axes**: These represent physical space or continuous gradients. These require discretization before they can be used in a numerical engine.
    

### Discretization Strategies

## 

For continuous axes, `op_system` allows for sophisticated definition patterns:

1.  **Explicit Coords**: Manually providing every point on the axis.
    
2.  **Domain + Size + Spacing**: Defining a range (e.g., `[0, 100]`), the number of points (e.g., `size: 50`), and the growth rate (`linear`, `log`, or `geom`).
    

### Architectural Significance

## 

When the system is normalized, these axis definitions are resolved and stored in the `NormalizedRhs.meta["axes"]` dictionary. This ensures that downstream adapters (like those for `flepimop2`) know exactly how to reshape flat state vectors into multi-dimensional arrays for calculation.

* * *

## 3\. Mixing Kernels: Modeling Interactions

## 

In many systems, the rate of change in one "cell" (e.g., a city) depends on the state of other "cells" (e.g., neighboring cities). This is defined as **Mixing**.

### The Kernel Concept

## 

A mixing kernel defines the weight of interaction between points on an axis. `op_system` supports several standard mathematical forms for these interactions:

*   **Gaussian**: For proximity-based decay.
    
*   **Exponential**: For rapid decay over distance.
    
*   **Power Law**: For long-range interactions (common in human mobility).
    
*   **Custom**: Allows users to pass validated parameters for specialized forms.
    

### Validation and Normalization

## 

Mixing blocks are not just strings; they are validated against their required parameters (e.g., a `gaussian` form must provide a `sigma` and `scale`). Once validated, they are stored in `meta["mixing"]`, ready for the `op_engine` to build "mixing kernels"—large matrices that can be multiplied by the state vector to calculate spatial spreads efficiently.

* * *

## 4\. Operators: Preparing for Advanced Solvers

## 

The `operators` field is a "forward-facing" feature of the `op_system` architecture. While basic ODEs can be solved with explicit methods (like Runge-Kutta), complex systems often require IMEX (Implicit-Explicit) schemes or operator splitting.

*   **Logic**: By tagging certain parts of the specification as "operators," you are telling the system which parts of the math are linear (which can be solved implicitly for stability) and which are non-linear.
    
*   **Status**: Currently, `op_system` normalizes and preserves this metadata in `meta["operators"]`, ensuring that when future versions of the engine support these advanced solvers, the specifications remain compatible.