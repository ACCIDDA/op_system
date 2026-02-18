# Welcome to `op_system`

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-ValQaMH70OOs7ARZ13i1s5s-Bay0djmlbA&s" width="400" alt="mkdocs"> 
</p>

**Domain-agnostic RHS specification and compilation utilities for scientific simulation systems.**

`op_system` provides a lightweight, backend-neutral framework designed to bridge the gap between high-level scientific models and high-performance numerical engines. It allows researchers to define complex system dynamics in a human-readable format while ensuring the resulting code is optimized for computation.

* * *

## Key Capabilities


The library streamlines the simulation workflow through three primary phases:

1.  **Define**: Specify system dynamics using YAML or JSON-friendly dictionaries, supporting both direct mathematical equations and transition-based compartmental models.
    
2.  **Normalize**: Transform these specifications into a structured, validated representation that preserves critical metadata like spatial axes and mixing kernels.
    
3.  **Compile**: Generate fast, callable right-hand-side (RHS) functions compatible with various numerical solvers, including GPU-backed engines.
    

* * *

## Core Pillars of the Framework

### Flexible RHS Specification

Choose the dialect that fits your domain:

*   **Expression Style (`kind: expr`)**: Directly define derivatives using mathematical strings.
    
*   **Transition Style (`kind: transitions`)**: Define flows between compartments, which the system automatically converts into balanced conservation-law equations.
    

### Advanced Metadata Preservation

# 

Beyond simple equations, `op_system` manages the "connective tissue" of complex models:

*   **Axes**: Support for categorical and continuous dimensions with built-in discretization strategies.
    
*   **Mixing Kernels**: Normalized interactions for spatial or group-based dynamics (e.g., Gaussian or Power Law).
    
*   **Operator Support**: Future-facing metadata for IMEX (Implicit-Explicit) integration and operator splitting.
    

* * *

## Project Status

# 

*   **Current Version**: `0.1.0`
    
*   **Scope**: Supports ODE RHS, algebraic expressions, and compartment transitions.
    
*   **Integrations**: Designed to work standalone or alongside `op_engine` and `flepimop2`.