## Safety & Validation

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTbOtdPNXMhNawe8lQ0qM1b3GPTeFkoN8LaMA&s" width="300" alt="mkdocs"> 
</p>

### Expression Security

Expressions are parsed with Python `ast` and restricted to:

- Arithmetic operations
- Named variables
- Selected NumPy math functions (`np.exp`, `np.log`, etc)

Disallowed operations (imports, attribute access, function injection) are rejected.

Validation errors raise descriptive built-in exceptions (`ValueError`, `TypeError`); no custom error wrapper is shipped.

---
