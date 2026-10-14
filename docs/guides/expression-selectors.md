# Expression selectors and reduction bindings

Within an expression, `X[loc, age:ap]` keeps the current `loc` selection while
binding the age position to the reduction variable `ap`. For example:

```text
total[loc] = apply_along(X[loc, age:ap], age=ap, kernel=sum)
```

`loc` comes from the alias's left-hand side; `ap` visits the declared age
coordinates. Nested reductions can bind additional axes without dropping the
unreduced group selector. The resulting total stays local to each group, so a
group block evaluator can retain its independence.

Use `axis:variable` inside expression brackets and `axis=variable` in the
reducer's keyword arguments. Historical examples that put `axis=variable`
inside a state subscript are not valid expression syntax in the typed-IR
compiler. Rewrite them using colon selectors and an expression-first reducer
call as above. Invalid alias expressions fail with the alias's context instead
of becoming partially expanded state names.

Initial-state parameter names and expanded transition labels retain a separate
compatibility string-substitution path. When that path binds a wildcard in a
mixed selector, it preserves all explicitly bound entries; it must not collapse
the whole selector into a cell name containing only the wildcard axis. This
compatibility behavior does not make equals selectors valid expression syntax.
