# Documentation

Build the public package documentation from the repository root:

```bash
LC_ALL=C LANG=C uv run sphinx-build -M html docs docs/_build -W
```

The tutorial describes the synthetic retrieval, the API page lists the shipped
interfaces, and the theory page records assumptions and research references.
