PYTHON := uv run

.PHONY: sync test lint format-check typecheck docs verify

sync:
	uv sync --dev

test:
	$(PYTHON) pytest -q

lint:
	$(PYTHON) ruff check .

format-check:
	$(PYTHON) black --check .

typecheck:
	$(PYTHON) mypy src

docs:
	LC_ALL=C LANG=C $(PYTHON) sphinx-build -M html docs docs/_build

verify: lint format-check typecheck test docs
