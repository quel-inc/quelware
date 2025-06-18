target = src tests testlibs scripts

.PHONY: format format-check lint typecheck test-without-device
format:
	@#uv run ruff format $(target)
	@uv run black $(target)
	@uv run ruff check --fix --select I $(target)

format-check:
	@#uv run ruff format $(target)
	@uv run black --check $(target)
	@uv run ruff check --select I $(target)

lint:
	@uv run ruff check .

typecheck:
	@uv run mypy --check-untyped-defs $(target)

test-without-device:
	@uv run pytest tests/without_device
