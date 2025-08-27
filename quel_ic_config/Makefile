py_target = src tests testlibs scripts

.PHONY: validate-json format-cpp format-cpp-check format-py format-py-check lint typecheck test-without-devices
validate-json:
	@find src/quel_ic_config/settings -name "*.json" -print -exec jq "." {} \; > /dev/null
	@find src/quel_ic_config_cli/settings -name "*.json" -print -exec jq "." {} \; > /dev/null

format-cpp:
	@clang-format -i adi_ad9082_v170/ad9082_wrapper.cpp

format-cpp-check:
	@clang-format -i adi_ad9082_v170/ad9082_wrapper.cpp -Werror --dry-run

format-py:
	@#uv run ruff format $(py_target)
	@uv run black $(py_target)
	@uv run ruff check --fix --select I $(py_target)

format-py-check:
	@#uv run ruff format --check $(py_target)
	@uv run black --check $(py_target)
	@uv run ruff check --select I $(py_target)

lint:
	@uv run ruff check $(py_target)

fix-lint-errors:
	@uv run ruff check --fix $(py_target)

typecheck:
	@uv run mypy src/quel_ic_config src/quel_ic_config_utils src/quel_ic_config_cli tests scripts

test-without-devices:
	@uv run pytest tests/without_devices

test-multi-environment:
	@uvx tox
