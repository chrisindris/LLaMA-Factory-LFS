.PHONY: build commit license quality quality-changed style test

check_dirs := scripts src tests tests_v1

ruff_version := 0.15.5

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
BUILD := $(shell command -v uv >/dev/null 2>&1 && echo "uv build" || echo "python -m build")
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUFF := $(shell command -v uv >/dev/null 2>&1 && echo "uvx ruff@$(ruff_version)" || echo "ruff")

build:
	$(BUILD)

commit:
	$(TOOL) pre-commit install
	$(TOOL) pre-commit run --all-files

license:
	$(RUN) python3 tests/check_license.py $(check_dirs)

quality:
	$(RUFF) check $(check_dirs)
	$(RUFF) format --check $(check_dirs)

# Lint/format only Python files changed vs BASE (default: origin/main).
# Example: make quality-changed
#          make quality-changed BASE=HEAD~3
BASE ?= origin/main
quality-changed:
	@files=$$(git diff --name-only --diff-filter=ACMR $(BASE)...HEAD -- $(check_dirs) | grep -E '\.py$$' || true); \
	if [ -z "$$files" ]; then echo "No changed Python files under $(check_dirs)."; exit 0; fi; \
	echo "$$files"; \
	ruff check $$files; \
	ruff format --check $$files

style:
	$(RUFF) check $(check_dirs) --fix
	$(RUFF) format $(check_dirs)

test:
	WANDB_DISABLED=true $(RUN) pytest -vv --import-mode=importlib tests/ tests_v1/
