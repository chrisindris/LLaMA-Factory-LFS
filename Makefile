.PHONY: build commit license quality quality-changed style test

check_dirs := scripts src tests tests_v1 setup.py

build:
	pip3 install build && python3 -m build

commit:
	pre-commit install
	pre-commit run --all-files

license:
	python3 tests/check_license.py $(check_dirs)

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

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
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest -vv tests/
