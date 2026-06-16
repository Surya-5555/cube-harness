.PHONY: install debug test gen


install:
	uv sync --all-extras
	uv pip install -e .

debug:
	uv run python -m bfcl_cube.debug

test:
	uv run cube test bfcl-cube

# Regenerate committed data from the BFCL v4 dataset (author-time).
gen:
	uv sync --group gen
	uv run scripts/create_task_metadata.py
