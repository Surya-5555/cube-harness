.PHONY: test debug install

test:
	uv run cube test timewarp-cube

debug:
	uv run python -m timewarp_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	# task_metadata.json is committed, so there is nothing to generate here — regenerating it is
	# a deliberate developer step: `uv run scripts/generate_task_metadata.py --force`.
