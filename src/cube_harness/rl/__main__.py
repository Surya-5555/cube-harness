from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from cube_harness.rl.event_publisher import EventPublisherConfig
from cube_harness.rl.rollout import RolloutConfig
from cube_harness.rl.service import configure_terminal_logging, serve


def main(
    service_config: Annotated[Path, typer.Option(help="Path to a RolloutConfig JSON file.")],
    host: Annotated[str, typer.Option(help="Bind host for the rollout service.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port for the rollout service.")] = 8765,
    persist_events_dir: Annotated[Path | None, typer.Option(help="Directory to spill rollout events as JSONL.")] = None,
    log_level: Annotated[str, typer.Option(help="Python logging level for cube-harness service logs.")] = "INFO",
) -> None:
    """Run the cube-harness rollout service."""
    configure_terminal_logging(log_level, force=True)
    config = RolloutConfig.model_validate(json.loads(service_config.read_text()))
    app = serve(
        event_publisher_config=EventPublisherConfig(persist_events_dir=persist_events_dir),
        config=config,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def cli() -> None:
    # Console-script entry point (`ch-rollout`): setuptools calls this with no
    # arguments, so the Typer wrapper must own argv parsing.
    typer.run(main)


if __name__ == "__main__":
    cli()
