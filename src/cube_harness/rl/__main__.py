from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from cube_harness.rl.rollout import RolloutConfig
from cube_harness.rl.service import configure_terminal_logging, serve
from cube_harness.rl.sink import EventSinkConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cube-harness rollout service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--persist-events-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", help="Python logging level for cube-harness service logs.")
    parser.add_argument(
        "--service-config",
        type=Path,
        required=True,
        help="Path to a RolloutConfig JSON file.",
    )
    args = parser.parse_args()
    configure_terminal_logging(args.log_level, force=True)
    service_config = RolloutConfig.model_validate(json.loads(args.service_config.read_text()))
    app = serve(
        sink_config=EventSinkConfig(persist_events_dir=args.persist_events_dir),
        config=service_config,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
