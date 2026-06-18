"""TimeWarp benchmark implementation for the CUBE framework.

TimeWarp's three web environments (wiki / news / webshop) are external Flask
servers, addressed via the ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` environment
variables. This cube runs in *manual* mode: the user starts the servers (see
``timewarp/scripts/environment/run_all_env.sh``) and ``_setup()`` only verifies
that the configured URLs are set and reachable — there is no Docker provisioning.
"""

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig

from timewarp_cube.task import TimeWarpTaskConfig, TimeWarpTaskMetadata

logger = logging.getLogger(__name__)

#: Environment variables that must point at reachable TimeWarp servers.
_REQUIRED_ENV_VARS = ("TW_WIKI", "TW_NEWS", "TW_WEBSHOP")

_START_HINT = (
    "Start the TimeWarp environments first, e.g.:\n"
    "  bash timewarp/scripts/environment/run_all_env.sh 1\n"
    "then export TW_WIKI / TW_NEWS / TW_WEBSHOP (and OPENAI_API_KEY for the llm_judge)."
)


class TimeWarpBenchmark(Benchmark["TimeWarpBenchmarkConfig"]):
    """Runtime pair — TimeWarp connects to externally-started Flask servers, so
    there is no infrastructure to provision; ``_setup()`` just verifies reachability.
    """

    def _setup(self) -> None:
        missing = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"Missing TimeWarp environment variable(s): {', '.join(missing)}.\n{_START_HINT}")

        for var in _REQUIRED_ENV_VARS:
            url = os.environ[var]
            try:
                urllib.request.urlopen(url, timeout=5)
            except urllib.error.HTTPError:
                pass  # Server answered (even a 4xx/5xx, e.g. login redirect or 404 at /) — it's up.
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise RuntimeError(f"Cannot reach TimeWarp server {var}={url}: {e}\n{_START_HINT}") from e
        logger.info("TimeWarp benchmark ready with %d tasks", self.config.num_tasks)

    def close(self) -> None:
        logger.info("TimeWarp benchmark closed.")


class TimeWarpBenchmarkConfig(BenchmarkConfig[TimeWarpTaskMetadata]):
    """CUBE BenchmarkConfig for TimeWarp — 231 temporal-UI web tasks.

    Manual environment mode: start the wiki/news/webshop servers yourself and set
    ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` (and ``OPENAI_API_KEY`` for the
    ``llm_judge`` evaluator). ``make()`` then verifies the servers are reachable.

    Filter by site in user-land via named subsets or glob:
        cfg.named_subset("wiki")
        cfg.subset_from_glob("sites", "*news*")

    task_metadata.json is a shipped package resource with lightweight public fields
    (sites, intent_template_id, eval_types). No heavy execution data exists — all
    task logic is available from the browsergym-timewarp library at runtime.

    To regenerate task_metadata.json (developer use only), run:
        scripts/generate_task_metadata.py
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="timewarp-cube",
        version="0.1.0",
        description=(
            "TimeWarp benchmark — 231 web tasks across wiki, news, and webshop, "
            "designed to test agent robustness to temporal changes in web UI. "
            "Use named_subset('wiki'/'news'/'webshop') to filter by site."
        ),
        num_tasks=231,
        tags=["browser", "web", "ui", "timewarp"],
        named_subsets={
            "wiki": ("sites", "*wiki*"),
            "news": ("sites", "*news*"),
            "webshop": ("sites", "*webshop*"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = TimeWarpTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = TimeWarpBenchmark

    def get_task_configs(self) -> Generator[TimeWarpTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield TimeWarpTaskConfig(metadata=tm, tool_config=self.tool_config)
