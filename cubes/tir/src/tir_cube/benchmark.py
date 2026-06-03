from typing import ClassVar, Generator, cast
import logging
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from cube.resource import InfraConfig

from tir_cube.task import TIRTaskConfig, TIRTaskMetadata
from tir_cube.test_sandbox import test_sandbox
from datasets import load_dataset

logger = logging.getLogger(__name__)


def process_aime_and_amc(dataset, dataset_name):
    for item in dataset:
        task = item["problem"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": dataset_name,
            "task": task,
            "answer": answer,
        }


def _load_aime_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    if year == 2025:
        aime_dataset = load_dataset(
            "MathArena/aime_2025",
            split="train",
            trust_remote_code=False,
        )
    else:
        aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train", trust_remote_code=False)
        aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    tasks_data = [s for s in process_aime_and_amc(aime_dataset, dataset_name) if s is not None]

    if upsample_factor > 0:
        tasks_data *= upsample_factor

    return tasks_data


def process_open_reasoner(dataset, dataset_name):
    for item in dataset:
        # Note: Open Reasoner tasks sometimes have preamble, e.g.
        # - Example 31 (2004 College Entrance Examination Hunan Paper)
        # - 8.
        # - 4. (7 points)
        # We are currently ignoring the preamble
        task = item["0"]["value"]
        answer = "\\boxed{" + item["1"]["ground_truth"]["value"] + "}"
        yield {"dataset": dataset_name, "task": task, "answer": answer}


def load_task_metadata() -> dict[str, TIRTaskMetadata]:
    """Load task metadata from the Open Reasoner Zero dataset."""

    # dataset_names = ["open_reasoner_zero_57k", "open_reasoner_zero_extended_72k", "aime_2025"]
    dataset_names = ["open_reasoner_zero_57k"]
    dataset_urls = {
        "open_reasoner_zero_57k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
        "open_reasoner_zero_extended_72k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json",
    }

    metadata: dict[str, TIRTaskMetadata] = {}

    for dataset_name in dataset_names:
        tasks_data = []
        if "open_reasoner" in dataset_name:
            dataset = load_dataset(
                "json",
                data_files=dataset_urls.get(dataset_name),
                split="train",
                trust_remote_code=False,
            )
            tasks_data = [s for s in process_open_reasoner(dataset, dataset_name) if s is not None]

        if "aime" in dataset_name:
            year = int(dataset_name.split("_")[1])
            if dataset_name.endswith("_original"):
                tasks_data = _load_aime_dataset(year, upsample_factor=0)
            else:
                tasks_data = _load_aime_dataset(year, upsample_factor=16)

        if len(tasks_data) == 0:
            logger.warning(f"No tasks loaded for dataset {dataset_name}")
            continue

        for i, t in enumerate(tasks_data):
            instance_id = f"{dataset_name}_{i}"
            metadata[instance_id] = TIRTaskMetadata(
                id=instance_id,
                abstract_description="Solve math by tool-using with Python and submit final LaTeX answer via MathAnswer",
                recommended_max_steps=3,
                split="train",
                dataset=t["dataset"],
                question=t["task"],
                expected=t["answer"],
                rewards={
                    "correct_answer_finished": 1.0,
                    "correct_answer_not_finished": 0,
                    "wrong_answer_finished": 0,
                    "wrong_answer_not_finished": 0,
                    "no_answer_finished": 0,
                    "no_answer_not_finished": 0,
                    "unparsable_finished": 0,
                    "unparsable_not_finished": 0,
                },
            )

        logger.info(f"Loading {dataset_name}: {len(tasks_data)} tasks")

    logger.info(f"Loading total of {len(metadata)} tasks")
    return metadata


class TIRBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class TIRBenchmarkConfig(BenchmarkConfig[TIRTaskMetadata]):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="tir-use",
        version="0.1.0",
        description="Multi-Turn Tool-Integrated Reasoning tasks requiring deterministic Python tool use before final LaTeX answer submission",
        num_tasks=128,
        tags=["example", "arithmetic", "tool-use", "python"],
    )

    task_config_class: ClassVar[type[TaskConfig]] = TIRTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = TIRBenchmark

    task_metadata: ClassVar[dict[str, TaskMetadata]] = load_task_metadata()

    def make(self, infra: InfraConfig | None = None) -> TIRBenchmark:
        # make sure sandbox is warm and reachable

        tool_config = self.tool_config
        assert tool_config is not None, "tool_config must be set"
        assert tool_config.sandbox_endpoint is not None, "sandbox_endpoint must be set in tool_config"
        assert test_sandbox(tool_config.sandbox_endpoint), "Sandbox is not reachable"

        return cast(TIRBenchmark, super().make(infra=infra))

    def get_task_configs(self) -> Generator[TIRTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield TIRTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
            )
