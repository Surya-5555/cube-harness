"""Tests for the F2a baseline-subtract behaviour in ``SWEBenchVerifiedTask.evaluate``.

Asserts the invariant: **if pass_to_pass already fails on the unpatched tree,
a post-patch p2p failure with the same shape must NOT score the agent 0.0**
(it is a pre-existing environmental issue, not an agent regression).

The other half of the invariant — strict p2p enforcement when baseline passes —
is also covered.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cube.tools.terminal import TerminalToolConfig

from swebench_verified_cube.task import (
    SWEBenchVerifiedExecutionInfo,
    SWEBenchVerifiedTask,
    SWEBenchVerifiedTaskMetadata,
)


def _make_task() -> SWEBenchVerifiedTask:
    """Construct a Task without launching a real container.

    Patches ``model_post_init`` so the base ``cube.task.Task`` does not try to
    provision an infra container or build a tool. The test only exercises
    ``evaluate``, which we drive by patching ``_run_tests`` and ``_apply_patch``.
    """
    with patch.object(SWEBenchVerifiedTask, "model_post_init", lambda *a, **kw: None):
        task = SWEBenchVerifiedTask(
            metadata=SWEBenchVerifiedTaskMetadata(
                id="sphinx-doc__sphinx-8475",
                description="linkcheck redirect handling",
                repo="sphinx-doc/sphinx",
                difficulty="15 min - 1 hour",
                version="4.0",
                base_commit="deadbeef",
            ),
            execution_info=SWEBenchVerifiedExecutionInfo(
                problem_statement="x",
                patch="diff",
                test_patch="diff",
                fail_to_pass=["tests/test_linkcheck.py::test_TooManyRedirects"],
                pass_to_pass=["tests/test_linkcheck.py::test_defaults"],
                eval_timeout=60,
            ),
            tool_config=TerminalToolConfig(working_dir="/testbed"),
            oracle_mode=False,
            append_submission_instructions=True,
        )
    return task


def test_p2p_pre_existing_failure_does_not_penalise_agent() -> None:
    """When p2p fails on baseline AND post-patch (same env-side root cause),
    evaluate() must return reward=1.0 if f2p passes — the agent is not at fault."""
    task = _make_task()

    # Sequence of _run_tests calls: (baseline p2p, f2p, post-patch p2p)
    run_outputs = [
        (False, "ConnectionResetError: google.com"),  # baseline p2p fails
        (True, "1 passed"),  # f2p passes
        (False, "ConnectionResetError: google.com"),  # post-patch p2p still fails (same shape)
    ]
    with patch.object(task, "_apply_patch", return_value=""), patch.object(task, "_run_tests", side_effect=run_outputs):
        reward, info = task.evaluate()

    assert reward == 1.0
    assert info["resolved"] is True
    assert info["fail_to_pass_passed"] is True
    assert info["pass_to_pass_passed"] is True
    assert info["pass_to_pass_baseline_passed"] is False
    assert "pre-existing p2p baseline failures" in info["pass_to_pass_output"]


def test_p2p_baseline_passes_then_post_patch_regresses_scores_zero() -> None:
    """Strict-enforcement half of the invariant: if baseline p2p was clean,
    a post-patch p2p failure IS the agent's regression — score 0.0."""
    task = _make_task()

    run_outputs = [
        (True, "1 passed"),  # baseline p2p passes
        (True, "1 passed"),  # f2p passes
        (False, "AssertionError: regression"),  # post-patch p2p fails
    ]
    with patch.object(task, "_apply_patch", return_value=""), patch.object(task, "_run_tests", side_effect=run_outputs):
        reward, info = task.evaluate()

    assert reward == 0.0
    assert info["resolved"] is False
    assert info["fail_to_pass_passed"] is True
    assert info["pass_to_pass_passed"] is False
    assert info["pass_to_pass_baseline_passed"] is True


def test_p2p_empty_skips_baseline_run() -> None:
    """Tasks with no pass_to_pass list should not run a baseline check."""
    task = _make_task()
    task.execution_info.pass_to_pass = []  # type: ignore[union-attr]

    # Only f2p call expected; baseline is skipped because pass_to_pass is empty.
    run_outputs = [(True, "1 passed")]
    mock_run = MagicMock(side_effect=run_outputs)
    with patch.object(task, "_apply_patch", return_value=""), patch.object(task, "_run_tests", mock_run):
        reward, info = task.evaluate()

    assert mock_run.call_count == 1
    assert reward == 1.0
    assert info["pass_to_pass_baseline_passed"] is True


def test_f2p_failure_scores_zero_regardless_of_p2p_baseline() -> None:
    """If f2p fails, resolution is 0 regardless of p2p baseline state."""
    task = _make_task()

    run_outputs = [
        (True, "1 passed"),  # baseline p2p passes
        (False, "AssertionError: f2p"),  # f2p fails
        (True, "1 passed"),  # post-patch p2p clean
    ]
    with patch.object(task, "_apply_patch", return_value=""), patch.object(task, "_run_tests", side_effect=run_outputs):
        reward, info = task.evaluate()

    assert reward == 0.0
    assert info["fail_to_pass_passed"] is False


def test_network_dependent_p2p_are_stripped_before_baseline() -> None:
    """Tests in ``_NETWORK_DEPENDENT_P2P`` are removed from pass_to_pass BEFORE
    the baseline check fires — they reach neither the baseline run nor the
    post-patch run. This is the exact, deterministic counterpart to PR#423's
    coarse baseline-subtract relaxation.
    """
    from swebench_verified_cube.task import _NETWORK_DEPENDENT_P2P

    task = _make_task()
    # Mix two skip-listed tests with one real test that must survive.
    skip_listed = sorted(_NETWORK_DEPENDENT_P2P)[:2]
    real_test = "tests/test_build_linkcheck.py::test_invalid_ssl"
    task.execution_info.pass_to_pass = [*skip_listed, real_test]  # type: ignore[union-attr]

    # The 2 skip-listed entries should never appear in any _run_tests call:
    # baseline gets only [real_test], post-patch gets only [real_test], f2p
    # unchanged.
    seen_p2p_args: list[list[str]] = []

    def _stub_run_tests(repo, directives, **_kw):  # noqa: ARG001
        if directives and directives != task._exec.fail_to_pass:
            seen_p2p_args.append(list(directives))
        return (True, "1 passed")

    with (
        patch.object(task, "_apply_patch", return_value=""),
        patch.object(task, "_run_tests", side_effect=_stub_run_tests),
    ):
        reward, info = task.evaluate()

    # Both p2p invocations (baseline + post-patch) saw the stripped list.
    assert len(seen_p2p_args) == 2
    for call_directives in seen_p2p_args:
        assert call_directives == [real_test]
        for skipped in skip_listed:
            assert skipped not in call_directives
    assert reward == 1.0


def test_skip_list_uses_pytest_id_format() -> None:
    """Sanity-pin the test IDs against the exact format stored in the task
    execution cache — guards against accidental ``tests.test_X.test_y``
    (django-style) or trailing-slash regressions that would silently make
    the skip list a no-op.
    """
    from swebench_verified_cube.task import _NETWORK_DEPENDENT_P2P

    for tid in _NETWORK_DEPENDENT_P2P:
        assert tid.startswith("tests/")
        assert "::" in tid, f"{tid!r} must use pytest 'path::name' form"
