"""Cheap, infra-free static guards over every cube's `task.py`.

These catch the class of cube-level regression a cross-cutting contract sweep can
introduce without a real model or infra — the kind that otherwise only surfaces in a
live Auto-CUBE run. Two were caught the hard way during the streamable-task refactor:

  * a `self._tool -> self.tool` sweep rewrote miniwob's `tool` *property override* body to
    `return self.tool`, which self-recurses (RecursionError on first `action_set` access);
  * the new per-seat hooks (`obs_postprocess` / `_post_action` / `_filter_actions`) take a
    `role` arg the base passes positionally — an override missing it raises at runtime.

Pure AST, no construction, no infra, no LLM — runs in CI in milliseconds.
"""

import ast
import pathlib

_CUBES_DIR = pathlib.Path(__file__).resolve().parent.parent / "cubes"
CUBE_TASK_FILES = sorted(_CUBES_DIR.glob("*/src/*/task.py"))

# Per-seat hooks the base `Task` calls with a positional `role`; an override that drops it
# (and has no **kwargs) breaks at runtime when the base invokes it.
_ROLE_HOOKS = {"obs_postprocess", "_post_action", "_filter_actions"}


def _properties(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        ):
            yield node


def _body_without_docstring(node: ast.FunctionDef) -> list[ast.stmt]:
    return [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]


def test_cube_task_files_discovered() -> None:
    # Guard the guard: if the glob breaks, the other tests would vacuously pass.
    assert CUBE_TASK_FILES, f"no cube task.py files found under {_CUBES_DIR}"


def test_no_self_recursive_properties() -> None:
    """`@property def X(self): return self.X` self-recurses → RecursionError on access."""
    offenders: list[str] = []
    for f in CUBE_TASK_FILES:
        tree = ast.parse(f.read_text(), str(f))
        for node in _properties(tree):
            body = _body_without_docstring(node)
            if len(body) == 1 and isinstance(body[0], ast.Return):
                r = body[0].value
                if (
                    isinstance(r, ast.Attribute)
                    and isinstance(r.value, ast.Name)
                    and r.value.id == "self"
                    and r.attr == node.name
                ):
                    offenders.append(f"{f}:{node.lineno}  @property {node.name}(self): return self.{node.name}")
    assert not offenders, "Self-recursive property override(s) found:\n  " + "\n  ".join(offenders)


def test_per_seat_hook_overrides_accept_role() -> None:
    """A cube override of a per-seat hook must accept `role` (or **kwargs)."""
    offenders: list[str] = []
    for f in CUBE_TASK_FILES:
        tree = ast.parse(f.read_text(), str(f))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in _ROLE_HOOKS:
                continue
            arg_names = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
            if "role" not in arg_names and node.args.kwarg is None:
                params = ", ".join(a.arg for a in node.args.args)
                offenders.append(f"{f}:{node.lineno}  {node.name}({params}) — missing `role` (and no **kwargs)")
    assert not offenders, "Per-seat hook override(s) missing `role`:\n  " + "\n  ".join(offenders)
