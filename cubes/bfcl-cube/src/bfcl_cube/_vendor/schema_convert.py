"""BFCL function-schema → OpenAI/JSON-Schema converter (Apache-2.0).

Adapted from ``bfcl_eval.model_handler.utils._cast_to_openai_type`` /
``convert_to_tool`` (OpenAI-completions path only). BFCL function docs use a
loose type vocabulary (``dict``/``integer``/``tuple``/``any`` …); LLM tool APIs
expect JSON-Schema types (``object``/``integer``/``array`` …). This module does
that one conversion so a per-task function schema can be handed to the agent as
a ``cube.core.ActionSchema``.

Provider-specific branches (Anthropic/Gemini/Amazon/Cohere…) from the upstream
``convert_to_tool`` are dropped: the cube emits provider-neutral
``ActionSchema``s and the harness/LiteLLM adapts them per provider.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from bfcl_cube._vendor.type_mappings import GORILLA_TO_OPENAPI


def normalize_function_name(name: str) -> str:
    """Normalise a BFCL function name to a tool-call-safe identifier.

    OpenAI/Anthropic tool names must match ``^[a-zA-Z0-9_-]{1,64}$`` — no dots.
    BFCL ships dotted names (e.g. ``math.factorial``); upstream replaces ``.``
    with ``_`` for these providers. We do the same uniformly so the agent-facing
    name and the checker's normalised ground-truth name agree.
    """
    return re.sub(r"\.", "_", name)


def _cast_to_openai_type(properties: dict[str, Any]) -> dict[str, Any]:
    """Recursively rewrite BFCL parameter types to OpenAPI/JSON-Schema types.

    Faithful port of the upstream ``_cast_to_openai_type`` for the
    ``GORILLA_TO_OPENAPI`` mapping (supports list-of-any / list-of-list /
    list-of-dict / dict-of-any one level deep, matching the dataset).
    """
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] = properties[key].get("description", "") + " This is a float type value."
            properties[key]["type"] = GORILLA_TO_OPENAPI.get(var_type, "string")

        if properties[key]["type"] in ("array", "object"):
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(properties[key]["properties"])
            elif "items" in properties[key]:
                items = properties[key]["items"]
                items["type"] = GORILLA_TO_OPENAPI[items["type"]]
                if items["type"] == "array" and "items" in items:
                    items["items"]["type"] = GORILLA_TO_OPENAPI[items["items"]["type"]]
                elif items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_to_openai_type(items["properties"])
    return properties


def bfcl_parameters_to_openai(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI-style JSON-Schema ``parameters`` object for a BFCL function.

    Deep-copies the input (the cast mutates in place). Forces the top-level type
    to ``object`` and casts every property type via :func:`_cast_to_openai_type`.
    """
    params = copy.deepcopy(parameters)
    params["type"] = "object"
    params["properties"] = _cast_to_openai_type(params.get("properties", {}))
    return params
