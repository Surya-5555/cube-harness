"""Python-only AST checker, vendored from ``bfcl_eval`` (Apache-2.0).

Faithful port of ``bfcl_eval.eval_checker.ast_eval.ast_checker`` restricted to
``Language.PYTHON``. The Java/JavaScript branches (and their tree-sitter type
converters) are removed, and per-model name normalisation is replaced by the
uniform ``.`` → ``_`` rule (see ``schema_convert.normalize_function_name``).

Entry point: :func:`ast_checker`. Inputs are already-decoded structures (the
cube hands the agent's tool calls in directly), so no text-decoding layer is
vendored.

- ``func_description``: list of BFCL function schemas for the task.
- ``model_output``: list of ``{function_name: {param: value}}`` — one per call.
- ``possible_answer``: BFCL ground truth, list of ``{function_name: {param: [acceptable, ...]}}``.
- ``test_category``: BFCL category string (``simple``/``multiple``/``parallel``/…).

Returns ``{"valid": bool, "error": [...], "error_type": str}``.
"""

from __future__ import annotations

import re
from typing import Any

from bfcl_cube._vendor.schema_convert import normalize_function_name
from bfcl_cube._vendor.type_mappings import (
    PYTHON_NESTED_TYPE_CHECK_LIST,
    PYTHON_TYPE_MAPPING,
)


def ast_checker(
    func_description: list[dict],
    model_output: list[dict],
    possible_answer: list[dict],
    test_category: str,
) -> dict[str, Any]:
    """Dispatch to the simple / multiple / parallel checker by category."""
    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer)
    elif "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer)
    else:
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }
        return simple_function_checker(func_description[0], model_output[0], possible_answer[0])


# ── Helpers ────────────────────────────────────────────────────────────────
def find_description(func_descriptions: Any, name: str) -> Any:
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list) -> type | None:
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def type_checker(
    param: str,
    value: Any,
    possible_answer: list,
    expected_type_description: str,
    expected_type_converted: type,
    nested_type_converted: type | None,
) -> dict[str, Any]:
    # Nested checking is one level deep only (sufficient for the dataset).
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}

    is_variable = False
    # If the ground-truth value type differs from the schema type, the answer
    # is a "variable" placeholder (stored as a string) — accept on type alone.
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type != expected_type_converted:
        is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True  # each parameter must match at least one possible answer
            if isinstance(possible_answer_item, list):
                for value_item in value:
                    checker_result = type_checker(
                        param,
                        value_item,
                        possible_answer_item,
                        str(nested_type_converted),
                        nested_type_converted,
                        None,
                    )
                    if not checker_result["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}

        result["valid"] = False
        result["error"] = [
            f"Nested type checking failed for parameter {param!r}. Expected outer type "
            f"{expected_type_description} with inner type {nested_type_converted!s}. "
            f"Parameter value: {value!r}."
        ]
        result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and type(value) == possible_answer_type:
        result["is_variable"] = True
        return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {param!r}. Expected type {expected_type_description}, "
        f"got {type(value).__name__}. Parameter value: {value!r}."
    )
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str) -> str:
    """Strip whitespace/punctuation and lowercase, so e.g. "April 1, 2024" and
    "April 1 2024" compare equal. Single quotes become double quotes."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", input_string).lower().replace("'", '"')


def string_checker(param: str, model_output: str, possible_answer: list) -> dict[str, Any]:
    standardize_model_output = standardize_string(model_output)
    standardize_possible_answer = [standardize_string(a) for a in possible_answer if isinstance(a, str)]
    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list) -> dict[str, Any]:
    standardize_model_output = list(model_output)
    for i in range(len(standardize_model_output)):
        if isinstance(standardize_model_output[i], str):
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer: list[list] = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            item = possible_answer[i][j]
            standardize_possible_answer[i].append(standardize_string(item) if isinstance(item, str) else item)

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [f"Invalid value for parameter {param!r}: {model_output!r}. Expected one of {possible_answer}."],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list) -> dict[str, Any]:
    # Simple (non-nested) dictionaries only — sufficient for the dataset.
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = standardize_string(value) if isinstance(value, str) else value
            standardize_possible_answer = [
                standardize_string(a) if isinstance(a, str) else a for a in possible_answer[key]
            ]
            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {key!r}: {value!r}. Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}
    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list) -> dict[str, Any]:
    # List of dicts; order must match one of the possible-answer orderings.
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer_index in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[answer_index]):
            result = {
                "valid": False,
                "error": ["Wrong number of dictionaries in the list."],
                "error_type": "value_error:list_dict_count",
            }
            continue
        for dict_index in range(len(model_output)):
            result = dict_checker(param, model_output[dict_index], [possible_answers[answer_index][dict_index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
) -> dict[str, Any]:
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}
    func_name = normalize_function_name(func_name)

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(f"Function name {func_name!r} not found in model output.")
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        nested_type_converted = None
        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = param_details[param]["items"]["type"]
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        # tuple values become lists after JSON round-tripping the ground truth.
        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)
        # Allow Python int → float promotion.
        if expected_type_description == "float" and type(value) == int:
            value = float(value)

        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        if not is_variable:
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {param!r}: {value!r}. Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(f"Optional parameter {param!r} not provided and not marked as optional.")
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
) -> dict[str, Any]:
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices: list[int] = []
    # Walk possible answers; greedily match each to an unused model output.
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors: list = []
        result = {"valid": False, "error": [], "error_type": "unmatched"}
        for index in range(len(model_output)):
            if index in matched_indices:
                continue
            result = simple_function_checker(func_description, model_output[index], possible_answers[i])
            if result["valid"]:
                matched_indices.append(index)
                break
            all_errors.append(
                {
                    f"Model Result Index {index}": {
                        "sub_error": result["error"],
                        "sub_error_type": result["error_type"],
                        "model_output_item": model_output[index],
                        "possible_answer_item": possible_answers[i],
                    }
                }
            )

        if not result["valid"]:
            considered_indices = [j for j in range(len(model_output)) if j not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model "
                f"output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def multiple_function_checker(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
) -> dict[str, Any]:
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(func_description, model_output[0], possible_answers[0])
