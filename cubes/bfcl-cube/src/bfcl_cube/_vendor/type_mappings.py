"""Type maps vendored from ``bfcl_eval.constants.type_mappings`` (Apache-2.0).

Trimmed to the Python-language subset this cube needs:
- ``GORILLA_TO_OPENAPI`` — BFCL type vocabulary → JSON-Schema/OpenAPI types,
  used when exposing per-task function schemas to the agent.
- ``PYTHON_TYPE_MAPPING`` / ``PYTHON_NESTED_TYPE_CHECK_LIST`` — used by the
  vendored AST checker (``ast_checker.py``).

Java/JS conversion tables are intentionally omitted (those categories are out
of scope for this cube — they need the tree-sitter type converters).
"""

# BFCL type string -> OpenAPI/JSON-Schema type string.
GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

# BFCL type string -> concrete Python type, for AST value/type checking.
PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# Types whose element values are recursively checked (one level deep).
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]
