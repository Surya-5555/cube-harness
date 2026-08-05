"""Vendored, trimmed BFCL evaluation logic (Apache-2.0).

Copied and adapted from the Berkeley Function Calling Leaderboard
(`bfcl_eval`, https://github.com/ShishirPatil/gorilla, Apache-2.0) so this
cube can score function calls at runtime **without** depending on the full
`bfcl-eval` package — whose import graph pulls in every model-provider SDK
(openai, anthropic, cohere, mistral, google-genai), vllm, torch and faiss.

Only the Python-language AST checker and the BFCL→OpenAI schema converter are
vendored. The Java/JavaScript type-converter paths (which need the
`tree_sitter` family) and the per-model name-normalisation table are dropped;
dotted function names are normalised uniformly (``.`` → ``_``), matching what
BFCL's own OpenAI-style handler does.

Upstream provenance is recorded per-module. When bumping the pinned BFCL data
version in ``scripts/create_task_metadata.py``, re-diff these files against the
matching `bfcl_eval` release.
"""
