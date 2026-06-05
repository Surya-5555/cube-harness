from __future__ import annotations

from typing import Any, Callable

from pydantic import Field, SecretStr

import cube_harness.llm as llm_core
from cube_harness.llm import BaseLLM, BaseLLMConfig, LLMResponse, Prompt

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class RolloutTokenCounter:
    def __init__(self, tokenizer_name: str):
        if AutoTokenizer is None:
            raise ImportError("AutoTokenizer is required for RolloutTokenCounter. Please install transformers library.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

    def count_prompt_tokens(self, messages, tools=None) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_special_tokens=True,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(token_ids)


class RolloutLLMConfig(BaseLLMConfig):
    """Per-rollout LLM override accepted from rollout clients.

    This is intentionally narrower than cube_harness.llm.LLMConfig. It is the
    trainer-facing API for selecting an OpenAI/vLLM-compatible endpoint and a
    small set of generation/logprob controls needed for RL data capture.
    """

    model_name: str
    temperature: float = 1.0
    max_completion_tokens: int = 8192
    max_tokens: int | None = None
    api_base: str
    api_key: SecretStr = Field(exclude=True)
    tokenizer_name: str
    top_p: float | None = None
    top_k: int | None = None
    num_retries: int = 1
    extra_body: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)

    def make(self) -> "RolloutLLM":
        """Create RolloutLLM instance from config."""
        return RolloutLLM(config=self)

    def make_counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return RolloutTokenCounter(tokenizer_name=self.tokenizer_name).count_prompt_tokens


def _safe_finish_reason(choice: Any) -> str | None:
    value = getattr(choice, "finish_reason", None)
    return value if isinstance(value, str) else None


class RolloutLLM(BaseLLM):
    config: RolloutLLMConfig

    def __init__(self, config: RolloutLLMConfig):
        super().__init__(config)

    def __call__(self, prompt: Prompt) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "api_base": self.config.api_base,
            "api_key": self.config.api_key.get_secret_value(),
            "messages": prompt.messages,
            "temperature": self.config.temperature,
            "logprobs": 1,
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            "timeout": self.config.timeout,
        }
        if self.config.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k

        extra_body = dict(self.config.extra_body or {})
        extra_body["return_token_ids"] = True
        extra_body["return_tokens_as_token_ids"] = True
        kwargs["extra_body"] = extra_body

        response = llm_core._completion_with_retry(self.config.num_retries, **kwargs)
        usage = llm_core._extract_usage(response)
        prompt_token_ids = self._extract_prompt_token_ids(response)
        completion_logprobs = self._extract_completion_logprobs(response)
        completion_token_ids = self._extract_completion_token_ids(
            response=response,
            completion_logprobs=completion_logprobs,
        )
        logprobs = [entry["logprob"] for entry in completion_logprobs] if completion_logprobs else None

        if not prompt_token_ids:
            raise ValueError(
                "vLLM did not return prompt_token_ids. Check that "
                "extra_body={'return_token_ids': True} reaches vLLM and that LiteLLM "
                "preserves provider-specific response fields."
            )

        if not completion_token_ids:
            raise ValueError(
                "vLLM did not return completion token IDs. Check logprobs=1 and return_tokens_as_token_ids=True."
            )

        if logprobs is None:
            raise ValueError("vLLM did not return completion logprobs. Rollout RL requires old logprobs.")

        if len(completion_token_ids) != len(logprobs):
            raise ValueError(
                f"completion_token_ids/logprobs length mismatch: {len(completion_token_ids)} != {len(logprobs)}"
            )

        return LLMResponse(
            message=response.choices[0].message,
            usage=usage,
            logprobs=logprobs,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            finish_reason=_safe_finish_reason(response.choices[0]),
        )

    def _extract_completion_logprobs(self, response: Any) -> list[dict[str, int | float]]:
        """Extract vLLM/OpenAI-compatible completion token IDs and logprobs."""
        result: list[dict[str, int | float]] = []
        choices = getattr(response, "choices", None)
        if not choices:
            return result

        choice = choices[0]
        logprobs = self._get_extra(choice, "logprobs")
        content = self._get_extra(logprobs, "content")
        if not isinstance(content, list):
            return result

        for entry in content:
            token_id = self._get_extra(entry, "token_id")
            if token_id is None:
                token_id = self._parse_token_id(self._get_extra(entry, "token"))
            logprob = self._get_extra(entry, "logprob")
            if isinstance(token_id, int) and isinstance(logprob, (int, float)):
                result.append({"token_id": token_id, "logprob": float(logprob)})
        return result

    def _extract_prompt_token_ids(self, response: Any) -> list[int] | None:
        """Extract vLLM prompt token IDs preserved by LiteLLM."""
        for key in ("prompt_token_ids", "prompt_tokens"):
            ids = self._coerce_token_id_list(self._get_extra(response, key))
            if ids is not None:
                return ids

        choices = getattr(response, "choices", None)
        if choices:
            choice = choices[0]
            for key in ("prompt_token_ids", "prompt_tokens"):
                ids = self._coerce_token_id_list(self._get_extra(choice, key))
                if ids is not None:
                    return ids

        dumped = self._dump_response(response)
        for key in ("prompt_token_ids", "prompt_tokens"):
            ids = self._coerce_token_id_list(self._find_key_recursive(dumped, key))
            if ids is not None:
                return ids
        return None

    def _extract_completion_token_ids(
        self,
        response: Any,
        completion_logprobs: list[dict[str, int | float]],
    ) -> list[int] | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None

        choice = choices[0]
        for obj in (choice, self._get_extra(choice, "message")):
            if obj is None:
                continue
            for key in ("token_ids", "completion_token_ids", "output_token_ids"):
                ids = self._coerce_token_id_list(self._get_extra(obj, key))
                if ids is not None:
                    return ids

        dumped = self._dump_response(choice)
        for key in ("token_ids", "completion_token_ids", "output_token_ids"):
            ids = self._coerce_token_id_list(self._find_key_recursive(dumped, key))
            if ids is not None:
                return ids

        if completion_logprobs:
            return [int(entry["token_id"]) for entry in completion_logprobs]
        return None

    @staticmethod
    def _parse_token_id(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return None
        if value.startswith("token_id:"):
            value = value.split(":", 1)[1]
        try:
            return int(value)
        except ValueError:
            return None

    @classmethod
    def _coerce_token_id_list(cls, value: Any) -> list[int] | None:
        if not isinstance(value, list):
            return None

        parsed: list[int] = []
        for item in value:
            token_id = cls._parse_token_id(item)
            if token_id is None:
                return None
            parsed.append(token_id)
        return parsed

    @staticmethod
    def _get_extra(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        value = getattr(obj, key, None)
        if value is not None:
            return value
        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict) and key in model_extra:
            return model_extra[key]
        hidden_params = getattr(obj, "_hidden_params", None)
        if isinstance(hidden_params, dict) and key in hidden_params:
            return hidden_params[key]
        return None

    @staticmethod
    def _dump_response(obj: Any) -> Any:
        if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump()
            except Exception:
                pass
        dict_method = getattr(obj, "dict", None)
        if callable(dict_method):
            try:
                return dict_method()
            except Exception:
                pass
        return None

    @classmethod
    def _find_key_recursive(cls, obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for value in obj.values():
                found = cls._find_key_recursive(value, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = cls._find_key_recursive(value, key)
                if found is not None:
                    return found
        return None
