from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from core.config import get_settings, Settings


class ThoughtSignatureChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that preserves provider-specific `extra_content`
    (such as Google Gemini 3.x `thought_signature`) on tool calls across turns.
    """

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(warnings=False)
        )
        choices = response_dict.get("choices") or []
        for choice, gen in zip(choices, result.generations):
            raw_tool_calls = (choice.get("message") or {}).get("tool_calls") or []
            extra_map = {
                tc["id"]: tc["extra_content"]
                for tc in raw_tool_calls
                if isinstance(tc, dict) and tc.get("id") and tc.get("extra_content")
            }
            if extra_map and isinstance(gen.message, AIMessage):
                gen.message.additional_kwargs["tool_call_extra_content"] = extra_map
        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()

        extra_map: dict[str, Any] = {}
        for msg in messages:
            if isinstance(msg, AIMessage):
                tc_extra = msg.additional_kwargs.get("tool_call_extra_content")
                if isinstance(tc_extra, dict):
                    extra_map.update(tc_extra)

        if extra_map:
            for msg_dict in payload.get("messages", []):
                if msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                    for tc in msg_dict["tool_calls"]:
                        tc_id = tc.get("id")
                        if tc_id in extra_map and "extra_content" not in tc:
                            tc["extra_content"] = extra_map[tc_id]

        return payload


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """
    Build and configure a dual-compatible ChatOpenAI model instance ($0 stack).
    Seamlessly operates against local Ollama or Google AI Studio via environment settings.
    """
    cfg = settings or get_settings()
    return ThoughtSignatureChatOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        temperature=0.0,
        timeout=120.0,
        max_retries=1,
    )
