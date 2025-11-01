from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, cast

from openai import Omit, OpenAI, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from .config import ClientConfig
from .prompt_store import SystemPromptStore
from .tools import ToolContext, ToolInvocation, ToolRegistry, ToolResult

META_PROMPT = """
You MUST follow these instructions exactly.
You MAY modify these instructions using the provided tools at any time, for any reason.
""".strip()


def _empty_message_list() -> list["Message"]:
    return []

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    identifier: str
    function_name: str
    arguments_json: str

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": self.arguments_json,
            },
        }


@dataclass
class Message:
    role: MessageRole
    content: str | None
    tool_call_id: str | None = None
    tool_calls: Sequence[ToolCall] | None = None

    def to_openai(self) -> ChatCompletionMessageParam:
        payload: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        elif self.role != "assistant":
            raise ValueError("Non-assistant messages must include content")
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None and self.tool_calls:
            payload["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]
        return cast(ChatCompletionMessageParam, payload)


@dataclass
class ChatSession:
    client: OpenAI
    config: ClientConfig
    prompt_store: SystemPromptStore
    tool_registry: ToolRegistry
    logger: logging.Logger
    _messages: list[Message] = field(default_factory=_empty_message_list)
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0
    _total_tokens: int = 0
    _last_prompt_tokens: int | None = None
    _last_completion_tokens: int | None = None
    _last_total_tokens: int | None = None

    def __post_init__(self) -> None:
        system_prompt = self.prompt_store.read()
        self._messages.append(Message(role="system", content=''))
        self._update_system_prompt(system_prompt)

    def send_user_message(self, content: str) -> Message:
        user_message = Message(role="user", content=content)
        self._messages.append(user_message)
        return self._generate_assistant_reply()

    def reset(self) -> None:
        self._messages = [Message(role="system", content='')]
        self._update_system_prompt(self.prompt_store.read())
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._last_prompt_tokens = None
        self._last_completion_tokens = None
        self._last_total_tokens = None
        self.logger.info("Chat session reset; system prompt retained")

    def usage_summary(self) -> Dict[str, int | None]:
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
            "last_prompt_tokens": self._last_prompt_tokens,
            "last_completion_tokens": self._last_completion_tokens,
            "last_total_tokens": self._last_total_tokens,
        }

    def _generate_assistant_reply(self) -> Message:
        while True:
            response = self._create_completion()
            self._update_usage(response)
            choice = response.choices[0]
            message = choice.message
            assistant_message = self._record_assistant_message(message)
            tool_calls = assistant_message.tool_calls or ()
            if not tool_calls:
                return assistant_message
            self._handle_tool_calls(message.tool_calls or [])

    def _create_completion(self) -> ChatCompletion:
        messages_payload: List[ChatCompletionMessageParam] = [
            message.to_openai() for message in self._messages
        ]
        tools_definitions = list(self.tool_registry.definitions())
        tools_argument: Sequence[ChatCompletionToolParam] | Omit
        if tools_definitions:
            tools_argument = cast(Sequence[ChatCompletionToolParam], tools_definitions)
        else:
            tools_argument = omit
        max_tokens_argument: int | None | Omit
        if self.config.max_tokens is None:
            max_tokens_argument = omit
        else:
            max_tokens_argument = self.config.max_tokens
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=messages_payload,
            tools=tools_argument,
            temperature=self.config.temperature,
            max_tokens=max_tokens_argument,
        )

    def _record_assistant_message(self, message: ChatCompletionMessage) -> Message:
        tool_calls = self._convert_tool_calls(message.tool_calls or [])
        assistant_message = Message(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls if tool_calls else None,
        )
        self._messages.append(assistant_message)
        return assistant_message

    def _update_usage(self, response: ChatCompletion) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None:
            self._total_prompt_tokens += prompt_tokens
            self._last_prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self._total_completion_tokens += completion_tokens
            self._last_completion_tokens = completion_tokens
        if total_tokens is not None:
            self._total_tokens += total_tokens
            self._last_total_tokens = total_tokens

    def _convert_tool_calls(
        self, tool_calls: Sequence[Any]
    ) -> Sequence[ToolCall]:
        result: List[ToolCall] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                self.logger.warning(
                    "Skipping unsupported tool call type: %s",
                    getattr(tool_call, "type", "unknown"),
                )
                continue
            result.append(
                ToolCall(
                    identifier=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments_json=tool_call.function.arguments,
                )
            )
        return tuple(result)

    def _handle_tool_calls(self, tool_calls: Sequence[Any]) -> None:
        tool_context = ToolContext(prompt_store=self.prompt_store)
        for tool_call in tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                self.logger.warning(
                    "Cannot execute unsupported tool call type: %s",
                    getattr(tool_call, "type", "unknown"),
                )
                continue
            try:
                invocation = ToolInvocation.from_openai_payload(tool_call)
            except ValueError as exc:
                self.logger.error("Failed to parse tool invocation: %s", exc)
                continue
            self.logger.info(
                "Tool call %s (%s) with arguments %s",
                invocation.call_id,
                invocation.name,
                invocation.arguments,
            )
            try:
                result = self.tool_registry.invoke(invocation, tool_context)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Tool execution failed: %s", exc)
                error_result = ToolResult(
                    content=f"Tool execution failed: {exc}",
                    updated_system_prompt=None,
                )
                result = error_result
            tool_message = Message(
                role="tool",
                content=result.content,
                tool_call_id=invocation.call_id,
            )
            self._messages.append(tool_message)
            self.logger.info(
                "Tool result %s: %s",
                invocation.call_id,
                result.content,
            )
            if result.updated_system_prompt is not None:
                self._update_system_prompt(result.updated_system_prompt)

    def _update_system_prompt(self, new_prompt: str) -> None:
        system_message = self._messages[0]
        if system_message.role != "system":
            raise RuntimeError("First message in the conversation must be system role")
        system_message.content = f'{META_PROMPT}\n{new_prompt}'
        self.logger.info("System prompt updated during session")

    def current_system_prompt(self) -> str:
        return self._messages[0].content or ''
