from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence

from openai.types.chat import ChatCompletionToolParam

from .diff_apply import DiffApplyError, apply_unified_diff
from .prompt_store import SystemPromptStore


@dataclass(frozen=True)
class ToolResult:
    content: str
    updated_system_prompt: str | None = None


@dataclass
class ToolContext:
    prompt_store: SystemPromptStore


class Tool(Protocol):
    name: str

    def definition(self) -> ChatCompletionToolParam:
        ...

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        ...


class PromptPatchTool:
    name: str = "apply_system_prompt_patch"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Apply a unified diff patch to the system prompt."
                    " The diff must be generated against the current prompt contents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "Unified diff to apply to the system prompt file.",
                        }
                    },
                    "required": ["diff"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        diff_value = arguments.get("diff")
        if not isinstance(diff_value, str):
            return ToolResult(
                content="Error: expected 'diff' argument to be a unified diff string.",
            )
        current_prompt = context.prompt_store.read()
        try:
            updated_prompt = apply_unified_diff(
                current_prompt, diff_value, filename=context.prompt_store.path.name
            )
        except DiffApplyError as exc:
            return ToolResult(content=f"Error applying diff: {exc}")
        context.prompt_store.write(updated_prompt)
        return ToolResult(
            content="System prompt updated successfully.",
            updated_system_prompt=updated_prompt,
        )


@dataclass
class ToolInvocation:
    name: str
    arguments: Dict[str, Any]
    call_id: str

    @staticmethod
    def from_openai_payload(payload: Any) -> "ToolInvocation":
        arguments_json = getattr(payload.function, "arguments", "{}")
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments were not valid JSON") from exc
        return ToolInvocation(
            name=getattr(payload.function, "name"),
            arguments=arguments,
            call_id=getattr(payload, "id"),
        )


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> Sequence[ChatCompletionToolParam]:
        return tuple(tool.definition() for tool in self._tools.values())

    def invoke(self, invocation: ToolInvocation, context: ToolContext) -> ToolResult:
        tool = self._tools.get(invocation.name)
        if tool is None:
            raise ValueError(f"Received call for unknown tool '{invocation.name}'")
        return tool.invoke(invocation.arguments, context)
