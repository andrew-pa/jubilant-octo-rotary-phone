from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence

from openai.types.chat import ChatCompletionToolParam

from .prompt_store import (
    PromptBlock,
    PromptStoreError,
    SystemPromptStore,
)


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


def _blocks_payload(blocks: Sequence[PromptBlock]) -> str:
    payload = {
        "blocks": [
            {
                "id": block.identifier,
                "text": block.text,
            }
            for block in blocks
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class PromptReadTool:
    name: str = "read_system_prompt_blocks"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Return the ordered list of system prompt blocks You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        blocks = context.prompt_store.get_blocks()
        return ToolResult(
            content=_blocks_payload(blocks),
            updated_system_prompt=None,
        )


class PromptReplaceTool:
    name: str = "replace_system_prompt_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Replace the text of an existing system prompt block You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Identifier of the block to replace.",
                        },
                        "text": {
                            "type": "string",
                            "description": "New text for the block.",
                        },
                    },
                    "required": ["id", "text"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        block_id = arguments.get("id")
        text = arguments.get("text")
        if not isinstance(block_id, str) or not block_id.strip():
            return ToolResult(content="Error: 'id' must be a non-empty string.")
        if not isinstance(text, str):
            return ToolResult(content="Error: 'text' must be a string.")
        normalized_id = block_id.strip()
        try:
            blocks = context.prompt_store.replace_block(
                identifier=normalized_id, text=text
            )
        except PromptStoreError as exc:
            return ToolResult(content=f"Error: {exc}")
        formatted = context.prompt_store.render_blocks(blocks)
        return ToolResult(
            content=f"Block '{normalized_id}' replaced successfully.",
            updated_system_prompt=formatted,
        )


class PromptAppendTool:
    name: str = "append_system_prompt_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Append a new system prompt block to the end of the list You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Identifier for the new block.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text content for the new block.",
                        },
                    },
                    "required": ["id", "text"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        block_id = arguments.get("id")
        text = arguments.get("text")
        if not isinstance(block_id, str) or not block_id.strip():
            return ToolResult(content="Error: 'id' must be a non-empty string.")
        if not isinstance(text, str):
            return ToolResult(content="Error: 'text' must be a string.")
        normalized_id = block_id.strip()
        try:
            blocks = context.prompt_store.append_block(
                identifier=normalized_id, text=text
            )
        except PromptStoreError as exc:
            return ToolResult(content=f"Error: {exc}")
        formatted = context.prompt_store.render_blocks(blocks)
        return ToolResult(
            content=f"Block '{normalized_id}' appended successfully.",
            updated_system_prompt=formatted,
        )


class PromptDeleteTool:
    name: str = "delete_system_prompt_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Delete a system prompt block by identifier You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Identifier of the block to delete.",
                        }
                    },
                    "required": ["id"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        block_id = arguments.get("id")
        if not isinstance(block_id, str) or not block_id.strip():
            return ToolResult(content="Error: 'id' must be a non-empty string.")
        normalized_id = block_id.strip()
        try:
            blocks = context.prompt_store.delete_block(identifier=normalized_id)
        except PromptStoreError as exc:
            return ToolResult(content=f"Error: {exc}")
        formatted = context.prompt_store.render_blocks(blocks)
        return ToolResult(
            content=f"Block '{normalized_id}' deleted successfully.",
            updated_system_prompt=formatted,
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
