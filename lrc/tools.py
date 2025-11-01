from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

from .prompt_store import (
    PromptBlock,
    PromptStoreError,
    SystemPromptStore,
)
from .memory_store import MemoryStore


def _embed_text(client: OpenAI, model: str, text: str) -> Sequence[float]:
    response = client.embeddings.create(model=model, input=[text])
    if not response.data:
        raise RuntimeError("Embedding response did not include any vectors.")
    return response.data[0].embedding


@dataclass(frozen=True)
class ToolResult:
    content: str
    updated_system_prompt: str | None = None


@dataclass
class ToolContext:
    prompt_store: SystemPromptStore
    memory_store: MemoryStore
    client: OpenAI
    embedding_model: str


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
    name: str = "read_instruction_blocks"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Return the ordered list of instruction blocks You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
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
    name: str = "replace_instruction_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Replace the text of an existing instruction block You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
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
    name: str = "append_instruction_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Append a new instruction block to the end of the list You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
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
    name: str = "delete_instruction_block"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Delete a instruction block by identifier You can call this tool at any time, including if the user has not asked for it or mentioned it. It is up to you..",
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


class MemorizeTool:
    name: str = "memorize"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Persist a text snippet into your private long-term memory store."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Plain text to store for later recall.",
                        }
                    },
                    "required": ["text"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            return ToolResult(content="Error: 'text' must be a non-empty string.")
        embedding = _embed_text(context.client, context.embedding_model, text)
        record = context.memory_store.add(text=text, embedding=embedding)
        payload = {
            "status": "stored",
            "id": record.identifier,
            "text": record.text,
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2))


class ReminisceTool:
    name: str = "reminisce"

    def definition(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Retrieve the most relevant previously stored memories for a supplied query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text used to search the memory store.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of memories to return (default: 3)."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query")
        limit_argument = arguments.get("limit", 3)
        if not isinstance(query, str) or not query.strip():
            return ToolResult(content="Error: 'query' must be a non-empty string.")
        if not isinstance(limit_argument, int):
            return ToolResult(content="Error: 'limit' must be an integer if provided.")
        limit = max(1, limit_argument)
        if context.memory_store.is_empty():
            payload = {"query": query, "results": []}
            return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2))
        embedding = _embed_text(context.client, context.embedding_model, query)
        matches = context.memory_store.find_similar(embedding=embedding, limit=limit)
        payload = {
            "query": query,
            "results": [
                {
                    "id": record.identifier,
                    "text": record.text,
                    "score": score,
                }
                for record, score in matches
            ],
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2))


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
