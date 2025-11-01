from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .chat_session import ChatSession
from .config import ClientConfig
from .prompt_store import SystemPromptStore
from .memory_store import MemoryStore
from .tools import (
    PromptAppendTool,
    PromptDeleteTool,
    PromptReadTool,
    PromptReplaceTool,
    MemorizeTool,
    ReminisceTool,
    ToolRegistry,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI chatbot")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name to use for chat completions (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Model name to use for generating embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional response token cap",
    )
    parser.add_argument(
        "--api-base",
        dest="api_base",
        default=None,
        help="Override the API base URL (set this to Ollama's endpoint to use Ollama)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key; defaults to environment variables understood by the OpenAI SDK",
    )
    parser.add_argument(
        "--organization",
        dest="organization",
        default=None,
        help="Optional organization ID for the API request",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        default="system_prompt.json",
        help="Path to the system prompt file to load and update",
    )
    parser.add_argument(
        "--memory-store",
        dest="memory_store",
        default=None,
        help=(
            "Path to the persistent memory store file "
            "(default: alongside the system prompt with suffix _memories.npz)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser


def configure_logging(level: str, console: Console) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def create_client(config: ClientConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        organization=config.organization,
    )


def create_prompt_session(history_path: Path) -> PromptSession[str]:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_path))
    style = Style.from_dict({
        "bottom-toolbar": "gray",
    })
    session = PromptSession[str](
        style=style,
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        complete_while_typing=False,
    )
    return session


def run_chat_loop(
    session: ChatSession,
    prompt_session: PromptSession[str],
    console: Console,
    model_name: str,
) -> None:
    console.print(
        Panel.fit(
            "Type your message and press Enter. Use Ctrl+D or :exit to quit.",
            title="Chat Started",
            border_style="bright_black",
        )
    )
    while True:
        try:
            user_input = prompt_session.prompt(
                HTML("<ansicyan><b>You</b></ansicyan> <b>›</b> "),
                rprompt=HTML(f"<ansimagenta>{model_name}</ansimagenta>"),
                bottom_toolbar=lambda: HTML(
                    "<bottom-toolbar>Ctrl+D to exit · :exit to quit</bottom-toolbar>"
                ),
            )
        except EOFError:
            console.print("\nGoodbye!", style="bold yellow")
            return
        except KeyboardInterrupt:
            console.print("\nKeyboard interrupt. Type :exit to quit.", style="bold red")
            continue
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.startswith(":"):
            command = stripped[1:].strip().lower()
            if command in {"exit", "quit", "q"}:
                console.print("Exiting chat.", style="bold yellow")
                return
            if command == "reset":
                session.reset()
                console.print(
                    "Chat history cleared. Current system prompt retained.",
                    style="bold yellow",
                )
                continue
            if command in {"usage", "tokens"}:
                summary = session.usage_summary()
                last_prompt = summary["last_prompt_tokens"] or 0
                last_completion = summary["last_completion_tokens"] or 0
                last_total = summary["last_total_tokens"] or 0
                lines = [
                    f"Cumulative prompt tokens: {summary['total_prompt_tokens']}",
                    f"Cumulative completion tokens: {summary['total_completion_tokens']}",
                    f"Cumulative total tokens: {summary['total_tokens']}",
                    f"Current context tokens (latest prompt): {last_prompt}",
                    f"Current completion tokens (latest reply): {last_completion}",
                    f"Current total tokens (latest call): {last_total}",
                ]
                console.print(
                    Panel.fit(
                        "\n".join(lines),
                        title="Token Usage",
                        border_style="green",
                    )
                )
                continue
            console.print(
                f"Unknown command '{stripped}'. Available commands: :exit, :reset, :usage",
                style="bold red",
            )
            continue
        console.rule("You", style="cyan")
        console.print(user_input)
        try:
            assistant_message = session.send_user_message(user_input)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("lrc").exception("Chat completion failed")
            console.print(f"Error: {exc}", style="bold red")
            continue
        console.rule("Assistant", style="magenta")
        assistant_text = assistant_message.content or ""
        if assistant_text:
            console.print(assistant_text)
        else:
            console.print("(No assistant text returned)", style="bright_black")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    console = Console()
    configure_logging(args.log_level, console)
    prompt_path = Path(args.system_prompt).expanduser().resolve()
    prompt_store = SystemPromptStore(prompt_path)
    prompt_store.ensure_exists()
    if args.memory_store is None:
        memory_store_path = prompt_path.with_name(f"{prompt_path.stem}_memories.npz")
    else:
        memory_store_path = Path(args.memory_store).expanduser().resolve()
    memory_store = MemoryStore(memory_store_path)
    config = ClientConfig(
        model=args.model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.api_base,
        api_key=args.api_key,
        organization=args.organization,
        system_prompt_path=prompt_path,
        memory_store_path=memory_store_path,
    )
    client = create_client(config)
    tool_registry = ToolRegistry(
        [
            PromptReadTool(),
            PromptReplaceTool(),
            PromptAppendTool(),
            PromptDeleteTool(),
            MemorizeTool(),
            ReminisceTool(),
        ]
    )
    chat_session = ChatSession(
        client=client,
        config=config,
        prompt_store=prompt_store,
        memory_store=memory_store,
        tool_registry=tool_registry,
        logger=logging.getLogger("lrc"),
    )
    history_path = Path.home() / ".cache" / "lrc" / "history"
    prompt_session = create_prompt_session(history_path)
    console.print(
        f"Loaded system prompt from {prompt_path}", style="bright_black"
    )
    console.print(
        Panel.fit(
            chat_session.current_system_prompt(),
            title="System Prompt",
            border_style="blue",
        )
    )
    run_chat_loop(chat_session, prompt_session, console, config.model)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation path
    raise SystemExit(main())
