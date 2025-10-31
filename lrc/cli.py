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
from .tools import PromptPatchTool, ToolRegistry


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI chatbot")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name to use for chat completions (default: %(default)s)",
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
        default="system_prompt.txt",
        help="Path to the system prompt file to load and update",
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
        if stripped in {":exit", ":quit", ":q"}:
            console.print("Exiting chat.", style="bold yellow")
            return
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
    config = ClientConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.api_base,
        api_key=args.api_key,
        organization=args.organization,
        system_prompt_path=prompt_path,
    )
    client = create_client(config)
    tool_registry = ToolRegistry([PromptPatchTool()])
    chat_session = ChatSession(
        client=client,
        config=config,
        prompt_store=prompt_store,
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
            prompt_store.read().strip(),
            title="System Prompt",
            border_style="blue",
        )
    )
    run_chat_loop(chat_session, prompt_session, console, config.model)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation path
    raise SystemExit(main())
