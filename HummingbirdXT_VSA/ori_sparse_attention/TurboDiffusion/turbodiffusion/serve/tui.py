# download from https://github.com/thu-ml/TurboDiffusion.git

"""
TurboDiffusion TUI Server with a text-based interface for video generation.
Supports both T2V (text-to-video) and I2V (image-to-video) modes.
"""

import argparse
import os

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .arg_utils import parse_args, validate_args
from .utils import RUNTIME_PARAMS, format_config, set_runtime_param
from .pipeline import load_models, generate_t2v, generate_i2v

console = Console()

# Slash commands
COMMANDS = {
    "/help": "Show available commands",
    "/show": "Show current configuration",
    "/set": "Set a runtime parameter: /set <param> <value>",
    "/reset": "Reset runtime parameters to defaults",
    "/quit": "Exit the server",
}

# Style for prompt_toolkit
PROMPT_STYLE = Style.from_dict({
    "prompt": "#00aa00 bold",
    "command": "#ffaa00",
})


def print_header(args: argparse.Namespace):
    """Print fancy server header."""
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    header = Text()
    header.append("TurboDiffusion TUI Server\n", style="bold blue")
    header.append("Mode: ")
    if args.mode == "t2v":
        header.append("T2V", style="cyan")
        header.append(" (text-to-video)\n")
    else:
        header.append("I2V", style="magenta")
        header.append(" (image-to-video)\n")
    header.append("Model: ")
    header.append(args.model, style="green")
    header.append(" | Resolution: ")
    header.append(args.resolution, style="yellow")
    header.append(f" ({w}x{h}) | Steps: ")
    header.append(str(args.num_steps), style="yellow")

    console.print(Panel(header, border_style="blue"))
    console.print("Start typing prompts for generation. [dim]Type [bold]/help[/bold] for commands. Use [bold]\\\\[/bold] for newline in prompts.[/dim]\n")


def print_help():
    """Print help for slash commands."""
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="yellow")
    table.add_column("Description")

    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)

    console.print(table)

    # Runtime params
    console.print("\n[bold cyan]Runtime Parameters[/bold cyan] (adjustable with /set):")
    for param, spec in RUNTIME_PARAMS.items():
        if "choices" in spec:
            console.print(f"  [yellow]{param}[/yellow]: {spec['choices']}")
        else:
            console.print(f"  [yellow]{param}[/yellow]: {spec['type'].__name__} (min: {spec.get('min', 'none')})")


def print_config(args: argparse.Namespace, defaults: dict):
    """Print current configuration."""
    console.print(format_config(args, defaults), markup=True)


def get_prompt_input(history: InMemoryHistory) -> str:
    """Get prompt from user with slash command completion and line continuation."""
    completer = WordCompleter(list(COMMANDS.keys()), ignore_case=True)

    try:
        lines = []
        is_continuation = False

        while True:
            prompt_str = "... " if is_continuation else "> "
            line = prompt(
                [("class:prompt", prompt_str)],
                style=PROMPT_STYLE,
                completer=completer if not is_continuation else None,
                history=history if not is_continuation else None,
            )

            if line.endswith("\\"):
                # Line continuation: remove trailing backslash and continue
                lines.append(line[:-1])
                is_continuation = True
            else:
                lines.append(line)
                break

        text = "\n".join(lines)
        return text.strip()
    except (EOFError, KeyboardInterrupt):
        return None


def get_path_input(prompt_text: str, default: str = None, must_exist: bool = False) -> str:
    """Get file path from user."""
    default_hint = f" [{default}]" if default else ""
    try:
        text = prompt(
            [("class:prompt", f"{prompt_text}{default_hint}: ")],
            style=PROMPT_STYLE,
        )
        text = text.strip()

        if not text and default:
            return default

        if must_exist and text and not os.path.isfile(text):
            console.print(f"[red]Error: File not found: {text}[/red]")
            return None

        return text if text else None
    except (EOFError, KeyboardInterrupt):
        return None


def handle_command(cmd: str, args: argparse.Namespace, defaults: dict) -> bool:
    """Handle slash command. Returns False if should quit."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/quit":
        return False
    elif command == "/help":
        print_help()
    elif command == "/show":
        print_config(args, defaults)
    elif command == "/set":
        if len(parts) != 3:
            console.print("[red]Usage: /set <param> <value>[/red]")
        else:
            success, msg = set_runtime_param(args, parts[1], parts[2])
            if success:
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(f"[red]Error: {msg}[/red]")
    elif command == "/reset":
        for param, default in defaults.items():
            setattr(args, param, default)
        console.print("[green]Runtime parameters reset to defaults.[/green]")
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return True


def run_tui(models: dict, args: argparse.Namespace):
    """Main TUI loop."""
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    last_output_path = "output/generated_video.mp4"
    last_image_path = None

    prompt_history = InMemoryHistory()

    print_header(args)

    while True:
        # Get prompt
        user_input = get_prompt_input(prompt_history)

        if user_input is None:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if not handle_command(user_input, args, defaults):
                console.print("[dim]Goodbye![/dim]")
                break
            continue

        prompt_text = user_input

        # For I2V mode, get image path
        image_path = None
        if args.mode == "i2v":
            image_path = get_path_input("image", last_image_path, must_exist=True)
            if image_path is None:
                console.print("[yellow]Cancelled.[/yellow]")
                continue
            last_image_path = image_path

        # Get output path
        output_path = get_path_input("output", last_output_path)
        if output_path is None:
            console.print("[yellow]Cancelled.[/yellow]")
            continue

        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        # Generate
        console.print()
        try:
            with console.status("[bold green]Generating video...", spinner="dots"):
                if args.mode == "t2v":
                    result_path = generate_t2v(models, args, prompt_text, output_path)
                else:
                    result_path = generate_i2v(models, args, prompt_text, image_path, output_path)

            console.print(f"[bold green]Done:[/bold green] {result_path}")
            last_output_path = result_path
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            import traceback
            traceback.print_exc()

        console.print()


def main(passed_args: argparse.Namespace = None):
    """Main entry point for TUI server."""
    args = passed_args if passed_args is not None else parse_args()

    validate_args(args)

    console.print("[dim]Loading models...[/dim]")
    models = load_models(args)

    try:
        run_tui(models, args)
    except KeyboardInterrupt:
        console.print("\n\n[dim]Interrupted. Goodbye![/dim]")


if __name__ == "__main__":
    main()
