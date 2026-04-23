"""ForgeMesh CLI entry point."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from forgemesh import __version__
from forgemesh.auth import ensure_api_key
from forgemesh.config import Config
from forgemesh.models import ModelCatalog
from forgemesh.server import build_state, create_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ForgeMesh — self-hosted, OpenAI-compatible LLM inference.",
)
models_app = typer.Typer(help="Manage the local GGUF model cache.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print the ForgeMesh version."""
    console.print(f"forgemesh {__version__}")


@app.command()
def serve(
    model: str = typer.Option(None, "--model", "-m", help="Model name (in cache) or path to .gguf."),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to forgemesh.yaml."
    ),
    host: str | None = typer.Option(None, "--host", help="Bind address."),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port."),
    no_auth: bool = typer.Option(False, "--no-auth", help="Disable API-key auth (dev only)."),
) -> None:
    """Start the ForgeMesh API server."""
    cfg = Config.load(config_file).resolve_paths()

    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port
    if no_auth:
        cfg.auth.enabled = False
    if model is not None:
        cfg.model = model

    if cfg.model is None:
        console.print(
            "[red]error:[/red] no model specified. "
            "Pass --model or set `model:` in your config file."
        )
        raise typer.Exit(code=2)

    catalog = ModelCatalog(cfg.model_dir)
    try:
        model_path = catalog.resolve(cfg.model)
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    model_name = model_path.stem

    api_key = ""
    if cfg.auth.enabled:
        api_key, new = ensure_api_key(cfg.auth.api_key_file)
        if new:
            console.print(
                "[yellow]Generated new API key.[/yellow] "
                f"Stored at {cfg.auth.api_key_file}"
            )
            console.print(f"  [bold]FORGEMESH_API_KEY={api_key}[/bold]")
            console.print(
                "  This key is printed once. "
                "Keep it; re-read it from the file above if you lose it."
            )

    state = build_state(cfg, model_path=model_path, model_name=model_name, api_key=api_key)

    console.print(f"[green]Starting llama-server[/green] with model {model_path.name}")
    try:
        state.llama.start()
    except Exception as e:
        console.print(f"[red]error starting llama-server:[/red] {e}")
        raise typer.Exit(code=1) from e

    fastapi_app = create_app(state)

    def _shutdown(*_):
        console.print("\n[yellow]Shutting down...[/yellow]")
        state.llama.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    auth_note = "API-key auth: [green]enabled[/green]" if cfg.auth.enabled else "API-key auth: [red]disabled[/red]"
    console.print(f"Listening on http://{cfg.host}:{cfg.port}  ({auth_note})")
    console.print(f"Model: [cyan]{model_name}[/cyan]")

    try:
        uvicorn.run(fastapi_app, host=cfg.host, port=cfg.port, log_config=None)
    finally:
        state.llama.stop()


@models_app.command("list")
def models_list(
    config_file: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """List locally cached GGUF models."""
    cfg = Config.load(config_file).resolve_paths()
    catalog = ModelCatalog(cfg.model_dir)
    items = catalog.list()
    if not items:
        console.print(f"No models in {cfg.model_dir}.")
        console.print("Pull one with: [cyan]forgemesh models pull <hf-repo> <filename>[/cyan]")
        return
    table = Table(title=f"Models in {cfg.model_dir}")
    table.add_column("name", style="cyan")
    table.add_column("size", justify="right")
    table.add_column("path")
    for m in items:
        table.add_row(m.name, f"{m.size_gb:.2f} GB", str(m.path))
    console.print(table)


@models_app.command("pull")
def models_pull(
    repo_id: str = typer.Argument(..., help="HuggingFace repo id, e.g. unsloth/Qwen3-8B-GGUF"),
    filename: str = typer.Argument(..., help="File inside the repo, e.g. Qwen3-8B-Q4_K_M.gguf"),
    config_file: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Download a GGUF from HuggingFace into the local cache."""
    cfg = Config.load(config_file).resolve_paths()
    catalog = ModelCatalog(cfg.model_dir)
    dst = catalog.pull(repo_id, filename)
    console.print(f"[green]ok[/green] {dst}")


@models_app.command("rm")
def models_rm(
    name: str = typer.Argument(..., help="Model name (stem) or filename."),
    config_file: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Remove a cached model."""
    cfg = Config.load(config_file).resolve_paths()
    catalog = ModelCatalog(cfg.model_dir)
    try:
        path = catalog.remove(name)
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e
    console.print(f"[green]removed[/green] {path}")


if __name__ == "__main__":
    app()
