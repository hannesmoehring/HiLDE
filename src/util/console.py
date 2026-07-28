"""Shared rich console + small logging helpers for the calc layer.

Keeps the analysis pipeline's progress output consistent (and pretty) without
scattering `Console()` instances or ad-hoc `print()`s across modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from src.types import Config

console = Console()

# Config knobs worth surfacing in the build banner (label -> config key).
_BANNER_KEYS: list[tuple[str, str]] = [
    ("DR method", "method"),
    ("Normalize", "normalize"),
    ("Hierarchical layers", "hierarchical_layers"),
    ("Cluster method", "cluster_method"),
    ("Min cluster size", "hclust_min_cluster_size"),
    ("Min samples", "hclust_min_samples"),
    ("hclust UMAP dims", "hclust_umap_n_components"),
]


def build_banner(dataset: str, feature_cols: list[str], config: Config) -> None:
    """Panel announcing a build and the config it runs with."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    table.add_row("Dataset", str(dataset))
    table.add_row("Features", f"{len(feature_cols)} selected")
    for label, key in _BANNER_KEYS:
        if key in config:
            table.add_row(label, str(config[key]))  # type: ignore[literal-required]
    console.print()
    console.print(Panel(table, title="[bold]Starting build with selected config[/bold]", border_style="cyan", expand=False))


def phase(message: str) -> None:
    """A top-level pipeline step, e.g. 'Computing analysis tree'."""
    console.print(f"[bold cyan]▶[/bold cyan] [bold]{message}[/bold]")


def substep(message: str) -> None:
    """A nested, lower-signal step (dim reduction / clustering during recursion)."""
    console.print(f"  [dim]· {message}[/dim]")


def success(message: str, **stats: Any) -> None:
    """A completed step, optionally with `key=value` stats appended."""
    tail = "  ".join(f"[cyan]{k}[/cyan]=[bold]{v}[/bold]" for k, v in stats.items())
    console.print(f"[bold green]✓[/bold green] {message}" + (f"   {tail}" if tail else ""))
