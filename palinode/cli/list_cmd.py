import click
from palinode.cli._api import api_client
from palinode.cli._format import console, print_result, get_default_format, OutputFormat
from palinode.core.parity import CATEGORIES
from palinode.core.where import WhereError, parse_where


def _validate_where(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> tuple[str, ...]:
    try:
        parse_where(value)
    except WhereError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc
    return value


@click.command()
@click.option("--category", type=click.Choice(CATEGORIES))
@click.option("--core", "core_only", is_flag=True, help="Only show core memory files")
@click.option(
    "--where",
    multiple=True,
    callback=_validate_where,
    help="Only files whose frontmatter has key=value (e.g. status=open). Repeat to require all.",
)
@click.option("--include-inbox", is_flag=True, help="Also list inbox/, where ActionItems are saved")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]))
def list_cmd(category, core_only, where, include_inbox, fmt):
    """List memory files."""
    try:
        # Browse, not injection: a superseded or archived core memory stays
        # visible here (marked with why it no longer acts), while the
        # injection consumers of the same endpoint get it withheld.
        data = api_client.list_files(
            category=category,
            core_only=core_only,
            include_retired_core=True,
            where=list(where) or None,
            include_inbox=include_inbox or None,
        )

        output_fmt = OutputFormat(fmt) if fmt else get_default_format()

        if output_fmt == OutputFormat.JSON:
            print_result(data, fmt=output_fmt)
        else:
            if not data:
                console.print("No files found.")
                return

            # Group by category for text output
            from collections import defaultdict
            grouped = defaultdict(list)
            core_count = 0
            for item in data:
                grouped[item["category"]].append(item)
                if item["core"]:
                    core_count += 1

            for cat, items in grouped.items():
                console.print(f"[bold blue]{cat}/[/bold blue]")
                for item in items:
                    name = item["file"].split("/")[-1]
                    summary = item["summary"]
                    # Both tags are escaped (``\[``) — an unescaped ``[core]``
                    # is rich markup, and rich renders markup it doesn't know
                    # as nothing at all, so the tag never reached the terminal.
                    if item.get("core_retired_reason"):
                        core_tag = (
                            f" [yellow]\\[retired: {item['core_retired_reason']}][/yellow]"
                        )
                    else:
                        core_tag = " [bold green]\\[core][/bold green]" if item["core"] else ""
                    if summary:
                        console.print(f"  {name:<16} — {summary}{core_tag}")
                    else:
                        console.print(f"  {name:<16}{core_tag}")
                console.print("")

            console.print(f"[bold]{len(data)} files ({core_count} core)[/bold]")

    except Exception as e:
        console.print(f"[red]Error listing files: {str(e)}[/red]")
        raise click.Abort()
