"""``where`` and ``include_inbox`` on every surface: parity registry, CLI and MCP.

The REST behaviour is in ``test_where_filter.py``. This file pins the other
surfaces to it:

* ``palinode.core.parity`` lists ``where`` (array) on ``search`` and ``list`` and
  ``include_inbox`` (boolean) on ``list``;
* the CLI takes a repeatable ``--where`` on ``list`` and ``search`` and
  ``--include-inbox`` on ``list``; a malformed entry is a click usage error
  (exit code 2) raised before any request is made;
* the MCP tools take ``where`` (array of strings) and, on ``palinode_list``,
  ``include_inbox``; a malformed entry is an ``Error:`` reply with no request.

On the wire: ``GET /list`` carries one ``where`` query parameter per entry plus
``include_inbox=true``; ``POST /search`` carries ``where`` as a JSON list.

Tests marked *red* fail on v0.22.0 for the reason in their docstring; the rest
pass on v0.22.0 and must keep passing. No test reaches the network: requests are
captured by ``httpx.MockTransport``, or served in-process by the real app.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from palinode import mcp
from palinode.api import server
from palinode.cli import main
from palinode.cli._api import api_client
from palinode.core import embedder, parity, store
from palinode.core.config import config
from palinode.indexer import reconcile


# ── helpers ─────────────────────────────────────────────────────────────────

class Recorder:
    """Captures every request and answers each with an empty JSON list."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=[])

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        return self.handle(request)

    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


@pytest.fixture()
def cli_recorder(monkeypatch) -> Recorder:
    recorder = Recorder()
    client = httpx.Client(
        transport=httpx.MockTransport(recorder.handle), base_url="http://testserver"
    )
    monkeypatch.setattr(api_client, "client", client)
    yield recorder
    client.close()


@pytest.fixture()
def mcp_recorder(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(mcp, "_http_transport", httpx.MockTransport(recorder.handle_async))
    monkeypatch.setattr(mcp, "_http_client", None)
    monkeypatch.setattr(mcp, "_http_client_loop", None)
    return recorder


def _call_tool(name: str, arguments: dict) -> str:
    return asyncio.run(mcp._dispatch_tool(name, arguments))[0].text


def _invoke(*args: str):
    return CliRunner().invoke(main, list(args))


# ── parity registry ─────────────────────────────────────────────────────────

def _param(operation: str, name: str):
    return {p.name: p for p in parity.by_name(operation).canonical_params}.get(name)


def test_search_registers_where_as_an_array_of_strings():
    """Red: no ``where`` parameter is registered on ``search``."""
    param = _param("search", "where")

    assert param is not None
    assert param.type == "array"
    assert not param.required


def test_list_registers_where_as_an_array_of_strings():
    """Red: no ``where`` parameter is registered on ``list``."""
    param = _param("list", "where")

    assert param is not None
    assert param.type == "array"
    assert not param.required


def test_list_registers_include_inbox_as_a_boolean():
    """Red: no ``include_inbox`` parameter is registered on ``list``."""
    param = _param("list", "include_inbox")

    assert param is not None
    assert param.type == "boolean"
    assert not param.required


def test_search_does_not_register_include_inbox():
    """Search already reaches the inbox; the flag is the listing's."""
    assert _param("search", "include_inbox") is None


# ── MCP tool schemas ────────────────────────────────────────────────────────

def _schema(tool_name: str) -> dict:
    tools = {tool.name: tool.input_schema for tool in mcp._all_tools()}
    return tools[tool_name]["properties"]


@pytest.mark.parametrize("tool_name", ["palinode_search", "palinode_list"])
def test_mcp_tool_declares_where_as_an_array_of_strings(tool_name):
    """Red: the tool schema has no ``where`` property."""
    prop = _schema(tool_name)["where"]

    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_mcp_list_declares_include_inbox_as_a_boolean():
    """Red: the tool schema has no ``include_inbox`` property."""
    assert _schema("palinode_list")["include_inbox"]["type"] == "boolean"


# ── CLI: what goes on the wire ──────────────────────────────────────────────

def test_cli_list_without_flags_sends_neither_parameter(cli_recorder):
    result = _invoke("list", "--format", "json")

    assert result.exit_code == 0, result.output
    params = cli_recorder.requests[-1].url.params
    assert cli_recorder.requests[-1].url.path == "/list"
    assert "where" not in params
    assert "include_inbox" not in params


def test_cli_list_sends_each_where_entry_and_the_inbox_flag(cli_recorder):
    """Red: ``--where`` and ``--include-inbox`` are unknown options (exit code 2)."""
    result = _invoke(
        "list", "--where", "status=open", "--where", "entities=project/my-app",
        "--include-inbox", "--format", "json",
    )

    assert result.exit_code == 0, result.output
    request = cli_recorder.requests[-1]
    assert request.method == "GET"
    assert request.url.path == "/list"
    assert request.url.params.get_list("where") == ["status=open", "entities=project/my-app"]
    assert request.url.params["include_inbox"] == "true"


def test_cli_search_without_where_sends_no_where_field(cli_recorder):
    result = _invoke("search", "release checklist", "--no-context", "--format", "json")

    assert result.exit_code == 0, result.output
    assert "where" not in cli_recorder.body()


def test_cli_search_sends_where_as_a_json_list(cli_recorder):
    """Red: ``--where`` is an unknown option (exit code 2)."""
    result = _invoke(
        "search", "release checklist", "--no-context", "--format", "json",
        "--where", "status=open", "--where", "entities=project/my-app",
    )

    assert result.exit_code == 0, result.output
    request = cli_recorder.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/search"
    assert cli_recorder.body()["where"] == ["status=open", "entities=project/my-app"]


@pytest.mark.parametrize("command", [["list"], ["search", "release checklist", "--no-context"]])
@pytest.mark.parametrize("entry", ["no-equals", "=value", "bad key=x", "1abc=x"])
def test_cli_malformed_where_is_a_usage_error_before_any_request(cli_recorder, command, entry):
    """Red: the output never names the entry. On v0.22.0 click answers with
    ``No such option: --where``, which is also exit code 2."""
    result = _invoke(*command, "--where", entry)

    assert result.exit_code == 2, result.output
    assert entry in result.output
    assert cli_recorder.requests == []


# ── CLI against the real app ────────────────────────────────────────────────

def _cli_http_client(test_client: TestClient) -> httpx.Client:
    """An ``httpx.Client`` whose transport dispatches to the in-process app (the
    CLI only ever talks to ``httpx``; see ``test_alias_curation.py``)."""

    def handler(request: httpx.Request) -> httpx.Response:
        reply = test_client.request(
            request.method,
            request.url.raw_path.decode("ascii"),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=reply.status_code,
            headers=reply.headers.multi_items(),
            content=reply.content,
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")


@pytest.fixture()
def real_app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.search, "retrieval_mode", "lexical")
    monkeypatch.setattr(config.git, "auto_commit", False)
    no_model = Mock(side_effect=AssertionError("a model was contacted"))
    monkeypatch.setattr(embedder, "embed", no_model)
    monkeypatch.setattr(embedder, "embed_many", no_model)
    monkeypatch.setattr(reconcile, "get_ollama_client", no_model)
    store.init_db()
    server._rate_counters.clear()
    with TestClient(server.app, raise_server_exceptions=False) as test_client:
        monkeypatch.setattr(api_client, "client", _cli_http_client(test_client))
        for slug, status in (("open-item", "open"), ("resolved-item", "resolved")):
            saved = test_client.post("/save", json={
                "content": f"harborlight {slug}", "type": "ActionItem", "slug": slug,
                "metadata": {"status": status},
            })
            assert saved.status_code == 200, saved.text
        yield test_client
    server._rate_counters.clear()


def test_cli_list_finds_the_open_actionitem_end_to_end(real_app):
    """Red: ``--where`` and ``--include-inbox`` are unknown options."""
    result = _invoke(
        "list", "--where", "status=open", "--include-inbox", "--format", "json"
    )

    assert result.exit_code == 0, result.output
    assert [row["file"] for row in json.loads(result.output)] == ["inbox/open-item.md"]


def test_cli_list_without_include_inbox_shows_no_actionitem_end_to_end(real_app):
    result = _invoke("list", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_cli_search_finds_the_open_actionitem_end_to_end(real_app):
    """Red: ``--where`` is an unknown option."""
    result = _invoke(
        "search", "harborlight", "--no-context", "--threshold", "0",
        "--format", "json", "--where", "status=open",
    )

    assert result.exit_code == 0, result.output
    assert [row["rel_path"] for row in json.loads(result.output)] == ["inbox/open-item.md"]


# ── MCP: what goes on the wire ──────────────────────────────────────────────

def test_mcp_list_without_arguments_sends_neither_parameter(mcp_recorder):
    _call_tool("palinode_list", {})

    params = mcp_recorder.requests[-1].url.params
    assert mcp_recorder.requests[-1].url.path == "/list"
    assert "where" not in params
    assert "include_inbox" not in params


def test_mcp_list_sends_each_where_entry_and_the_inbox_flag(mcp_recorder):
    """Red: both arguments are ignored."""
    _call_tool("palinode_list", {
        "where": ["status=open", "entities=project/my-app"], "include_inbox": True,
    })

    params = mcp_recorder.requests[-1].url.params
    assert params.get_list("where") == ["status=open", "entities=project/my-app"]
    assert params["include_inbox"] == "true"


def test_mcp_list_does_not_send_include_inbox_when_false(mcp_recorder):
    _call_tool("palinode_list", {"include_inbox": False})

    assert "include_inbox" not in mcp_recorder.requests[-1].url.params


def test_mcp_search_without_where_sends_no_where_field(mcp_recorder):
    _call_tool("palinode_search", {"query": "release checklist"})

    assert "where" not in mcp_recorder.body()


def test_mcp_search_sends_where_as_a_json_list(mcp_recorder):
    """Red: the argument is ignored."""
    _call_tool("palinode_search", {
        "query": "release checklist", "where": ["status=open", "entities=project/my-app"],
    })

    assert mcp_recorder.requests[-1].url.path == "/search"
    assert mcp_recorder.body()["where"] == ["status=open", "entities=project/my-app"]


@pytest.mark.parametrize("tool_name, arguments", [
    ("palinode_list", {}),
    ("palinode_search", {"query": "release checklist"}),
])
@pytest.mark.parametrize("entry", ["no-equals", "=value", "bad key=x", "1abc=x"])
def test_mcp_malformed_where_is_an_error_reply_before_any_request(
    mcp_recorder, tool_name, arguments, entry
):
    """Red: the entry is ignored and the call goes to the API."""
    text = _call_tool(tool_name, {**arguments, "where": [entry]})

    assert text.startswith("Error:")
    assert entry in text
    assert mcp_recorder.requests == []
