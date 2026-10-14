"""Tests for the bearer-token auth middleware (Tier A — last public-scan high).

The middleware is default-off so existing local-dev workflows keep working
without configuration. When ``PALINODE_API_TOKEN`` (or
``PALINODE_API_TOKEN_FILE``) is set, every request must carry an
``Authorization: Bearer <token>`` header — except the ``/health`` and
``/health/watcher`` probes, which stay open for uptime checks.

The startup gate is the loud-fail counterpart: with a non-loopback
``PALINODE_API_HOST`` (or ``PALINODE_API_BIND_INTENT=public``) and no token
configured, the API must refuse to start so an operator cannot accidentally
expose an unauthenticated service. ``PALINODE_API_ALLOW_UNAUTH=1`` is the
explicit opt-out for deliberately token-less, network-isolated hosts.
"""
from __future__ import annotations

import importlib
import inspect
import os

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GATE_ENV_VARS = (
    "PALINODE_API_TOKEN",
    "PALINODE_API_TOKEN_FILE",
    "PALINODE_API_HOST",
    "PALINODE_API_BIND_INTENT",
    "PALINODE_API_ALLOW_UNAUTH",
)


def _import_app_in_subprocess(env_overrides: dict[str, str]):
    """Mirror ``uvicorn palinode.api.server:app`` in a fresh interpreter.

    uvicorn imports the module to read the ``app`` attribute and never calls
    ``main()``, so the import itself is the real startup path under systemd.
    Returns the completed process; callers assert on returncode + output.
    """
    import subprocess
    import sys

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # The conftest's sandbox. Without it the fresh interpreter resolves
        # ``~/palinode`` and logs its startup warning into the real store.
        "PALINODE_DIR": os.environ["PALINODE_DIR"],
        **env_overrides,
    }
    for k in _GATE_ENV_VARS:
        if k not in env_overrides:
            env.pop(k, None)
    return subprocess.run(
        [sys.executable, "-c", "from palinode.api.server import app  # noqa: F401"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(autouse=True, scope="module")
def _leave_a_tokenless_server_behind():
    """Re-import the server without auth env once this module is done.

    ``_reload_server`` reloads the shared module in place, so the last test here
    leaves ``palinode.api.server.app`` wrapped in bearer auth with that test's
    token. Every later test on the same worker that does
    ``from palinode.api.server import app`` then gets 401s, which made modules
    such as ``test_correction_review.py`` fail only when collected after this
    one. Module-scoped, so it runs after each test's ``monkeypatch`` has
    already restored the environment.
    """
    yield
    with pytest.MonkeyPatch.context() as mp:
        for key in _GATE_ENV_VARS:
            mp.delenv(key, raising=False)
        import palinode.api.server as server_mod  # noqa: PLC0415

        importlib.reload(server_mod)


def _reload_server(monkeypatch: pytest.MonkeyPatch, **env: str | None):
    """Re-import ``palinode.api.server`` with patched env vars.

    The module reads ``PALINODE_API_TOKEN`` / ``PALINODE_API_TOKEN_FILE`` /
    ``PALINODE_API_HOST`` / ``PALINODE_API_BIND_INTENT`` /
    ``PALINODE_API_ALLOW_UNAUTH`` once at import time, so each test that
    cares about a different configuration needs a fresh module. Returns
    the freshly loaded module.
    """
    # Clear any previous run's values so monkeypatch can supply fresh ones.
    for key in _GATE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is None:
            continue
        monkeypatch.setenv(key, value)
    # Force a clean re-import. importlib.reload preserves the module
    # identity so other test modules that imported it earlier see the
    # latest config too.
    import palinode.api.server as server_mod  # noqa: PLC0415
    return importlib.reload(server_mod)


# ---------------------------------------------------------------------------
# Token loader
# ---------------------------------------------------------------------------


def test_no_token_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    """No env var, no file → ``_load_api_token`` returns None."""
    server = _reload_server(monkeypatch)
    assert server._api_token is None
    assert server._load_api_token() is None


def test_token_loaded_from_env(monkeypatch: pytest.MonkeyPatch):
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="secret-abc")
    assert server._api_token == "secret-abc"


def test_token_loaded_from_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """``PALINODE_API_TOKEN_FILE`` reads token from disk and strips trailing whitespace."""
    token_file = tmp_path / "tok"
    token_file.write_text("file-token-xyz\n", encoding="utf-8")
    server = _reload_server(
        monkeypatch, PALINODE_API_TOKEN_FILE=str(token_file)
    )
    assert server._api_token == "file-token-xyz"


def test_env_token_wins_over_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """When both are set, the env var takes precedence (cheaper, no I/O)."""
    token_file = tmp_path / "tok"
    token_file.write_text("from-file", encoding="utf-8")
    server = _reload_server(
        monkeypatch,
        PALINODE_API_TOKEN="from-env",
        PALINODE_API_TOKEN_FILE=str(token_file),
    )
    assert server._api_token == "from-env"


def test_unreadable_token_file_falls_through_to_none(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
):
    """Missing file path → returns None and logs an error (no crash)."""
    missing = tmp_path / "does-not-exist"
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN_FILE=str(missing))
    assert server._api_token is None


def test_whitespace_only_token_treated_as_unset(monkeypatch: pytest.MonkeyPatch):
    """A pure-whitespace env var must not enable auth (would 401 every request)."""
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="   \n\t  ")
    assert server._api_token is None


# ---------------------------------------------------------------------------
# Startup gate
# ---------------------------------------------------------------------------


def test_validate_auth_config_raises_when_public_bind_no_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """The gate fires at MODULE IMPORT, not just from main(), so it
    propagates under uvicorn-direct invocation (the canonical systemd
    ExecStart pattern). See server.py module-scope
    ``_validate_auth_config(_api_token)`` call.
    """
    # SystemExit must propagate out of the import itself, not require an
    # explicit follow-up call.
    monkeypatch.delenv("PALINODE_API_TOKEN", raising=False)
    monkeypatch.delenv("PALINODE_API_TOKEN_FILE", raising=False)
    monkeypatch.setenv("PALINODE_API_BIND_INTENT", "public")
    import palinode.api.server  # noqa: F401  (preload baseline)
    import importlib
    with pytest.raises(SystemExit) as excinfo:
        importlib.reload(palinode.api.server)
    msg = str(excinfo.value)
    assert "REFUSING TO START" in msg
    assert "PALINODE_API_TOKEN" in msg


def test_validate_auth_config_passes_when_public_bind_with_token(
    monkeypatch: pytest.MonkeyPatch,
):
    server = _reload_server(
        monkeypatch,
        PALINODE_API_BIND_INTENT="public",
        PALINODE_API_TOKEN="set-token",
    )
    # Should not raise.
    server._validate_auth_config(server._api_token)


def test_validate_auth_config_passes_when_loopback_no_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """Default deployment (loopback bind, no token) must keep working —
    that's the local-first promise."""
    server = _reload_server(monkeypatch)
    assert server._bind_intent_public is False
    assert server._api_token is None
    # Should not raise.
    server._validate_auth_config(server._api_token)


def test_validate_auth_config_fires_under_uvicorn_direct_import():
    """Regression test: systemd-style deployments invoke uvicorn directly
    with ``palinode.api.server:app``, bypassing ``main()``. If the gate
    is only inside ``main()``, the public bind would start unauthenticated
    despite ``PALINODE_API_BIND_INTENT=public``. This test simulates the
    uvicorn-direct path: a fresh Python subprocess that imports
    ``palinode.api.server:app`` (the FastAPI object), with the failing
    config in env. The import must raise SystemExit during the import
    itself.
    """
    # No token; no token-file.
    result = _import_app_in_subprocess({"PALINODE_API_BIND_INTENT": "public"})
    assert result.returncode != 0, (
        f"Import should fail under public-bind+no-token, but exit was 0.\n"
        f"stderr: {result.stderr}"
    )
    combined = (result.stderr or "") + (result.stdout or "")
    assert "REFUSING TO START" in combined, (
        f"Expected SystemExit message in output:\n{combined}"
    )
    assert "PALINODE_API_TOKEN" in combined


# ---------------------------------------------------------------------------
# Bind gate — keys on the resolved bind host, not on stated intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"])
def test_is_loopback_host_accepts_loopback(host: str):
    from palinode.core.auth import is_loopback_host

    assert is_loopback_host(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "100.64.0.9", "palinode-host", ""])
def test_is_loopback_host_rejects_network(host: str):
    from palinode.core.auth import is_loopback_host

    assert is_loopback_host(host) is False


def test_validate_bind_auth_refuses_non_loopback_no_token():
    from palinode.core.auth import validate_bind_auth

    with pytest.raises(SystemExit) as excinfo:
        validate_bind_auth("0.0.0.0", None, allow_unauth=False)
    msg = str(excinfo.value)
    assert "REFUSING TO START" in msg
    assert "PALINODE_API_HOST=0.0.0.0" in msg
    assert "PALINODE_API_TOKEN" in msg
    assert "PALINODE_API_ALLOW_UNAUTH=1" in msg


@pytest.mark.parametrize(
    ("host", "token", "allow_unauth"),
    [
        ("127.0.0.1", None, False),
        ("localhost", None, False),
        ("0.0.0.0", "tok", False),
        ("0.0.0.0", None, True),
        ("192.168.1.5", "tok", False),
    ],
)
def test_validate_bind_auth_passes(host: str, token: str | None, allow_unauth: bool):
    from palinode.core.auth import validate_bind_auth

    validate_bind_auth(host, token, allow_unauth=allow_unauth)


def test_bind_gate_refuses_import_when_host_non_loopback_no_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """PALINODE_API_HOST=0.0.0.0 alone — no token, no intent var — must
    refuse at import. This is the path the gate previously missed."""
    for key in _GATE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PALINODE_API_HOST", "0.0.0.0")
    import palinode.api.server  # noqa: F401  (preload baseline)

    with pytest.raises(SystemExit) as excinfo:
        importlib.reload(palinode.api.server)
    msg = str(excinfo.value)
    assert "REFUSING TO START" in msg
    assert "PALINODE_API_ALLOW_UNAUTH=1" in msg


def test_bind_gate_opt_out_starts_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """PALINODE_API_ALLOW_UNAUTH=1 lets a token-less non-loopback bind start,
    and the app still warns on every start."""
    import logging

    with caplog.at_level(logging.WARNING, logger="palinode.api"):
        server = _reload_server(
            monkeypatch, PALINODE_API_HOST="0.0.0.0", PALINODE_API_ALLOW_UNAUTH="1"
        )
    assert server._api_token is None
    assert server._allow_unauth is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "0.0.0.0" in m and "No authentication is configured" in m for m in warnings
    ), warnings
    client = TestClient(server.app)
    assert client.get("/health").status_code == 200


def test_bind_gate_non_loopback_with_token_starts_silently(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import logging

    with caplog.at_level(logging.WARNING, logger="palinode.api"):
        server = _reload_server(
            monkeypatch, PALINODE_API_HOST="0.0.0.0", PALINODE_API_TOKEN="t-secret"
        )
    assert server._api_token == "t-secret"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_bind_gate_public_intent_still_requires_token_despite_opt_out(
    monkeypatch: pytest.MonkeyPatch,
):
    """BIND_INTENT=public keeps its meaning — "token required" — and is not
    overridden by ALLOW_UNAUTH; the two vars pull opposite ways, so refuse."""
    for key in _GATE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PALINODE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("PALINODE_API_BIND_INTENT", "public")
    monkeypatch.setenv("PALINODE_API_ALLOW_UNAUTH", "1")
    import palinode.api.server  # noqa: F401

    with pytest.raises(SystemExit) as excinfo:
        importlib.reload(palinode.api.server)
    assert "PALINODE_API_BIND_INTENT=public requires" in str(excinfo.value)


def test_bind_gate_fires_under_uvicorn_direct_import():
    """Real startup path: a fresh interpreter importing ``palinode.api.server:app``
    (what uvicorn does) with PALINODE_API_HOST=0.0.0.0 and no token exits
    non-zero with the refusal; the same env plus PALINODE_API_ALLOW_UNAUTH=1
    imports cleanly."""
    refused = _import_app_in_subprocess({"PALINODE_API_HOST": "0.0.0.0"})
    assert refused.returncode != 0, (
        f"Import should fail under 0.0.0.0+no-token, but exit was 0.\n"
        f"stderr: {refused.stderr}"
    )
    combined = (refused.stderr or "") + (refused.stdout or "")
    assert "REFUSING TO START" in combined, combined
    assert "PALINODE_API_ALLOW_UNAUTH=1" in combined

    allowed = _import_app_in_subprocess(
        {"PALINODE_API_HOST": "0.0.0.0", "PALINODE_API_ALLOW_UNAUTH": "1"}
    )
    assert allowed.returncode == 0, (
        f"Import should succeed with the explicit opt-out.\nstderr: {allowed.stderr}"
    )
    assert "REFUSING TO START" not in (allowed.stderr or "") + (allowed.stdout or "")


# ---------------------------------------------------------------------------
# Middleware behaviour — TestClient drives the full ASGI stack
# ---------------------------------------------------------------------------


def test_no_token_loopback_search_works_without_auth(monkeypatch: pytest.MonkeyPatch):
    """Local-dev default: /health AND /search must not require auth."""
    server = _reload_server(monkeypatch)
    client = TestClient(server.app, raise_server_exceptions=False)

    h = client.get("/health")
    assert h.status_code == 200

    s = client.post("/search", json={"query": "anything"})
    # /search may 200 or 5xx depending on backend availability, but it
    # must NOT 401 — that would mean auth was wired even though no token
    # is configured.
    assert s.status_code != 401


def test_token_set_no_auth_header_yields_401(monkeypatch: pytest.MonkeyPatch):
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-secret")
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post("/search", json={"query": "x"})
    assert r.status_code == 401
    assert r.json() == {"detail": "Unauthorized"}
    # WWW-Authenticate must indicate Bearer per RFC 6750
    assert r.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_health_endpoints_skip_auth_even_with_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """Probes (k8s, systemd, Tailscale Funnel) shouldn't have to know the token."""
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-secret")
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.get("/health")
    assert r.status_code == 200


def test_token_set_wrong_token_yields_401(monkeypatch: pytest.MonkeyPatch):
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-correct")
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post(
        "/search",
        json={"query": "x"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_token_set_correct_token_authorises(monkeypatch: pytest.MonkeyPatch):
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-correct")
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post(
        "/search",
        json={"query": "x"},
        headers={"Authorization": "Bearer t-correct"},
    )
    # The auth layer let it through. The route handler may return any
    # non-401 status (200 if Ollama happens to be up in CI; 500/422 if
    # not) — what matters here is that the middleware did not block.
    assert r.status_code != 401


@pytest.mark.parametrize(
    "header",
    [
        "Basic abc",       # wrong scheme
        "Bearer",          # missing token
        "bearer t-correct",  # lowercase scheme — RFC 7235 says Bearer is case-insensitive,
                             # but compare_digest is byte-exact and we MUST be strict.
        "Bearer  t-correct",  # extra space
        "Token t-correct",  # wrong scheme name
    ],
)
def test_malformed_authorization_header_yields_401(
    monkeypatch: pytest.MonkeyPatch, header: str
):
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-correct")
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post(
        "/search",
        json={"query": "x"},
        headers={"Authorization": header},
    )
    assert r.status_code == 401, f"header={header!r} should have 401'd"


# ---------------------------------------------------------------------------
# Direct ASGI tests for non-HTTP scope and timing-safe compare
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_scope_passes_through(monkeypatch: pytest.MonkeyPatch):
    """ASGI ``lifespan`` (and any non-http scope) must skip auth — auth on
    a lifespan event would deadlock startup."""
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-secret")
    called = False

    async def fake_app(scope, recv, snd):
        nonlocal called
        called = True

    mw = server._BearerAuthMiddleware(fake_app, token="t-secret")
    await mw({"type": "lifespan"}, lambda: None, lambda msg: None)
    assert called


def test_compare_digest_is_used():
    """The middleware MUST use ``hmac.compare_digest`` rather than ``==``
    to compare the bearer token, to avoid leaking it via response timing.

    A naive ``provided == expected`` short-circuits on the first byte
    mismatch in CPython's bytes comparison, leaking timing information
    per byte. ``compare_digest`` runs in constant time over the inputs.

    This test inspects the source of ``__call__`` and asserts the
    constant-time compare is referenced verbatim — a guard against a
    future refactor accidentally introducing ``provided == self._expected_header``.
    """
    from palinode.api import server  # noqa: PLC0415

    src = inspect.getsource(server._BearerAuthMiddleware.__call__)
    assert "hmac.compare_digest" in src, (
        "_BearerAuthMiddleware.__call__ must compare with hmac.compare_digest"
    )
    # The obvious antipattern (string ``==`` on the expected header) must
    # be absent. We don't blanket-ban ``==`` because legitimate uses like
    # ``scope.get("type") != "http"`` are unrelated to the secret compare.
    assert "self._expected_header ==" not in src
    assert "== self._expected_header" not in src


# ---------------------------------------------------------------------------
# Middleware registration order
# ---------------------------------------------------------------------------


def test_bearer_middleware_registered_on_app(monkeypatch: pytest.MonkeyPatch):
    """The middleware is wired into the app — a future refactor must not
    drop the ``add_middleware`` call without also dropping these tests."""
    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-secret")
    classes = [mw.cls for mw in server.app.user_middleware]
    assert server._BearerAuthMiddleware in classes


# ---------------------------------------------------------------------------
# Own clients send the bearer
#
# ``BearerAuthMiddleware`` has no loopback exemption, so the moment an
# operator sets ``PALINODE_API_TOKEN`` per the docs, the MCP server and the
# CLI — Palinode's own two API clients — must present the token or every
# call 401s. Both are driven here against the real FastAPI app with the
# middleware armed: MCP over ``httpx.ASGITransport``, the CLI over a transport
# that hands each request to ``TestClient``. The negative half of each pair
# (token server-side only) proves the header is what fixes it.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _own_client_memory(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A one-file memory dir so ``GET /list`` has something to return."""
    from palinode.core.config import config  # noqa: PLC0415

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "own-client.md").write_text(
        "---\ncategory: decisions\ntype: Decision\ndescription: own client\n"
        "last_updated: 2026-08-16\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    return tmp_path


def _testclient_transport(tc: TestClient):
    """Sync ``httpx`` transport that dispatches to an in-process ``TestClient``."""
    import httpx  # noqa: PLC0415

    def _handler(request: httpx.Request) -> httpx.Response:
        tc_response = tc.request(
            request.method,
            request.url.raw_path.decode("ascii"),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=tc_response.status_code,
            headers=tc_response.headers.multi_items(),
            content=tc_response.content,
            request=request,
        )

    return httpx.MockTransport(_handler)


@pytest.mark.asyncio
async def test_mcp_client_sends_bearer_when_token_configured(
    monkeypatch: pytest.MonkeyPatch, _own_client_memory
):
    import httpx  # noqa: PLC0415

    from palinode import mcp  # noqa: PLC0415

    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-own-client")
    monkeypatch.setattr(mcp, "_http_transport", httpx.ASGITransport(app=server.app))
    monkeypatch.setattr(mcp, "_http_client", None)

    # Token set server-side AND client-side (same env var, same loader).
    result = await mcp.call_tool("palinode_list", {})
    text = result[0].text
    assert not text.startswith(mcp.DISPATCH_ERROR_PREFIXES), text
    assert "own-client.md" in text

    # Same server, token withdrawn from the client's environment only: the
    # request must now be refused, proving the bearer is what let it through.
    monkeypatch.delenv("PALINODE_API_TOKEN")
    await mcp._close_http()
    result = await mcp.call_tool("palinode_list", {})
    text = result[0].text
    assert text.startswith("API Error:"), text
    assert "Unauthorized" in text


@pytest.mark.asyncio
async def test_mcp_reuses_one_client_across_calls(
    monkeypatch: pytest.MonkeyPatch, _own_client_memory
):
    """One lazily-created ``httpx.AsyncClient`` per process, closed on shutdown."""
    import httpx  # noqa: PLC0415

    from palinode import mcp  # noqa: PLC0415

    server = _reload_server(monkeypatch)
    monkeypatch.setattr(mcp, "_http_transport", httpx.ASGITransport(app=server.app))
    monkeypatch.setattr(mcp, "_http_client", None)

    await mcp.call_tool("palinode_list", {})
    first = mcp._http_client
    assert isinstance(first, httpx.AsyncClient)
    await mcp.call_tool("palinode_list", {})
    assert mcp._http_client is first

    await mcp._close_http()
    assert first.is_closed
    assert mcp._http_client is None


def test_cli_client_sends_bearer_when_token_configured(
    monkeypatch: pytest.MonkeyPatch, _own_client_memory
):
    from click.testing import CliRunner  # noqa: PLC0415

    from palinode.cli import main as cli  # noqa: PLC0415
    from palinode.cli import _api  # noqa: PLC0415

    server = _reload_server(monkeypatch, PALINODE_API_TOKEN="t-own-client")
    tc = TestClient(server.app, raise_server_exceptions=False)

    # Default constructor path: headers come from PalinodeAPI itself.
    api = _api.PalinodeAPI(transport=_testclient_transport(tc))
    assert api.client.headers["authorization"] == "Bearer t-own-client"
    assert api.list_files()[0]["file"] == "decisions/own-client.md"

    monkeypatch.setattr(_api.api_client, "client", api.client)
    out = CliRunner().invoke(cli, ["list", "--format", "json"]).output
    assert "own-client.md" in out
    assert "401" not in out

    # Token withdrawn from the CLI's environment only → 401 from the same app.
    monkeypatch.delenv("PALINODE_API_TOKEN")
    unauth = _api.PalinodeAPI(transport=_testclient_transport(tc))
    assert "authorization" not in unauth.client.headers
    with pytest.raises(_api.HTTPStatusError) as exc:
        unauth.list_files()
    assert exc.value.response.status_code == 401
