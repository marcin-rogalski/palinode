# openclaw-palinode — Installation and Host Configuration

## Prerequisites

- **Palinode API server** running and reachable (default: `http://localhost:6340`). See the root-level `README.md` for server setup.
- **OpenClaw 2026.10.3** or later. This release uses the current manifest's root CLI command declaration, which is absent from the supplied 2026.5.2 manifest contract.

## Installing the plugin

For bundled releases, the plugin is included automatically. For non-bundled installs, copy or symlink the `plugin/` directory into your OpenClaw plugins folder and reference it from your host config.

## Host-side hook access

Non-bundled plugins need explicit conversation access for `before_prompt_build` and `agent_end`. `allowPromptInjection` defaults to allowed, but an explicit `false` blocks `before_prompt_build`. The `palinode_*` tools are registered separately; they require the plugin to load and their names to appear in the manifest's `contracts.tools`.

Add the following block to your OpenClaw host config under `plugins.entries.openclaw-palinode`:

```json
{
  "plugins": {
    "entries": {
      "openclaw-palinode": {
        "hooks": {
          "allowPromptInjection": true,
          "allowConversationAccess": true
        }
      }
    }
  }
}
```

### Why each flag is required

| Flag | Required for | Effect without it |
|------|-------------|-------------------|
| `allowPromptInjection` | `autoRecall` through `before_prompt_build` | An explicit `false` blocks prompt mutation; omitted defaults to allowed |
| `allowConversationAccess` | `before_prompt_build` recall and `agent_end` capture for a non-bundled plugin | Those conversation hooks are blocked |

The example sets both flags explicitly. Conversation access is required for non-bundled installs; prompt injection must be allowed for automatic recall.

`after_compaction` and `before_reset` remain lifecycle observations. The plugin declares `activation.onStartup` so those hooks and its service register when the Gateway starts.

## Verifying the install

### 1. Check the gateway log at startup

When the plugin registers, you should see a line like:

```
openclaw-palinode: registered (api: http://localhost:6340, dir: /home/user/palinode, autoRecall: true, autoCapture: false)
```

The log line alone does not prove the active Gateway has the hooks. Run `openclaw plugins inspect openclaw-palinode --runtime --json`, then send an agent turn and look for `openclaw-palinode: turn 1 — profile=...`. If the turn line is absent, check `allowConversationAccess`, `allowPromptInjection`, and the plugin's enabled/allowlist state.

### 2. Confirm the agent quotes real injected memory

Send the agent a message that should match something in your Palinode memory store. If autoRecall is working, the agent's reply will reference specific facts from your memory files. If the agent's answer looks generic or hallucinates details that contradict your actual memory files, the context injection is not reaching it.

You can also ask the agent directly:

> "What context did you receive from Palinode at the start of this session?"

A working install will quote the `<palinode-memory profile="...">` block contents. A misconfigured one will say it received nothing.

### 3. Verify tool availability

Ask the agent to call `palinode_status`. It should return file counts and index health from your Palinode server. A `tool not found` error means the plugin did not register, not a hook issue — check that the plugin loaded at all (look for the `registered` log line).

## Plugin config fields

These live under `plugins.entries.openclaw-palinode.config`. The `config` and `hooks` blocks are siblings:

| Field | Default | Description |
|-------|---------|-------------|
| `palinodeApiUrl` | `http://localhost:6340` | Palinode API server URL |
| `palinodeDir` | `~/palinode` | Path to your Palinode memory directory |
| `autoRecall` | `true` | Inject core memory + semantic recall before each agent turn |
| `autoCapture` | `false` | Opt in to bounded transcript capture through the API at agent end and reset |
| `midTurnMode` | `none` | Core-memory injection on turns after the first: `none` (skip), `summary` (one line per core file), `full` (whole files) |
| `recallProfile` | `coding` | Named recall preset: `coding`, `monitoring`, `investigation`, `writing`, `conversation`, `minimal`, `off` |
| `recallProfileConfig` | — | Per-field overrides on top of the named preset |
| `promptsDir` | `specs/prompts` | Path to extraction prompts, relative to `palinodeDir` |

### Capture and pause controls

Transcript capture is off until `autoCapture: true` is explicitly configured.
Both agent-end and reset capture check the server policy before inspecting the
last 20 messages, then send at most 2,000 characters through `/session-end`
for normal store writes and Git provenance. There is no local transcript fallback.
The hook must provide an absolute workspace directory; unknown scope, unavailable
controls, or denied policy skips automatic capture and recall.

`palinode controls pause` stops future supported API capture and recall.
`palinode controls resume` restores them; the plugin still requires its capture
opt-in. Recall is checked again before returning new context. Already delivered
client context and a request already executing cannot be recalled by this control.
The plugin requires the matching controls-capable API version.
