import * as path from "path";
import { afterEach, describe, expect, it } from "vitest";

import palinodePlugin from "../index";

type Execute = (id: string, params: Record<string, unknown>) => Promise<any>;

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
});

function captureSearchTool(): { parameters: any; execute: Execute } {
  let captured: { parameters: any; execute: Execute } | undefined;
  const api: any = {
    pluginConfig: {
      palinodeDir: path.join(process.cwd(), "test-memory"),
      promptsDir: "specs/prompts",
      autoRecall: false,
      autoCapture: false,
      midTurnMode: "none",
    },
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
    registerTool: (tool: any) => {
      if (typeof tool === "function") tool = tool({});
      if (tool.name === "palinode_search") captured = tool;
    },
    on: () => undefined,
    registerCli: () => undefined,
    registerService: () => undefined,
  };
  palinodePlugin.register(api);
  if (!captured) throw new Error("palinode_search was not registered");
  return captured;
}

describe("palinode_search takes no where", () => {
  it("declares no where parameter", () => {
    const { parameters } = captureSearchTool();

    expect(Object.keys(parameters.properties)).not.toContain("where");
  });

  it("sends no where field, whatever the caller passes", async () => {
    const bodies: Record<string, unknown>[] = [];
    global.fetch = async (_url: any, init?: any) => {
      if (init?.body) bodies.push(JSON.parse(init.body));
      return new Response(JSON.stringify([]), { status: 200 });
    };
    const { execute } = captureSearchTool();

    await execute("call-1", { query: "release checklist", where: ["status=open"] });

    const searchBodies = bodies.filter((body) => body.query === "release checklist");
    expect(searchBodies).toHaveLength(1);
    expect(searchBodies[0]).not.toHaveProperty("where");
  });
});
