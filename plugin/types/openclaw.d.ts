// Build-time fallback for the SDK core subpath. The release check uses the
// real OpenClaw declarations and excludes this file.
declare module "openclaw/plugin-sdk/core" {
  type PalinodeToolResult = {
    content: { type: "text"; text: string }[];
    details: unknown;
  };

  export interface OpenClawPluginApi {
    pluginConfig?: Record<string, unknown>;
    logger: {
      info(message: string): void;
      warn(message: string): void;
      error(message: string): void;
    };
    registerTool(tool: {
      name: string;
      label?: string;
      description: string;
      parameters: unknown;
      execute: (...args: any[]) => Promise<PalinodeToolResult>;
    } | ((context: { workspaceDir?: string }) => {
      name: string;
      label?: string;
      description: string;
      parameters: unknown;
      execute: (...args: any[]) => Promise<PalinodeToolResult>;
    }), options?: { name?: string; names?: string[]; optional?: boolean }): void;
    on(hook: "message_received" | "after_compaction" | "before_prompt_build" | "agent_end" | "before_reset",
      handler: (...args: any[]) => unknown): void;
    registerCli(registrar: (context: { program: any }) => void | Promise<void>, options?: {
      commands?: readonly string[];
      descriptors?: readonly { name: string; description: string; hasSubcommands: boolean }[];
    }): void;
    registerService(service: {
      id: string;
      start: (...args: any[]) => void | Promise<void>;
      stop?: (...args: any[]) => void | Promise<void>;
    }): void;
  }

  export interface OpenClawPluginDefinition {
    id?: string;
    name?: string;
    description?: string;
    version?: string;
    configSchema?: { parse?: (value: unknown) => unknown; jsonSchema?: object };
    register?: (api: OpenClawPluginApi) => void;
  }
}
