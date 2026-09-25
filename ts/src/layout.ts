/**
 * Put placed memory into requests an application already builds. Memory is data, so it goes into user content,
 * never the system prompt: front modules as a user message before the history, tail modules at the start of the
 * final user message.
 */
import type { Placement, PlacedModule } from "./placer.ts";

type Block = { type: string; text?: string; cache_control?: unknown; [key: string]: unknown };
type AnthropicMessage = { role: "user" | "assistant"; content: string | Block[] };
export interface AnthropicRequest {
  system?: string | Block[];
  messages: AnthropicMessage[];
  [key: string]: unknown;
}

const blocks = (content: string | Block[]): Block[] =>
  typeof content === "string" ? [{ type: "text", text: content }] : content.map(({ cache_control: _, ...b }) => ({ ...b }));
const text = (m: PlacedModule): Block => ({ type: "text", text: m.content });

/**
 * An Anthropic Messages request with memory placed and cache markers set. `request.messages` is the history
 * followed by the new user message. Existing cache markers are replaced; at most three are set:
 * the first front module (or the end of the system prompt when nothing is in front), the last stable front
 * module, and the end of the last history message, which the next request reads.
 */
export function layoutAnthropic(request: AnthropicRequest, placement: Placement,
                                { ttl }: { ttl?: "5m" | "1h" } = {}): AnthropicRequest {
  const marker = ttl ? { type: "ephemeral", ttl } : { type: "ephemeral" };
  const messages = request.messages;
  if (!messages.length || messages[messages.length - 1].role !== "user") {
    throw new Error("the last message must be the new user message");
  }
  const system = request.system === undefined ? undefined : blocks(request.system);
  const front = placement.front.map(text);
  const history = messages.slice(0, -1).map((m) => ({ role: m.role, content: blocks(m.content) }));
  const last = messages[messages.length - 1];
  const question = { role: "user" as const, content: [...placement.tail.map(text), ...blocks(last.content)] };

  if (front.length) {
    front[0].cache_control = marker;
    const lastStable = placement.front.map((m) => m.stable).lastIndexOf(true);
    if (lastStable > 0) front[lastStable].cache_control = marker;
  } else if (system?.length) {
    system[system.length - 1].cache_control = marker;
  }
  const lastHistory = history[history.length - 1];
  if (lastHistory) lastHistory.content[lastHistory.content.length - 1].cache_control = marker;

  const out: AnthropicRequest = { ...request, messages: [...(front.length ? [{ role: "user" as const, content: front }] : []),
                                                        ...history, question] };
  if (system !== undefined) out.system = system;
  return out;
}

type ChatMessage = { role: string; content: string | Array<{ type: string; text?: string; [key: string]: unknown }>;
                     [key: string]: unknown };

/**
 * OpenAI chat messages with memory placed. Automatic prefix caching needs no markers, only a stable order:
 * leading system or developer messages, then front memory, then history, then tail memory with the new message.
 */
export function layoutOpenAIChat(messages: ChatMessage[], placement: Placement): ChatMessage[] {
  if (!messages.length || messages[messages.length - 1].role !== "user") {
    throw new Error("the last message must be the new user message");
  }
  const lead = messages.findIndex((m) => m.role !== "system" && m.role !== "developer");
  const head = messages.slice(0, lead);
  const history = messages.slice(lead, -1);
  const last = messages[messages.length - 1];
  const parts = (content: ChatMessage["content"]) => (typeof content === "string" ? [{ type: "text", text: content }] : content);
  const front: ChatMessage[] = placement.front.length
    ? [{ role: "user", content: placement.front.map((m) => ({ type: "text", text: m.content })) }] : [];
  const question = { ...last, content: [...placement.tail.map((m) => ({ type: "text", text: m.content })), ...parts(last.content)] };
  return [...head, ...front, ...history, question];
}
