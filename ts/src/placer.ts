/**
 * Self-placing memory: each module picks the front of the prompt or the tail (after the conversation history)
 * from its own observed change rate. A port of `pcf.placement.MemoryPlacer`; the decisions match the Python
 * implementation turn for turn (test/placer.test.ts).
 *
 * A front module is re-billed with everything after it when it changes: expected p * (m + H) per turn, where H is
 * the front modules after it plus the history. A tail module is paid every turn: m. With cache prices (write w,
 * read r, relative to uncached input) a module goes to the tail when p * (w - r) * (m + H) > (1 - r) * m.
 * Modules are assumed stable until a change is seen. Moving back to the front re-bills m + H once: while the
 * cache is warm a tail module returns only after going quiet (unchanged for more than twice its average gap
 * between changes) and when, at its decayed rate, the per-turn saving over the turns still to come repays that
 * rewrite. On a cold turn modules are re-placed from the decayed rate. While the lead, the front modules and the
 * history are below the provider's minimum cacheable length the prompt is not cached, so tail modules return to the
 * front: first, in order until the prompt reaches the minimum, those whose cache writes cost no more than uncached
 * input (p * (w - r) <= 1 - r), then any other that leaves the prompt below the minimum even in front. Pass the
 * stable/tool/system lead as `prefixTokens`.
 */

import { createHash } from "node:crypto";

const DIGEST = /^sha256:[0-9a-f]{64}$/;

export interface MemoryModule {
  /** Stable identity across turns. */
  id: string;
  /** The module's text as it will be sent; any change to it counts as a change. */
  content: string;
  /** Modules with instruction authority never move (instructions precede data). */
  authority?: "instruction" | "data";
}

export interface PlacedModule extends MemoryModule {
  /** False when the module should carry no cache marker this turn. */
  stable: boolean;
}

export interface Placement {
  front: PlacedModule[];
  tail: PlacedModule[];
}

export interface PlacerOptions {
  /** Token count of a text; supply the provider's tokenizer or an estimate. */
  countTokens: (text: string) => number;
  /** Cache write price relative to uncached input (Anthropic 5-minute cache: 1.25). */
  writeMultiplier?: number;
  /** Cache read price relative to uncached input (typically 0.1). */
  readMultiplier?: number;
  /** Weight of the old rate when re-placing on a cold turn, in [0, 1). */
  decay?: number;
  /** The provider's minimum cacheable prompt length; a prompt below it is not cached. */
  minCacheableTokens?: number;
  /** Typical conversation length, if known; otherwise as many more turns as have passed are assumed. */
  expectedTurns?: number;
  /** Stable identity of countTokens; required to detect tokenizer drift across durable restores. */
  tokenizerId?: string;
  /** Maximum retry decisions retained in durable state. */
  maxIdempotencyEntries?: number;
}

export interface MemoryPlacerState {
  memoryPlacerStateVersion: 2;
  configuration: {
    tokenizerId: string;
    writeMultiplier: number;
    readMultiplier: number;
    decay: number;
    minCacheableTokens: number;
    expectedTurns?: number;
    maxIdempotencyEntries: number;
  };
  revision: number;
  turns: number;
  quiet: Record<string, number>;
  seen: Record<string, State>;
  previousFront: string[] | null;
  /** Retry decisions by turn id; `revision` is the placer revision after that turn and orders eviction. */
  decisions: Record<string, Decision>;
}

export class ConcurrentPlacementUpdate extends Error {}

interface Decision {
  inputHash: string;
  front: Array<{ id: string; stable: boolean }>;
  revision: number;
}

interface State {
  obs: number;
  changes: number;
  rate: number;
  last: string;
  inTail: boolean;
}

export class MemoryPlacer {
  readonly writeMultiplier: number;
  readonly readMultiplier: number;
  readonly decay: number;
  readonly minCacheableTokens: number;
  readonly expectedTurns: number | undefined;
  readonly tokenizerId: string;
  readonly maxIdempotencyEntries: number;
  private readonly countTokens: (text: string) => number;
  private readonly seen = new Map<string, State>();
  private readonly quiet = new Map<string, number>();
  private turns = 0;
  private previousFront: string[] | null = null;
  private stateRevision = 0;
  private readonly decisions = new Map<string, Decision>();  // oldest first

  constructor(options: PlacerOptions) {
    const { countTokens, writeMultiplier = 1.25, readMultiplier = 0.1, decay = 0.7, minCacheableTokens = 0,
            expectedTurns, tokenizerId = "unspecified", maxIdempotencyEntries = 256 } = options;
    if (!Number.isInteger(minCacheableTokens) || minCacheableTokens < 0) throw new RangeError("minCacheableTokens must be a non-negative integer");
    if (expectedTurns !== undefined && (!Number.isInteger(expectedTurns) || expectedTurns < 1)) {
      throw new RangeError("expectedTurns must be a positive integer");
    }
    if (!Number.isInteger(maxIdempotencyEntries) || maxIdempotencyEntries < 1) {
      throw new RangeError("maxIdempotencyEntries must be a positive integer");
    }
    if (!(decay >= 0 && decay < 1)) throw new RangeError("decay must be in [0, 1)");
    if (!(readMultiplier >= 0 && readMultiplier < 1) || !(writeMultiplier > readMultiplier)) {
      throw new RangeError("need 0 <= readMultiplier < 1 and writeMultiplier > readMultiplier");
    }
    this.countTokens = countTokens;
    this.writeMultiplier = writeMultiplier;
    this.readMultiplier = readMultiplier;
    this.decay = decay;
    this.minCacheableTokens = minCacheableTokens;
    this.expectedTurns = expectedTurns;
    this.maxIdempotencyEntries = maxIdempotencyEntries;
    if (!tokenizerId) throw new RangeError("tokenizerId must be non-empty");
    this.tokenizerId = tokenizerId;
  }

  get revision(): number {
    return this.stateRevision;
  }

  private configuration(): MemoryPlacerState["configuration"] {
    return { tokenizerId: this.tokenizerId, writeMultiplier: this.writeMultiplier,
             readMultiplier: this.readMultiplier, decay: this.decay,
             minCacheableTokens: this.minCacheableTokens, expectedTurns: this.expectedTurns,
             maxIdempotencyEntries: this.maxIdempotencyEntries };
  }

  exportState(): MemoryPlacerState {
    return { memoryPlacerStateVersion: 2, configuration: this.configuration(), revision: this.stateRevision,
             turns: this.turns, quiet: Object.fromEntries([...this.quiet.entries()].sort()),
             seen: Object.fromEntries([...this.seen.entries()].sort()),
             previousFront: this.previousFront === null ? null : [...this.previousFront],
             decisions: Object.fromEntries([...this.decisions.entries()].sort().map(([id, decision]) =>
               [id, { inputHash: decision.inputHash, front: decision.front.map((item) => ({ ...item })),
                      revision: decision.revision }])) };
  }

  restoreState(state: MemoryPlacerState): void {
    if (state?.memoryPlacerStateVersion !== 2 || JSON.stringify(state.configuration) !== JSON.stringify(this.configuration())) {
      throw new RangeError("MemoryPlacer state configuration/version mismatch");
    }
    if (!Number.isInteger(state.revision) || state.revision < 0 || state.revision !== state.turns) {
      throw new RangeError("MemoryPlacer state revision/turns are invalid");
    }
    if (state.previousFront !== null && (!Array.isArray(state.previousFront) ||
        state.previousFront.some((id) => typeof id !== "string" || !id) ||
        new Set(state.previousFront).size !== state.previousFront.length)) {
      throw new RangeError("MemoryPlacer previousFront is invalid");
    }
    const quiet = new Map<string, number>();
    for (const [id, value] of Object.entries(state.quiet ?? {})) {
      if (!id.trim() || !Number.isInteger(value) || value < 0) {
        throw new RangeError("MemoryPlacer quiet state is invalid");
      }
      quiet.set(id, value);
    }
    const seen = new Map<string, State>();
    for (const [id, value] of Object.entries(state.seen ?? {})) {
      if (!id.trim() || typeof value !== "object" || value === null ||
          Object.keys(value).sort().join() !== "changes,inTail,last,obs,rate" ||
          !Number.isInteger(value.obs) || value.obs < 0 || !Number.isInteger(value.changes) ||
          value.changes < 0 || value.changes > value.obs || !(value.rate >= 0 && value.rate <= 1) ||
          typeof value.last !== "string" || !DIGEST.test(value.last) || typeof value.inTail !== "boolean") {
        throw new RangeError("MemoryPlacer observed state is invalid");
      }
      seen.set(id, { ...value });
    }
    const stored = Object.entries(state.decisions ?? {});
    if (stored.length > this.maxIdempotencyEntries) {  // an exported state never exceeds the cap
      throw new RangeError("MemoryPlacer state holds more decisions than maxIdempotencyEntries");
    }
    const decisions: Array<[string, Decision]> = [];
    for (const [id, value] of stored) {
      if (!id.trim() || typeof value !== "object" || value === null ||
          Object.keys(value).sort().join() !== "front,inputHash,revision" ||
          typeof value.inputHash !== "string" || !DIGEST.test(value.inputHash) || !Array.isArray(value.front) ||
          value.front.some((item) => typeof item !== "object" || item === null || typeof item.id !== "string" ||
                                     !item.id.trim() || Object.keys(item).sort().join() !== "id,stable" ||
                                     typeof item.stable !== "boolean") ||
          new Set(value.front.map((item) => item.id)).size !== value.front.length ||
          !Number.isInteger(value.revision) || value.revision < 1 || value.revision > state.revision) {
        throw new RangeError("MemoryPlacer idempotency state is invalid");
      }
      decisions.push([id, { inputHash: value.inputHash, front: value.front.map((item) => ({ ...item })),
                            revision: value.revision }]);
    }
    if (new Set(decisions.map(([, value]) => value.revision)).size !== decisions.length) {
      throw new RangeError("MemoryPlacer decision revisions must be unique");
    }
    decisions.sort((a, b) => a[1].revision - b[1].revision);  // oldest first, so eviction removes the lowest revision
    this.stateRevision = state.revision;
    this.turns = state.turns;
    this.previousFront = state.previousFront === null ? null : [...state.previousFront];
    this.quiet.clear();
    this.seen.clear();
    this.decisions.clear();
    for (const [id, value] of quiet) this.quiet.set(id, value);
    for (const [id, value] of seen) this.seen.set(id, value);
    for (const [id, value] of decisions) this.decisions.set(id, value);
  }

  private inputHash(memory: MemoryModule[], historyTokens: number, cold: boolean, prefixTokens: number): string {
    const input = JSON.stringify({ memory: memory.map(({ id, content, authority }) => ({ id, content, authority })),
                                   historyTokens, cold, prefixTokens });
    return `sha256:${createHash("sha256").update(input).digest("hex")}`;
  }

  private contentHash(content: string): string {
    return `sha256:${createHash("sha256").update(content).digest("hex")}`;
  }

  private replay(memory: MemoryModule[], decision: { front: Array<{ id: string; stable: boolean }> }): Placement {
    const byId = new Map(memory.map((item) => [item.id, item]));
    if (byId.size !== memory.length || decision.front.some((item) => !byId.has(item.id))) {
      throw new RangeError("stored placement decision does not match memory modules");
    }
    const frontIds = new Set(decision.front.map((item) => item.id));
    return { front: decision.front.map((item) => ({ ...byId.get(item.id)!, stable: item.stable })),
             tail: memory.filter((item) => !frontIds.has(item.id)).map((item) => ({ ...item, stable: false })) };
  }

  /**
   * Place this turn's modules; call once per turn. `historyTokens` is the size of the conversation so far.
   * Pass `cold: true` when the provider cache has expired, predicted from the time since the last request.
   * `prefixTokens` is the stable lead before memory (tools, system); with the front modules and history it decides
   * whether the prompt reaches the provider's cache minimum.
   *
   * On a warm turn where the front changes, some providers read only at markers present in the request, so a
   * marker must sit at an entry written earlier. When modules were only appended, the appended ones are returned
   * unstable and the previous last front module keeps the marker; otherwise the front modules after the first
   * are returned unstable and the first front module carries it.
   */
  split(memory: MemoryModule[], historyTokens: number,
        { cold = false, turnId, expectedRevision, prefixTokens = 0 }:
        { cold?: boolean; turnId?: string; expectedRevision?: number; prefixTokens?: number } = {}): Placement {
    if (turnId !== undefined && !turnId) throw new RangeError("turnId must be non-empty");
    if (expectedRevision !== undefined && (!Number.isInteger(expectedRevision) || expectedRevision < 0)) {
      throw new RangeError("expectedRevision must be a non-negative integer");
    }
    if (!Number.isInteger(prefixTokens) || prefixTokens < 0) throw new RangeError("prefixTokens must be non-negative");
    const inputHash = this.inputHash(memory, historyTokens, cold, prefixTokens);
    if (turnId !== undefined && this.decisions.has(turnId)) {
      const decision = this.decisions.get(turnId)!;
      if (decision.inputHash !== inputHash) throw new RangeError("turnId was already used with different inputs");
      return this.replay(memory, decision);
    }
    if (turnId !== undefined && expectedRevision === undefined) {
      throw new RangeError("a new durable turnId requires expectedRevision");
    }
    if (expectedRevision !== undefined && expectedRevision !== this.stateRevision) {
      throw new ConcurrentPlacementUpdate(
        `placement revision changed: expected ${expectedRevision}, found ${this.stateRevision}`);
    }
    let behind = historyTokens;
    this.turns += 1;
    const inTail: boolean[] = new Array(memory.length).fill(false);
    const sizes: number[] = new Array(memory.length).fill(0);
    const rates: number[] = new Array(memory.length).fill(0);
    for (let i = memory.length - 1; i >= 0; i--) {
      const module = memory[i];
      const moduleHash = this.contentHash(module.content);
      const prior = this.seen.get(module.id) ?? { obs: -1, changes: 0, rate: 0, last: moduleHash, inTail: false };
      const changed = moduleHash !== prior.last ? 1 : 0;
      const quiet = changed ? 0 : (this.quiet.get(module.id) ?? 0) + 1;
      this.quiet.set(module.id, quiet);
      let obs = prior.obs + 1;
      let changes = prior.changes + changed;
      const rate = (rates[i] = this.decay * prior.rate + (1 - this.decay) * changed);
      const m = (sizes[i] = this.countTokens(module.content));
      let tail = prior.inTail;
      if (module.authority === "instruction") {
        tail = false;
      } else if (cold) {
        tail = this.tailPays(rate, m, behind);
        if (!tail) {
          obs = 0;
          changes = 0;
        }
      } else if (!tail) {
        tail = this.tailPays(changes / (obs + 1), m, behind);
      } else if (quiet > (2 * (obs + 1)) / Math.max(changes, 1) && this.returnPays(rate, m, behind)) {
        tail = false;
        obs = 0;
        changes = 0;
      }
      this.seen.set(module.id, { obs, changes, rate, last: moduleHash, inTail: tail });
      inTail[i] = tail;
      if (!tail) behind += m;
    }

    // A prompt below the provider minimum is not cached. While lead + front modules + history is below it, tail modules
    // go back to the front in two passes. First, in memory order (so list modules stable-first) until the prompt
    // reaches the minimum, those whose decayed rate p has p * (w - r) <= 1 - r: once the prompt is cached, writes on
    // their changes cost no more than the uncached input the tail would leave. Then any other tail module that leaves
    // the prompt below the minimum even in front: nothing is cached either way, so moving saves nothing. In the other
    // order, or in one pass, a module moved because the prompt stayed uncached could end up at the front of a prompt
    // that a later move lifts to the minimum, rewriting everything after it on each of its frequent changes. A second-
    // pass move never lifts the prompt to the minimum, so it cannot invalidate any other decision.
    let cacheable = prefixTokens + behind;  // behind now holds history plus every front module
    const w = this.writeMultiplier;
    const r = this.readMultiplier;
    const minimum = this.minCacheableTokens;
    for (const band of [true, false]) {
      for (let i = 0; i < memory.length; i++) {
        if (inTail[i] && cacheable < minimum &&
            (band ? rates[i] * (w - r) <= 1 - r : cacheable + sizes[i] < minimum)) {
          inTail[i] = false;
          this.seen.set(memory[i].id, { ...this.seen.get(memory[i].id)!, inTail: false });
          cacheable += sizes[i];
        }
      }
    }

    let front: PlacedModule[] = memory.filter((_, i) => !inTail[i]).map((m) => ({ ...m, stable: true }));
    const tail: PlacedModule[] = memory.filter((_, i) => inTail[i]).map((m) => ({ ...m, stable: false }));
    const ids = front.map((m) => m.id);
    const previous = this.previousFront ?? [];
    this.previousFront = ids;
    const moved = previous.length > 0 && !cold && (previous.length !== ids.length || previous.some((id, i) => id !== ids[i]));
    if (moved) {
      const appended = previous.every((id, i) => ids[i] === id);
      const keep = appended ? previous.length : 1;
      front = front.map((m, i) => (i < keep ? m : { ...m, stable: false }));
    }
    const result = { front, tail };
    this.stateRevision += 1;
    if (turnId !== undefined) {
      this.decisions.set(turnId, { inputHash, front: front.map(({ id, stable }) => ({ id, stable })),
                                   revision: this.stateRevision });
      while (this.decisions.size > this.maxIdempotencyEntries) {  // insertion order is revision order
        this.decisions.delete(this.decisions.keys().next().value!);
      }
    }
    return result;
  }

  splitAndSnapshot(memory: MemoryModule[], historyTokens: number,
                   options: { cold?: boolean; turnId?: string; expectedRevision?: number; prefixTokens?: number } = {}):
                   { placement: Placement; state: MemoryPlacerState } {
    const placement = this.split(memory, historyTokens, options);
    return { placement, state: this.exportState() };
  }

  private returnPays(p: number, m: number, behind: number): boolean {
    const w = this.writeMultiplier;
    const r = this.readMultiplier;
    const saving = (1 - r) * m - p * (w - r) * (m + behind);
    const turns = this.expectedTurns === undefined ? this.turns : this.expectedTurns - this.turns;
    return saving > 0 && saving * turns > (w - r) * (m + behind);
  }

  private tailPays(p: number, m: number, behind: number): boolean {
    const w = this.writeMultiplier;
    const r = this.readMultiplier;
    return p * (w - r) * (m + behind) > (1 - r) * m;
  }
}
