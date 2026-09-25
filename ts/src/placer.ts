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
 * rewrite. On a cold turn modules are re-placed from the decayed rate. Nothing moves while a module and what
 * follows it are below the provider's minimum cacheable length.
 */

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
  /** The provider's minimum cacheable prompt length; below it nothing is cached, so nothing moves. */
  minCacheableTokens?: number;
  /** Typical conversation length, if known; otherwise as many more turns as have passed are assumed. */
  expectedTurns?: number;
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
  private readonly countTokens: (text: string) => number;
  private readonly seen = new Map<string, State>();
  private readonly quiet = new Map<string, number>();
  private turns = 0;
  private previousFront: string[] | null = null;

  constructor(options: PlacerOptions) {
    const { countTokens, writeMultiplier = 1.25, readMultiplier = 0.1, decay = 0.7, minCacheableTokens = 0,
            expectedTurns } = options;
    if (!Number.isInteger(minCacheableTokens) || minCacheableTokens < 0) throw new RangeError("minCacheableTokens must be a non-negative integer");
    if (expectedTurns !== undefined && (!Number.isInteger(expectedTurns) || expectedTurns < 1)) {
      throw new RangeError("expectedTurns must be a positive integer");
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
  }

  /**
   * Place this turn's modules; call once per turn. `historyTokens` is the size of the conversation so far.
   * Pass `cold: true` when the provider cache has expired, predicted from the time since the last request.
   *
   * On a warm turn where the front changes, some providers read only at markers present in the request, so a
   * marker must sit at an entry written earlier. When modules were only appended, the appended ones are returned
   * unstable and the previous last front module keeps the marker; otherwise the front modules after the first
   * are returned unstable and the first front module carries it.
   */
  split(memory: MemoryModule[], historyTokens: number, { cold = false } = {}): Placement {
    let behind = historyTokens;
    this.turns += 1;
    const inTail: boolean[] = new Array(memory.length).fill(false);
    for (let i = memory.length - 1; i >= 0; i--) {
      const module = memory[i];
      const prior = this.seen.get(module.id) ?? { obs: -1, changes: 0, rate: 0, last: module.content, inTail: false };
      const changed = module.content !== prior.last ? 1 : 0;
      const quiet = changed ? 0 : (this.quiet.get(module.id) ?? 0) + 1;
      this.quiet.set(module.id, quiet);
      let obs = prior.obs + 1;
      let changes = prior.changes + changed;
      const rate = this.decay * prior.rate + (1 - this.decay) * changed;
      const m = this.countTokens(module.content);
      let tail = prior.inTail;
      if (module.authority === "instruction") {
        tail = false;
      } else if (cold) {
        tail = this.tailPays(rate, m, behind);
        if (!tail) {
          obs = 0;
          changes = 0;
        }
      } else if (m + behind < this.minCacheableTokens) {
        // nothing here is cached: moving it saves nothing
      } else if (!tail) {
        tail = this.tailPays(changes / (obs + 1), m, behind);
      } else if (quiet > (2 * (obs + 1)) / Math.max(changes, 1) && this.returnPays(rate, m, behind)) {
        tail = false;
        obs = 0;
        changes = 0;
      }
      this.seen.set(module.id, { obs, changes, rate, last: module.content, inTail: tail });
      inTail[i] = tail;
      if (!tail) behind += m;
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
    return { front, tail };
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
