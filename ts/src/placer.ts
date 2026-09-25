/**
 * Self-placing memory: each module picks the front of the prompt or the tail (after the conversation history)
 * from its own observed change rate. A port of `pcf.placement.MemoryPlacer`; the decisions match the Python
 * implementation turn for turn (test/placer.test.ts).
 *
 * A front module is re-billed with everything after it when it changes: expected p * (m + H) per turn, where H is
 * the front modules after it plus the history. A tail module is paid every turn: m. With cache prices (write w,
 * read r, relative to uncached input) a module goes to the tail when p * (w - r) * (m + H) > (1 - r) * m.
 * Modules are assumed stable until a change is seen. Moving back to the front re-bills m + H at once, so while
 * the cache is warm the tail is sticky; on a cold turn modules are re-placed from a decayed change rate.
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
  private readonly countTokens: (text: string) => number;
  private readonly seen = new Map<string, State>();
  private previousFront: string[] | null = null;

  constructor(options: PlacerOptions) {
    const { countTokens, writeMultiplier = 1, readMultiplier = 0, decay = 0.7 } = options;
    if (!(decay >= 0 && decay < 1)) throw new RangeError("decay must be in [0, 1)");
    if (!(readMultiplier >= 0 && readMultiplier < 1) || !(writeMultiplier > readMultiplier)) {
      throw new RangeError("need 0 <= readMultiplier < 1 and writeMultiplier > readMultiplier");
    }
    this.countTokens = countTokens;
    this.writeMultiplier = writeMultiplier;
    this.readMultiplier = readMultiplier;
    this.decay = decay;
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
    const inTail: boolean[] = new Array(memory.length).fill(false);
    for (let i = memory.length - 1; i >= 0; i--) {
      const module = memory[i];
      const prior = this.seen.get(module.id) ?? { obs: -1, changes: 0, rate: 0, last: module.content, inTail: false };
      const changed = module.content !== prior.last ? 1 : 0;
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
      } else if (!tail) {
        tail = this.tailPays(changes / (obs + 1), m, behind);
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

  private tailPays(p: number, m: number, behind: number): boolean {
    const w = this.writeMultiplier;
    const r = this.readMultiplier;
    return p * (w - r) * (m + behind) > (1 - r) * m;
  }
}
