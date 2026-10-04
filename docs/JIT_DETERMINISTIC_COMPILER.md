# JIT deterministic compiler v1

This is the first executable compile target in PCF's JIT control plane.

The target is intentionally not arbitrary generated code. A compiled artifact is a validated declarative expression tree over finite JSON-like inputs. The runtime does not use `eval`, `exec`, dynamic imports, filesystem access, network access, subprocesses, attribute lookup, callbacks, or user-provided functions.

## Supported operations

- input lookup through mapping keys only
- JSON constants
- object and array construction
- bounded arithmetic: add, subtract, multiply, divide
- numeric comparisons
- boolean `and`, `or`, `not`
- conditional expressions
- string concat, lower, upper, strip

The compiler enforces node, depth, string-length, and collection-size limits. Missing inputs, invalid operand types, division by zero, non-finite values, unsupported operations, malformed nodes, and forbidden path components fail closed.

## Artifact identity

Each artifact has a stable SHA-256 identity over its schema version, name, version, declared dependencies, and canonical JSON expression. Dependency versions are part of identity so a change to a fixed lookup/formula/policy dependency produces a different artifact.

## Authority and promotion boundary

A deterministic artifact is only an execution primitive. It does not authorize itself.

Promotion still requires the existing PCF sequence:

1. task fingerprint stability and audit coverage
2. FAAR authority gate
3. quality and HToken admissibility
4. shadow evidence
5. promotion statistics
6. route registry lifecycle
7. canary before active traffic

Jev may provide task-shape evidence but remains advisory and cannot authorize or promote deterministic execution.

## Intended first workloads

Good v1 targets are stable, bounded transforms such as unit conversions, fixed formulas, fixed-schema projection/validation helpers, normalization, and policy-table mappings whose dependencies are explicit and versioned.

Open-ended generation, semantic interpretation, external side effects, dynamic data access, and authority decisions are out of scope.
