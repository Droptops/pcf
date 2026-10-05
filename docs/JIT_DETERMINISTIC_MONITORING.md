# Deterministic JIT active monitoring

Active deterministic routes are monitored with evidence bound to the exact active route generation.

## Health states

A monitoring window produces one of three effective states:

- **healthy**: enough observations and the existing promotion/quality/risk/economics gates still pass;
- **demote**: enough observations exist and the gates fail;
- **hold**: the window is too small to make either claim.

Insufficient evidence does not automatically demote a low-traffic route.

## Demotion

`demote_unhealthy_deterministic_active()` accepts only actionable failing health evidence tied to the current route ID, generation, candidate key, fingerprint, and monitoring window. Stale evidence from a prior active generation cannot demote a newer generation.

An actionable failure moves the route from `ACTIVE` back to `CANARY` at the configured reduced traffic fraction. From there it can be re-evaluated, rolled back further through existing lifecycle APIs, or promoted again only with fresh canary evidence.

## Scope

The reference monitor uses paired `TraceReplaySummary` evidence and the existing `PromotionEvaluator`. Production deployments may source that summary from live sampled traffic, but the evidence binding and lifecycle rules should remain unchanged.
