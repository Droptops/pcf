"""Hard, near-boundary Jev task-shape benchmark cases.

These cases intentionally remove explicit routing hints and include paired tasks
whose surface form is similar but whose execution requirements differ. They are
used to measure route-shape discrimination before Jev advice may narrow PCF's
candidate set.
"""
from __future__ import annotations

from .jev_benchmark import JevBenchmarkCase
from .types import ExecutionPrimitive


def hard_jev_benchmark_cases() -> tuple[JevBenchmarkCase, ...]:
    """Return adjudicated OOD/near-boundary task-shape cases.

    The suite is balanced across all five execution primitives. Labels are
    intentionally based on the minimum sufficient execution primitive, assuming
    authority/safety/economic gates are handled elsewhere by PCF/FAAR.
    """
    return (
        # Result cache: reuse is semantically valid because the relevant state is unchanged.
        JevBenchmarkCase(
            "ood_cache_paraphrase_same_record_version",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "request": "What support number should I use for Acme?",
                "prior_request": "Give me Acme's support phone number.",
                "prior_answer_status": "verified",
                "record_version": "customer-directory:v18",
                "prior_record_version": "customer-directory:v18",
            },
            ("ood", "boundary", "cache-vs-model"),
        ),
        JevBenchmarkCase(
            "ood_cache_same_document_hash_new_wording",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "request": "Which covenant bucket does this clause fall into?",
                "document_hash": "sha256:4aa1",
                "prior_document_hash": "sha256:4aa1",
                "taxonomy_version": "loan-covenants:7",
                "prior_taxonomy_version": "loan-covenants:7",
                "prior_result_status": "verified",
            },
            ("ood", "boundary", "cache-vs-specialist"),
        ),
        JevBenchmarkCase(
            "ood_cache_repeat_report_same_inputs",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "request": "Show the same weekly totals again.",
                "input_snapshot": "week-2026-40:v3",
                "prior_input_snapshot": "week-2026-40:v3",
                "calculation_version": "rev-12",
                "prior_calculation_version": "rev-12",
                "prior_result_status": "verified",
            },
            ("ood", "boundary", "cache-vs-code"),
        ),
        JevBenchmarkCase(
            "ood_cache_repeat_summary_same_source",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "request": "Remind me of the three takeaways from that memo.",
                "source_hash": "sha256:b91f",
                "prior_source_hash": "sha256:b91f",
                "prior_summary_status": "verified",
            },
            ("ood", "boundary", "cache-vs-small"),
        ),

        # Deterministic code: exact rules/transformations, despite natural-language wrappers.
        JevBenchmarkCase(
            "ood_code_discount_formula",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "request": "What is the invoice total after a 7.5% discount and 8.25% tax?",
                "subtotal": 1842.17,
                "rule": "discount first, then tax, round final total to cents",
            },
            ("ood", "boundary", "code-vs-small"),
        ),
        JevBenchmarkCase(
            "ood_code_fixed_policy_table",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "request": "Return the escalation level for severity 2 and customer tier platinum.",
                "table": {
                    "severity-1": {"platinum": "P0"},
                    "severity-2": {"platinum": "P1"},
                },
                "missing_value_behavior": "error",
            },
            ("ood", "boundary", "code-vs-specialist"),
        ),
        JevBenchmarkCase(
            "ood_code_phone_normalization",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "request": "Normalize +1 (415) 555-0198 to E.164.",
                "default_country": "US",
                "output_format": "E.164",
            },
            ("ood", "boundary", "code-vs-small"),
        ),
        JevBenchmarkCase(
            "ood_code_schema_projection",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "request": "Return only id, created_at, and status from this object using the fixed field mapping.",
                "mapping": {"id": "id", "created_at": "createdAt", "status": "state"},
                "inference_required": False,
            },
            ("ood", "boundary", "code-vs-small"),
        ),

        # Specialist: bounded semantic classification in a stable domain taxonomy.
        JevBenchmarkCase(
            "ood_specialist_sip_incident_family",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "request": "Assign this incident to one of our twelve carrier fault families.",
                "notes": "Calls connect, then drop at 31-33 seconds; ACK appears on one leg but not the other behind the SBC.",
                "taxonomy": "carrier-faults:v12",
            },
            ("ood", "boundary", "specialist-vs-frontier"),
        ),
        JevBenchmarkCase(
            "ood_specialist_loan_clause_bucket",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "request": "Map this clause to the closest category in our fixed covenant taxonomy.",
                "excerpt": "The borrower shall maintain consolidated fixed-charge coverage not less than 1.25 to 1.00.",
                "taxonomy": "credit-covenants:v7",
            },
            ("ood", "boundary", "specialist-vs-small"),
        ),
        JevBenchmarkCase(
            "ood_specialist_clinical_code_family",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "request": "Choose one of the predefined coding families for this de-identified note.",
                "note": "Persistent atrial fibrillation documented; no ablation performed during this encounter.",
                "allowed_families": ["rhythm", "ischemia", "valvular", "heart-failure"],
            },
            ("ood", "boundary", "specialist-vs-small"),
        ),
        JevBenchmarkCase(
            "ood_specialist_security_alert_family",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "request": "Classify this alert into the SOC's fixed investigation taxonomy.",
                "evidence": "OAuth token reused from two ASNs within four minutes; impossible-travel detector also fired.",
                "taxonomy": "soc-investigation:v9",
            },
            ("ood", "boundary", "specialist-vs-frontier"),
        ),

        # Small model: generic bounded language work even when surface text contains domain jargon.
        JevBenchmarkCase(
            "ood_small_rewrite_domain_jargon",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "request": "Rewrite this note to be concise and professional without changing technical terms.",
                "text": "SIP 503s spiked after the route flip; pls have carrier confirm if they changed CPS caps.",
            },
            ("ood", "boundary", "small-vs-specialist"),
        ),
        JevBenchmarkCase(
            "ood_small_extract_customer_asks",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "request": "List the customer's three explicit asks and paraphrase each in one sentence.",
                "text": "Please send revised pricing, confirm the migration date, and tell us whether SSO is included.",
            },
            ("ood", "boundary", "small-vs-code"),
        ),
        JevBenchmarkCase(
            "ood_small_summary_with_numbers",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "request": "Summarize this update in two bullets, preserving the numbers.",
                "text": "Pilot adoption reached 62% this week, up from 48%. Median setup time fell from 18 to 11 minutes.",
            },
            ("ood", "boundary", "small-vs-code"),
        ),
        JevBenchmarkCase(
            "ood_small_email_intent",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "request": "State the sender's primary intent in one short phrase.",
                "text": "We like the proposal, but before legal review we need an updated security appendix and DPA.",
            },
            ("ood", "boundary", "small-vs-specialist"),
        ),

        # Frontier: familiar domains but novel/ambiguous interactions that exceed a bounded taxonomy.
        JevBenchmarkCase(
            "ood_frontier_telecom_cross_layer_root_cause",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "request": "Explain the most plausible root cause and what evidence would discriminate alternatives.",
                "evidence": [
                    "503 rate rose only for one upstream after failover",
                    "SBC CPU is normal",
                    "DNS answers differ across two resolvers",
                    "call failures cluster in one region",
                ],
            },
            ("ood", "boundary", "frontier-vs-specialist"),
        ),
        JevBenchmarkCase(
            "ood_frontier_strategy_conflicting_signals",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "request": "Recommend a launch strategy and explain which assumptions matter most.",
                "facts": [
                    "enterprise demand is strong",
                    "support capacity is constrained",
                    "competitor pricing just fell 25%",
                    "regulatory guidance may change next quarter",
                ],
            },
            ("ood", "boundary", "frontier-vs-small"),
        ),
        JevBenchmarkCase(
            "ood_frontier_architecture_partial_failures",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "request": "Design a diagnosis plan for an intermittent distributed-system failure.",
                "symptoms": [
                    "duplicate effects occur only during datastore failover",
                    "request retries are idempotent in unit tests",
                    "queue lag and clock skew are both elevated",
                ],
            },
            ("ood", "boundary", "frontier-vs-code"),
        ),
        JevBenchmarkCase(
            "ood_frontier_conflicting_research",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "request": "Reconcile the conflicting findings and propose the next experiment that best separates the hypotheses.",
                "findings": [
                    "study A shows a large effect in enterprises",
                    "study B finds no effect after controlling for tenure",
                    "study C shows the effect only for new users",
                ],
            },
            ("ood", "boundary", "frontier-vs-specialist"),
        ),
    )
