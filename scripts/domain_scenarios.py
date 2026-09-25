"""Synthetic workloads for scripts/live_domain_sessions.py: six assistant sessions in healthcare, government and
enterprise settings. Every person, identifier, amount and rule here is invented for testing.

Each scenario has a stable policy prompt, a large stable reference module, and four record modules that change at
different rates (never, every 6th turn, every 3rd, every turn). Turn t asks the question tied to module t % 4, so
earlier replies in the history state values that later records replace.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VERIFIED = ("The staff member using this assistant completed identity verification for this session before it "
            "started; the records in this prompt belong to the verified person.")
LEAD = ("Answer from the current records in this prompt. Lead with the requested value, then at most one short "
        "sentence of context.")


@dataclass
class Ask:
    text: str
    module: str
    kind: str                       # "choice", "number", "money" or "date"
    value: Callable[[int], str]     # expected value at a turn
    choices: tuple[str, ...] = ()   # the answer set for "choice"


@dataclass
class Scenario:
    key: str
    sector: str
    title: str
    policy: list[str]
    reference: dict
    modules: dict[str, Callable[[int], dict]]  # name -> record at a turn, in stable-first order
    asks: list[Ask]
    replies: list[str]                          # assistant reply templates, "{v}" is the value, "{f}" filler
    filler: list[str] = field(default_factory=list)

    def system(self) -> str:
        return "\n".join([f"You assist staff in a {self.title}. All records are synthetic test data.", LEAD,
                          VERIFIED, *self.policy])

    def volatile(self) -> set[str]:
        """Record modules that change during a session (every module but the first)."""
        return set(list(self.modules)[1:])

    def question(self, turn: int) -> tuple[Ask, str]:
        ask = self.asks[turn % len(self.asks)]
        return ask, ask.value(turn)

    def reply(self, turn: int, value: str) -> str:
        template = self.replies[turn % len(self.replies)]
        return template.format(v=value, f=self.filler[turn % len(self.filler)])


def cycle(values, every: int, offset: int = 0) -> Callable[[int], str]:
    return lambda t: values[(t // every + offset) % len(values)] if every else values[offset]


def _rng(key: str) -> random.Random:
    return random.Random(key)


def manual(key: str, subjects: list[str], steps: list[str], n: int = 90) -> list[str]:
    """A synthetic procedure manual: n numbered entries, the size of the reference text real deployments carry."""
    rng = _rng(key)
    return [f"{key.upper()}-{i + 1:03d}. For {rng.choice(subjects)}: {rng.choice(steps)}, then "
            f"{rng.choice(steps)}; document the outcome and the time." for i in range(n)]


def log(key: str, turn: int, entries: list[str], n: int = 12) -> list[str]:
    """The n most recent entries of a record's activity log at a turn."""
    rng = _rng(f"{key}-{turn}")
    return [f"{key} log {turn * 3 + k}: {rng.choice(entries)}" for k in range(n)]


# Healthcare: inpatient nurse copilot -------------------------------------------------------------------------------
DRUGS = ["metoprolol", "lisinopril", "furosemide", "apixaban", "atorvastatin", "insulin glargine", "insulin lispro",
         "pantoprazole", "ondansetron", "acetaminophen", "oxycodone", "heparin", "vancomycin", "cefepime",
         "piperacillin-tazobactam", "levetiracetam", "sertraline", "quetiapine", "tamsulosin", "senna", "docusate",
         "magnesium sulfate", "potassium chloride", "enoxaparin", "amlodipine", "hydralazine", "carvedilol",
         "spironolactone", "prednisone", "albuterol", "ipratropium", "montelukast", "gabapentin", "melatonin",
         "famotidine", "polyethylene glycol", "nicotine patch", "thiamine", "folic acid", "multivitamin"]
MONITOR = ["heart rate and blood pressure", "potassium and creatinine", "daily weight and electrolytes",
           "signs of bleeding", "liver enzymes", "fingerstick glucose", "QTc", "sedation score", "trough level",
           "renal function"]


def _formulary() -> dict:
    rng = _rng("formulary")
    return {"unit_formulary": [f"{d}: usual adult range per order set; hold parameters on MAR; monitor "
                               f"{rng.choice(MONITOR)}; high-alert={'yes' if rng.random() < 0.2 else 'no'}"
                               for d in DRUGS],
            "escalation_criteria": [f"Rapid response if {c}" for c in [
                "SBP < 90 or > 200", "HR < 40 or > 130", "RR < 8 or > 28", "SpO2 < 90% on current oxygen",
                "acute change in mental status", "new chest pain", "urine output < 0.5 mL/kg/h for 2 hours",
                "potassium < 3.0 or > 6.0", "glucose < 60 or > 400"]],
            "procedures": manual("nursing", ["central line care", "insulin drips", "telemetry alarms", "fall events",
                                             "blood transfusion", "restraint orders", "pressure injury", "sepsis screen",
                                             "discharge teaching", "pain reassessment", "heparin protocol",
                                             "wound vac", "tube feeding", "isolation precautions"],
                                 ["verify the order in the MAR", "check two patient identifiers",
                                  "reassess within 60 minutes", "notify the charge nurse",
                                  "page the covering provider", "complete the unit checklist",
                                  "review the latest labs", "educate the patient and family"])}


METOPROLOL = ["12.5 mg", "25 mg", "37.5 mg", "50 mg", "75 mg", "100 mg"]
DIETS = ["NPO", "clear liquids", "full liquids", "cardiac diet", "diabetic diet", "renal diet"]
POTASSIUM = [f"{x / 10:.1f}" for x in range(30, 52)]
ALLERGENS = ("penicillin", "sulfa", "latex", "codeine", "iodinated contrast", "peanuts", "aspirin")


def _patient(_t: int) -> dict:
    return {"patient": "SYNTHETIC-A, Maria (test record)", "mrn": "SYN-4471902", "age": 67, "sex": "F",
            "code_status": "Full code", "allergies": ["penicillin (hives)"], "attending": "Dr. Test Attending",
            "admitting_diagnosis": "acute decompensated heart failure",
            "history": ["type 2 diabetes", "hypertension", "atrial fibrillation", "CKD stage 3a", "osteoarthritis",
                        "hyperlipidemia", "former smoker, 30 pack-years"],
            "home_meds": [f"{d} (home)" for d in DRUGS[:14]]}


def _care_plan(t: int) -> dict:
    return {"diet": cycle(DIETS, 6)(t), "activity": ["bed rest", "chair", "ambulate with assist"][(t // 6) % 3],
            "fall_risk": ["high", "moderate"][(t // 6) % 2], "goals": ["diurese 1-2 L/day", "wean oxygen to room air",
                                                                     "glucose 140-180", "PT evaluation"]}


def _mar(t: int) -> dict:
    return {"metoprolol_tartrate": f"{cycle(METOPROLOL, 3)(t)} PO BID, hold for HR < 55 or SBP < 95",
            "scheduled": [f"{d}: see order" for d in DRUGS[14:26]], "prn": [f"{d}: PRN per order" for d in DRUGS[26:34]]}


def _vitals(t: int) -> dict:
    rng = _rng(f"vitals-{t}")
    return {"time": f"{6 + t % 12:02d}:00", "potassium": POTASSIUM[(t * 7) % len(POTASSIUM)],
            "hr": 60 + rng.randint(0, 30), "bp": f"{110 + rng.randint(0, 30)}/{60 + rng.randint(0, 20)}",
            "spo2": f"{92 + rng.randint(0, 6)}%", "glucose": 120 + rng.randint(0, 90), "net_io_ml": -400 - 50 * t,
            "nursing_notes": log("nursing", t, ["patient resting, no distress", "ambulated to chair with assist",
                                                "telemetry sinus rhythm", "pain 3/10 after acetaminophen",
                                                "family at bedside", "tolerating diet", "voided 300 mL",
                                                "lungs with fine crackles at bases"])}


CLINICAL = Scenario(
    "clinical", "healthcare", "hospital cardiac step-down unit (nurse copilot)",
    policy=["Verify patient identity with two identifiers before discussing orders.",
            "Medication questions are answered from the active medication administration record (MAR), not from "
            "earlier conversation.",
            "Report critical lab values to the responsible provider within 30 minutes and document the call.",
            "Hold beta-blockers per the order's hold parameters and notify the provider.",
            "Diet orders come from the current care plan; NPO status overrides any earlier diet.",
            "Allergies are listed in the patient record; confirm before any new antibiotic.",
            "Never recommend a dose change; state the current order and suggest contacting the prescriber.",
            "Use plain language and standard units (mg, mmol/L, mL).",
            "Escalate per the unit escalation criteria in the reference module.",
            "Fall-risk precautions follow the care plan's current fall-risk level."],
    reference=_formulary(),
    modules={"patient": _patient, "care_plan": _care_plan, "mar": _mar, "vitals": _vitals},
    asks=[Ask("Before I hang the cefepime, does she have any drug allergies?", "patient", "choice",
              lambda t: "penicillin", ALLERGENS),
          Ask("What diet is she on right now?", "care_plan", "choice", cycle(DIETS, 6), tuple(DIETS)),
          Ask("What's her current metoprolol dose?", "mar", "choice", cycle(METOPROLOL, 3), tuple(METOPROLOL)),
          Ask("What was her latest potassium?", "vitals", "number",
              lambda t: POTASSIUM[(t * 7) % len(POTASSIUM)])],
    replies=["{v}. {f}", "It's {v}. {f}", "{v}, per the current record. {f}"],
    filler=["Pharmacy verified the order this morning.", "Telemetry shows no new alarms.",
            "Her daughter called for an update.", "PT plans to see her after lunch.", "Strict I&O continues.",
            "Daily weight was charted at 06:00."])


# Healthcare: health plan member services ---------------------------------------------------------------------------
CHANNELS = ("email", "text message", "phone", "postal mail", "member portal")
AUTH = ("pending review", "approved", "denied", "more information requested", "peer-to-peer scheduled")
PLANS = ("Silver PPO", "Gold HMO", "Bronze HSA", "Platinum PPO")


def _benefits() -> dict:
    services = ["primary care visit", "specialist visit", "urgent care", "emergency room", "inpatient stay",
                "outpatient surgery", "MRI/CT", "lab work", "x-ray", "physical therapy", "mental health visit",
                "telehealth", "generic drugs", "preferred brand drugs", "specialty drugs", "durable medical equipment",
                "home health", "skilled nursing", "ambulance", "chiropractic", "vision exam", "hearing aids"]
    rng = _rng("benefits")
    return {"benefit_grid_silver_ppo": [f"{s}: in-network copay ${rng.choice([0, 10, 25, 40, 75, 150, 350])}, "
                                        f"coinsurance {rng.choice([0, 10, 20, 30])}% after deductible, prior auth "
                                        f"{'required' if rng.random() < 0.3 else 'not required'}" for s in services],
            "deductible": "$2,500 individual / $5,000 family", "oop_max": "$7,500 individual / $15,000 family",
            "procedures": manual("member", ["claim disputes", "prior authorization", "provider directory questions",
                                            "ID card requests", "coordination of benefits", "appeals",
                                            "pharmacy exceptions", "billing errors", "out-of-network care",
                                            "newborn enrollment", "COBRA questions", "grievances"],
                                 ["authenticate the caller", "read the current record", "quote the benefit grid",
                                  "open a service request", "warm-transfer to clinical review",
                                  "send a follow-up in the preferred channel", "explain appeal rights",
                                  "log the call reason code"])}


def _member(_t: int) -> dict:
    return {"member": "SYNTHETIC-B, James (test record)", "member_id": "SYN-M-88213407", "plan": "Silver PPO",
            "group": "Test Manufacturing Co.", "pcp": "Test Family Clinic", "dependents": ["spouse", "child (9)"],
            "accumulators": {"deductible_met": "$1,180", "oop_met": "$2,040"}}


def _member_prefs(t: int) -> dict:
    return {"contact_channel": cycle(CHANNELS, 6)(t), "language": "English", "authorized_rep": "spouse"}


def _auth(t: int) -> dict:
    return {"prior_auth_lumbar_mri": cycle(AUTH, 3)(t), "auth_id": "SYN-PA-55120", "requested_by": "Test Ortho Group",
            "criteria": ["6 weeks conservative therapy", "documented radiculopathy", "no red-flag symptoms"]}


def _balance(t: int) -> dict:
    return {"amount_owed": f"${95 + 37 * t:,}.{(t * 13) % 100:02d}", "claims_in_process": 1 + t % 4,
            "last_claim": f"SYN-CL-{700100 + t}",
            "claim_history": log("claims", t, ["lab panel processed, member share applied to deductible",
                                               "office visit paid at in-network rate", "pharmacy claim adjusted",
                                               "urgent care claim pending itemized bill", "EOB issued",
                                               "provider refund requested", "physical therapy visit paid"])}


CLAIMS = Scenario(
    "claims", "healthcare", "health plan member services center (agent copilot)",
    policy=["Authenticate the caller with member ID and date of birth before sharing plan details.",
            "Quote benefits from the benefit grid; never promise coverage for a service still under review.",
            "Prior authorization status comes from the authorization record, not from earlier calls.",
            "Balances and claim status come from the latest claims record.",
            "Use the member's preferred contact channel for follow-ups.",
            "Explain deductibles and coinsurance in plain language.",
            "Grievances and appeals must be offered when a service is denied.",
            "Do not share details with anyone other than the member or an authorized representative."],
    reference=_benefits(),
    modules={"member": _member, "preferences": _member_prefs, "authorization": _auth, "claims": _balance},
    asks=[Ask("Which plan is he enrolled in?", "member", "choice", lambda t: "Silver PPO", PLANS),
          Ask("How does he want us to follow up with him?", "preferences", "choice", cycle(CHANNELS, 6), CHANNELS),
          Ask("Where does the prior auth for his lumbar MRI stand?", "authorization", "choice", cycle(AUTH, 3), AUTH),
          Ask("How much does he owe right now?", "claims", "money", lambda t: _balance(t)["amount_owed"])],
    replies=["{v}. {f}", "{v} is what the record shows. {f}", "Currently {v}. {f}"],
    filler=["He asked whether telehealth counts toward the deductible.", "A claim from the lab is still processing.",
            "His spouse is listed as an authorized representative.", "He mentioned switching pharmacies.",
            "The EOB was mailed last week.", "He asked about the appeal timeline."])


# Government: benefits caseworker copilot ---------------------------------------------------------------------------
STAGES = ("intake", "interview scheduled", "verification", "eligibility review", "approved", "recertification due")


def _program_rules() -> dict:
    return {"program": "Synthetic State Food and Cash Assistance (test rules)",
            "gross_income_limits": [f"household of {n}: ${1580 + 560 * (n - 1):,}/month" for n in range(1, 11)],
            "net_income_limits": [f"household of {n}: ${1215 + 430 * (n - 1):,}/month" for n in range(1, 11)],
            "verification_documents": ["photo ID", "proof of address", "pay stubs (30 days)", "rent or mortgage "
                                       "statement", "utility bill", "childcare receipts", "bank statements",
                                       "immigration documents where applicable", "medical expense receipts (60+)",
                                       "child support orders", "unemployment award letter"],
            "timelines": ["expedited decision within 7 days when eligible", "standard decision within 30 days",
                          "recertify every 12 months", "report changes within 10 days"],
            "procedures": manual("casework", ["expedited screening", "missed interviews", "income verification",
                                              "household composition changes", "work requirements",
                                              "domestic violence waivers", "overpayment claims", "fair hearings",
                                              "language access", "homeless households", "student eligibility",
                                              "self-employment income"],
                                 ["confirm the case number", "check the current case stage", "send a verification "
                                  "checklist", "schedule the interview", "record the contact in case notes",
                                  "offer an interpreter", "explain hearing rights", "escalate to a supervisor"])}


def _household(_t: int) -> dict:
    return {"case": "SYN-CASE-310775 (test record)", "head_of_household": "SYNTHETIC-C, Alex",
            "household_size": "4", "county": "Test County", "members": ["adult", "adult", "child (6)", "child (3)"],
            "income_sources": ["part-time wages", "child support"], "language": "English"}


def _stage(t: int) -> dict:
    return {"case_stage": cycle(STAGES, 6)(t), "assigned_worker": "Test Caseworker", "expedited": False}


def _documents(t: int) -> dict:
    missing = 1 + (t // 3) % 5
    return {"outstanding_documents": missing, "received": ["photo ID", "proof of address"][: 2 - (t // 3) % 2]}


def _appointment(t: int) -> dict:
    day = 1 + (t * 5) % 28
    return {"next_appointment": f"{MONTHS[(9 + t // 6) % 12]} {day}", "type": "phone interview", "time": "10:30",
            "case_notes": log("case", t, ["client called about document list", "reminder letter mailed",
                                          "pay stub received and scanned", "voicemail left for client",
                                          "interview rescheduled at client request", "landlord statement pending",
                                          "childcare receipt received"])}


BENEFITS = Scenario(
    "benefits", "government", "state human services agency (caseworker copilot)",
    policy=["Confirm the case number before discussing a case.",
            "Case stage, documents and appointments come from the current case record, not earlier conversation.",
            "Explain what a client must do next in plain language.",
            "Offer language assistance and reasonable accommodations.",
            "Never share case details with anyone not on the case.",
            "Follow the program rules module for income limits and timelines.",
            "Document every client contact in the case notes."],
    reference=_program_rules(),
    modules={"household": _household, "stage": _stage, "documents": _documents, "appointment": _appointment},
    asks=[Ask("How many people are in this household?", "household", "number", lambda t: "4"),
          Ask("What stage is the case in now?", "stage", "choice", cycle(STAGES, 6), STAGES),
          Ask("How many verification documents are still outstanding?", "documents", "number",
              lambda t: str(1 + (t // 3) % 5)),
          Ask("When is the client's next appointment?", "appointment", "date",
              lambda t: _appointment(t)["next_appointment"])],
    replies=["{v}. {f}", "{v}, according to the case record. {f}", "It's {v}. {f}"],
    filler=["The client asked about childcare deductions.", "A reminder letter went out Monday.",
            "The landlord statement arrived by fax.", "The client prefers morning calls.",
            "Interpreter services were not requested.", "A change report was filed last week."])


# Government: taxpayer assistance -----------------------------------------------------------------------------------
STATUSES = ("single", "married filing jointly", "married filing separately", "head of household")
NOTICES = ("CP14", "CP2000", "CP501", "CP503", "LT11", "CP90")
PLAN_STATES = ("active", "pending review", "in default", "reinstated", "closed")


def _tax_reference() -> dict:
    return {"agency": "Synthetic Revenue Department (test rules)",
            "notice_guide": [f"{n}: see notice guide section {i + 1}" for i, n in enumerate(NOTICES)],
            "payment_plan_rules": ["short-term plan up to 180 days", "long-term plan up to 72 months",
                                   "penalty and interest continue to accrue", "default after a missed payment "
                                   "without contact", "reinstatement fee applies"],
            "brackets": [f"bracket {i}: {10 + 2 * i}% over ${11600 * (i + 1):,}" for i in range(7)],
            "verification": ["prior-year AGI", "date of birth", "last four of taxpayer ID", "current address"],
            "procedures": manual("taxpayer", ["balance due notices", "identity verification", "penalty abatement",
                                              "installment agreements", "levy holds", "refund offsets",
                                              "amended returns", "third-party authorization", "hardship status",
                                              "audit reconsideration", "address changes", "lien releases"],
                                 ["authenticate the caller", "read the current account record",
                                  "explain the notice in plain language", "state the response deadline",
                                  "offer a payment option", "document the contact", "refer to a specialist",
                                  "place a 60-day hold"])}


def _taxpayer(_t: int) -> dict:
    return {"taxpayer": "SYNTHETIC-D, Priya (test record)", "tin_last4": "0000", "filing_status":
            "married filing jointly", "tax_years_open": [2023, 2024, 2025], "address": "100 Test Street, Test City"}


def _notices(t: int) -> dict:
    return {"latest_notice": cycle(NOTICES, 6)(t), "notice_date": f"{MONTHS[(t // 6) % 12]} 12"}


def _plan(t: int) -> dict:
    return {"installment_agreement": cycle(PLAN_STATES, 3)(t), "monthly_payment": "$250"}


def _tax_balance(t: int) -> dict:
    return {"balance_due": f"${4800 - 115 * t:,}.{(t * 29) % 100:02d}", "as_of": "today",
            "transactions": log("account", t, ["payment posted", "penalty assessed", "interest accrued",
                                               "notice issued", "hold placed", "hold released",
                                               "refund offset applied"])}


TAX = Scenario(
    "tax", "government", "tax agency taxpayer assistance line (agent copilot)",
    policy=["Authenticate the caller with the verification items before discussing an account.",
            "Balances, notices and payment plans come from the current account record, not earlier conversation.",
            "Explain notices in plain language and state the response deadline.",
            "Never give legal advice; refer complex cases to a specialist.",
            "Offer a payment plan when the taxpayer cannot pay in full.",
            "Record every contact on the account."],
    reference=_tax_reference(),
    modules={"taxpayer": _taxpayer, "notices": _notices, "payment_plan": _plan, "balance": _tax_balance},
    asks=[Ask("What's her filing status?", "taxpayer", "choice",
              lambda t: "married filing jointly", STATUSES),
          Ask("Which notice did we send her most recently?", "notices", "choice", cycle(NOTICES, 6), NOTICES),
          Ask("What's the status of her installment agreement?", "payment_plan", "choice", cycle(PLAN_STATES, 3),
              PLAN_STATES),
          Ask("What's her balance due right now?", "balance", "money", lambda t: _tax_balance(t)["balance_due"])],
    replies=["{v}. {f}", "The account shows {v}. {f}", "{v} as of today. {f}"],
    filler=["She asked about penalty abatement.", "Her preparer may call on her behalf.",
            "She moved in March.", "A payment posted yesterday.", "She wants a copy of the notice.",
            "She asked about the response deadline."])


# Enterprise: IT helpdesk copilot -----------------------------------------------------------------------------------
LICENSES = ("Standard", "Premium", "Frontline", "Enterprise", "Developer")
COMPLIANCE = ("compliant", "noncompliant", "in grace period", "not evaluated")
DEPARTMENTS = ("Finance", "Legal", "Engineering", "Sales", "Human Resources", "Operations")


def _kb() -> dict:
    return {"knowledge_base": [f"KB-{1000 + i}: {topic}" for i, topic in enumerate([
        "reset multifactor authentication", "reimage a laptop", "request a software license", "VPN will not connect",
        "mailbox is full", "shared drive access request", "printer mapping", "encrypt a laptop disk",
        "conditional access blocked sign-in", "join a device to management", "rotate a service account password",
        "report a phishing email", "restore a deleted file", "Teams-style meeting audio issues",
        "calendar delegation", "mobile device enrollment", "password expiry policy", "guest access request",
        "data loss prevention alert", "browser extension allow-list", "hardware refresh schedule",
        "remote desktop access", "screen sharing blocked", "admin rights request"])],
        "slas": ["P1: 1 hour response", "P2: 4 hours", "P3: 1 business day", "P4: 3 business days"],
        "procedures": manual("servicedesk", ["lost devices", "account lockouts", "new hires", "leavers", "license "
                                             "changes", "security alerts", "noncompliant devices", "VPN outages",
                                             "shared mailboxes", "software requests", "phishing reports",
                                             "privileged access"],
                             ["verify the employee's sign-in name", "check device compliance", "check the license "
                              "record", "link the knowledge base article", "set the incident priority",
                              "escalate to the on-call engineer", "record the action on the incident",
                              "confirm resolution with the employee"])}


def _employee(_t: int) -> dict:
    return {"employee": "SYNTHETIC-E, Sam (test record)", "upn": "sam.test@example.com", "department": "Finance",
            "manager": "Test Manager", "location": "Test Office, floor 4",
            "devices": [f"LAPTOP-SYN-{k:03d}" for k in range(3)] + ["PHONE-SYN-001"]}


def _license(t: int) -> dict:
    return {"license_tier": cycle(LICENSES, 6)(t), "add_ons": ["e-signature", "analytics"]}


def _compliance(t: int) -> dict:
    return {"primary_laptop_compliance": cycle(COMPLIANCE, 3)(t), "last_check_in": "today",
            "failing_policies": ["disk encryption", "OS version"][: (t // 3) % 3]}


def _incidents(t: int) -> dict:
    return {"open_incidents": 1 + (t * 3) % 7, "latest": f"INC-SYN-{55000 + t}",
            "activity": log("incident", t, ["agent added a work note", "employee replied by email",
                                            "priority reviewed", "assigned to endpoint team", "awaiting employee",
                                            "remote session completed", "knowledge article linked"])}


HELPDESK = Scenario(
    "helpdesk", "enterprise", "corporate IT service desk (agent copilot)",
    policy=["Verify the employee with their sign-in name before changing anything.",
            "License, device compliance and incident details come from the current records, not earlier "
            "conversation.",
            "Never ask for a password.", "Link the relevant knowledge base article.",
            "Escalate P1 incidents to the on-call engineer.", "Record every action on the incident."],
    reference=_kb(),
    modules={"employee": _employee, "license": _license, "compliance": _compliance, "incidents": _incidents},
    asks=[Ask("Which department is Sam in?", "employee", "choice", lambda t: "Finance", DEPARTMENTS),
          Ask("What license tier does Sam have now?", "license", "choice", cycle(LICENSES, 6), LICENSES),
          Ask("Is Sam's primary laptop compliant right now?", "compliance", "choice", cycle(COMPLIANCE, 3), COMPLIANCE),
          Ask("How many open incidents does Sam have?", "incidents", "number", lambda t: str(1 + (t * 3) % 7))],
    replies=["{v}. {f}", "Records show {v}. {f}", "{v} at the moment. {f}"],
    filler=["He mentioned the VPN dropped twice yesterday.", "A hardware refresh is scheduled next quarter.",
            "His manager approved the software request.", "The phishing report was closed.",
            "He is travelling next week.", "The printer ticket is waiting on facilities."])


# Enterprise: sales CRM copilot -------------------------------------------------------------------------------------
STAGES_CRM = ("Qualify", "Develop", "Propose", "Negotiate", "Closed Won")
INDUSTRIES = ("Manufacturing", "Retail", "Healthcare", "Financial Services", "Public Sector", "Energy")


def _playbook() -> dict:
    return {"sales_playbook": [f"Stage {s}: exit criteria {i + 1}a, {i + 1}b and {i + 1}c; required fields: "
                               f"budget, decision maker, timeline, competition" for i, s in enumerate(STAGES_CRM)],
            "discount_policy": ["up to 10% rep approval", "10-20% manager approval", "over 20% deal desk"],
            "product_catalog": [f"SKU-SYN-{100 + i}: product line {chr(65 + i % 6)}, tier {i % 3 + 1}"
                                for i in range(30)],
            "procedures": manual("sales", ["discovery calls", "security reviews", "pricing approvals", "pilots",
                                           "renewals", "multi-year deals", "partner-sourced deals", "competitive "
                                           "displacement", "procurement redlines", "executive sponsors",
                                           "forecast calls", "lost-deal reviews"],
                                 ["update the CRM record", "confirm the opportunity stage", "check the discount "
                                  "policy", "log the next step", "loop in the deal desk", "send a mutual action "
                                  "plan", "confirm the decision maker", "review the close date"])}


def _account_rec(_t: int) -> dict:
    return {"account": "Synthetic Test Industries (test record)", "industry": "Manufacturing", "employees": 4200,
            "hq": "Test City", "contacts": [f"Test Contact {k} ({role})" for k, role in
                                            enumerate(["CFO", "CIO", "procurement", "plant manager", "IT director"])]}


def _stage_crm(t: int) -> dict:
    return {"opportunity_stage": cycle(STAGES_CRM, 6)(t), "competitor": "Test Competitor"}


def _deal(t: int) -> dict:
    return {"deal_amount": f"${480 + 35 * (t // 3):,},000", "discount": f"{(t // 3) % 4 * 5}%"}


def _close(t: int) -> dict:
    return {"days_until_close": 90 - 3 * t, "next_step": ["demo", "pricing review", "security review"][t % 3],
            "activity": log("crm", t, ["email to CFO logged", "call with procurement", "demo delivered",
                                       "proposal sent", "security questionnaire returned", "meeting booked",
                                       "champion confirmed budget"])}


CRM = Scenario(
    "crm", "enterprise", "B2B sales team (CRM copilot)",
    policy=["Opportunity stage, amount and dates come from the current CRM record, not earlier conversation.",
            "Follow the discount policy; never quote above the approval level.",
            "Summaries lead with the number the seller asked for.", "Flag missing required fields for the stage.",
            "Keep customer data inside the CRM."],
    reference=_playbook(),
    modules={"account": _account_rec, "stage": _stage_crm, "deal": _deal, "close": _close},
    asks=[Ask("What industry is this account in?", "account", "choice", lambda t: "Manufacturing", INDUSTRIES),
          Ask("What stage is the opportunity in now?", "stage", "choice", cycle(STAGES_CRM, 6), STAGES_CRM),
          Ask("What's the current deal amount?", "deal", "money", lambda t: _deal(t)["deal_amount"]),
          Ask("How many days until the expected close date?", "close", "number", lambda t: str(90 - 3 * t))],
    replies=["{v}. {f}", "The CRM shows {v}. {f}", "{v} right now. {f}"],
    filler=["The CFO asked for a revised proposal.", "Procurement wants net-60 terms.",
            "The security questionnaire is half done.", "A pilot at one plant went well.",
            "The competitor offered a discount.", "The champion is out next week."])

SCENARIOS = {s.key: s for s in (CLINICAL, CLAIMS, BENEFITS, TAX, HELPDESK, CRM)}


# Grading -----------------------------------------------------------------------------------------------------------
NUMBER = re.compile(r"(?<![\w$.,:/-])(\d{1,3}(?:\.\d)?)(?![\w:/]|[.,]\d)")
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")
DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b", re.I)
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
ALIASES = {"text message": ("text", "texts", "SMS"), "member portal": ("portal",), "postal mail": ("mail",),
           "noncompliant": ("non-compliant", "not compliant"),
           "in grace period": ("grace period",), "more information requested": ("more information",
                                                                             "additional information"),
           "peer-to-peer scheduled": ("peer-to-peer",), "pending review": ("pending",),
           "recertification due": ("recertification",)}


def _pattern(value: str) -> str:
    return r"(?<![\w-])" + re.escape(value).replace(r"\ ", r"\s*") + r"(?![\w-])"


def normalize(kind: str, value: str) -> str:
    if kind == "money":
        return f"{float(value.replace('$', '').replace(',', '').strip()):.2f}"
    if kind == "date":
        m = DATE.search(value)
        return f"{m.group(1)[:3].title()} {int(m.group(2))}" if m else value
    return value.lower().strip()


def _first(ask: Ask, text: str) -> str:
    if ask.kind == "money":
        m = MONEY.search(text)
        return normalize("money", m.group(1)) if m else ""
    if ask.kind == "date":
        m = DATE.search(text)
        return normalize("date", m.group(0)) if m else ""
    if ask.kind == "number":
        m = NUMBER.search(text)
        return m.group(1) if m else ""
    hits = [(m.start(), -len(form), v) for v in ask.choices for form in (v, *ALIASES.get(v, ()))
            for m in re.finditer(_pattern(form), text, re.I)]
    return min(hits)[2].lower() if hits else ""


def answer_value(ask: Ask, text: str) -> str:
    """The first value the reply asserts from the question's answer set; a bolded value wins."""
    for span in BOLD.findall(text):
        if value := _first(ask, span):
            return value
    return _first(ask, text)
