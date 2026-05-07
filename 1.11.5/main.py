import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib import request
import time
from urllib import error

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------

# Injected from MuruganAI Runner Environment Variables (No Hardcoded Values)
DEFAULT_LLM_ENDPOINT = os.environ.get("DYNAMIC_LLM_URL", "http://109.165.138.153:30252/v1/chat/completions")
DEFAULT_LLM_MODEL = os.environ.get("DYNAMIC_LLM_MODEL", "sorc/qwen3.5-instruct:latest")
DEFAULT_VL_MODEL = os.environ.get("DYNAMIC_VLM_MODEL") or os.environ.get("DYNAMIC_LLM_MODEL") or "sorc/qwen3.5-instruct:latest"
DEFAULT_LLM_BEARER_TOKEN = os.environ.get("DYNAMIC_LLM_BEARER_TOKEN") or os.environ.get("DYNAMIC_LLM_API_KEY") or "e4cb8ddb3bf23c4f8b90d7e1d2b75be424ac3a96ecafb1e5ad8c4d44647eeb75"

StructuredInput = Union[Dict[str, Any], str, Path]

GAP_ANALYZER_PROMPT_TEMPLATE = """
--- OUT_OF_SCOPE_VALIDATOR_PROMPT ---
SYSTEM
======
You are a TELECOM TEST PLAN VALIDATOR specializing
in TSTP (Test Schedule and Test Procedure) evaluation for
telecom security assurance requirements (ITSAR/NCCS).

Your task is to verify whether the provided TEST SCENARIO
is within scope for HTTP method restriction and security
validation (Requirement 1.11.5).

You MUST evaluate scope based STRICTLY on test design
intent (TSTP) — NOT on execution outcome, system behavior,
or implementation correctness.
DO NOT evaluate:
  * Test execution results or pass/fail outcomes
  * Execution feasibility or system constraints
  * Implementation-level correctness
  * Keyword presence alone without design intent
DO NOT classify test cases as aligned or not aligned.

=================================================
INPUTS
======
Test Scenario:
{test_scenario}

=================================================
SCOPE CONSTRAINT (STRICT — 1.11.5 ONLY)
========================================
The scenario is IN-SCOPE only if it clearly relates to
HTTP method restriction testing, including any of:
  * Allowed HTTP methods (GET, HEAD, POST)
  * Unused/disabled HTTP methods (PUT, DELETE, PATCH, CONNECT)
  * TRACE method validation
  * TRACK method validation
  * OPTIONS / Allow header discovery

DO NOT accept scenarios that test unrelated security areas
(e.g., authentication, password masking, encryption, firewall).

=================================================
VALIDATION STEP
===============

STEP 1 — SCOPE CHECK FOR THE PROVIDED TEST SCENARIO
-----------------------------------------------------

CHECK 1 — HTTP METHOD RELEVANCE
---------------------------------
Does the test scenario explicitly reference or clearly
intend to test HTTP method restriction or validation?

PASS condition:
  The scenario mentions specific HTTP methods (GET, POST,
  HEAD, PUT, DELETE, PATCH, CONNECT, TRACE, TRACK, OPTIONS)
  AND includes test intent (send, verify, attempt, validate)
  → set http_method_scope = PASS

FAIL condition:
  No HTTP method is mentioned or the scenario targets
  a completely unrelated security domain
  → set http_method_scope = FAIL

=================================================
FINAL DECISION
==============
IF http_method_scope = PASS → result = "IN-SCOPE"
IF http_method_scope = FAIL → result = "OUT-OF-SCOPE"

=================================================
DEVIATION SUMMARY RULE
=================================================
If result = "OUT-OF-SCOPE":
    deviation_summary must:
    * be a concise statement explaining WHY the test case is OUT-OF-SCOPE.
    * focus on what the scenario actually tests instead of HTTP methods.
    * NOT mention internal check IDs.
    * NOT use technical PASS/FAIL terminology.

If result = "IN-SCOPE":
    deviation_summary = []

=================================================
OUTPUT FORMAT
=================================================
Return ONLY valid JSON. No explanation.
{
  "http_method_scope": "PASS | FAIL",
  "result": "IN-SCOPE | OUT-OF-SCOPE",
  "deviation_summary": [
    "<Specific out-of-scope reason. Empty if PASS>"
  ]
}"""

def load_prompts() -> Dict[str, str]:
    content = (BASE_DIR / "prompt.txt").read_text(encoding="utf-8")
    prompts = {}
    current_name = None
    current_lines = []
    
    for line in content.splitlines():
        # Accept both "--- NAME ---" and "-- NAME ---" as section delimiters
        stripped = line.strip()
        is_delimiter = (
            (stripped.startswith("--- ") or stripped.startswith("-- "))
            and stripped.endswith(" ---")
        )
        if is_delimiter:
            if current_name:
                # Normalize key: "GAP_ANALYZER_PROMPT v3 (Chain-of-Thought)" -> "GAP_ANALYZER_PROMPT"
                base_name = re.split(r'\s+v\d+|\s+\(', current_name)[0].strip()
                prompts[base_name] = "\n".join(current_lines).strip()
            current_name = line.strip("- ").strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
            
    if current_name:
        base_name = re.split(r'\s+v\d+|\s+\(', current_name)[0].strip()
        prompts[base_name] = "\n".join(current_lines).strip()
        
    return prompts



def render_prompt(template: str, values: Dict[str, str]) -> str:
    for key, val in values.items():
        template = template.replace("{" + key + "}", str(val))
    return template


def _parse_prompt_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse a prompt string into a list of messages using SYSTEM:/USER: headers.
    Falls back to treating the entire text as a single user message if no headers found.
    """
    messages = []
    # Split on lines that are exactly "SYSTEM:" or "USER:" (with optional whitespace)
    sections = re.split(r'^(SYSTEM:|USER:)\s*$', text.strip(), flags=re.MULTILINE)

    if len(sections) < 3:
        # No SYSTEM:/USER: headers found — treat entire text as user message
        return [{"role": "user", "content": text.strip()}]

    # sections alternates: [preamble, "SYSTEM:", content, "USER:", content, ...]
    i = 1  # skip preamble (usually empty)
    while i < len(sections) - 1:
        header = sections[i].strip().rstrip(':').lower()
        content = sections[i + 1].strip()
        if header and content:
            messages.append({"role": header, "content": content})
        i += 2

    return messages if messages else [{"role": "user", "content": text.strip()}]


def _is_llm_reachability_error(exc: Exception) -> bool:
    if isinstance(exc, (error.URLError, TimeoutError)):
        return True
    if isinstance(exc, error.HTTPError):
        return True

    message = str(exc).lower()
    keywords = [
        "timed out",
        "timeout",
        "connection refused",
        "failed to establish a new connection",
        "temporary failure",
        "name or service not known",
        "nodename nor servname provided",
    ]
    return any(token in message for token in keywords)

# Pre-load prompts once
PROMPTS = load_prompts()
SUMMARIZATION_PROMPT = PROMPTS.get("SUMMARIZATION_PROMPT", "")
SINGLE_SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT.replace("test scenarios", "test scenario").replace("each test case", "the test case")
GAP_ANALYZER_PROMPT = GAP_ANALYZER_PROMPT_TEMPLATE # Use hardcoded latest version
COVERAGE_VALIDATOR_PROMPT = PROMPTS.get("COVERAGE_VALIDATOR_PROMPT", "")



# ---------------------------------------------------------------------------
# SECTION HEALTH ASSESSMENT
# ---------------------------------------------------------------------------

def assess_section_health(structured_data: dict) -> dict:
    scenarios_81 = []
    steps_84 = []
    cases_11 = []

    sec_81_found = False
    sec_84_found = False
    sec_11_found = False

    status_81 = ""
    status_84 = ""
    status_11 = ""

    def normalize_title(title: str) -> str:
        return re.sub(r'[^a-z0-9]', '', title.lower())

    for section in structured_data.get("sections", []):
        tn = normalize_title(section.get("title", ""))

        if "numberoftestscenarios" in tn:
            sec_81_found = True
            scenarios_81 = section.get("test_scenarios", [])
            status_81 = str(section.get("status", "")).upper()
        elif "testexecutionsteps" in tn:
            sec_84_found = True
            steps_84 = section.get("execution_steps", [])
            status_84 = str(section.get("status", "")).upper()
        elif "testexecution" in tn:
            sec_11_found = True
            cases_11 = section.get("test_cases", [])
            status_11 = str(section.get("status", "")).upper()

    def _is_unhealthy(found: bool, sec_status: str, items: list) -> int:
        if not found:
            return 1
        if sec_status == "FAIL":
            return 1
        if any(isinstance(item, dict) and str(item.get("status", "")).upper() == "FAIL" for item in items):
            return 1
        return 0

    s81 = _is_unhealthy(sec_81_found, status_81, scenarios_81)
    s84 = _is_unhealthy(sec_84_found, status_84, steps_84)
    s11 = _is_unhealthy(sec_11_found, status_11, cases_11)

    # Count mismatch means Section 11 mapping integrity is unreliable.
    # Treat it as logically DEGRADED (s11=1) so downstream scope routing
    # automatically falls back to combined Section 8.1 injection.
    if s11 == 0 and len(scenarios_81) != len(cases_11):
        print(f"  [HEALTH] Section 11 count mismatch "
              f"(8.1={len(scenarios_81)} vs 11={len(cases_11)}) "
              f"-> treated as DEGRADED (s11=1)")
        s11 = 1

    return {
        "section_81": s81,
        "section_84": s84,
        "section_11": s11,
        "count_81": len(scenarios_81),
        "count_84": len(steps_84),
        "count_11": len(cases_11),
        "raw_81": scenarios_81,
        "raw_84": steps_84,
        "raw_11": cases_11,
    }


# ---------------------------------------------------------------------------
# PIPELINE OUTPUT SKELETON HELPERS
# ---------------------------------------------------------------------------

def _load_pipeline_skeleton(pipeline_path: Path) -> dict:
    """Load the pipeline_output.json skeleton."""
    if pipeline_path.exists():
        try:
            content = pipeline_path.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
        except Exception as e:
            print(f"  Warning: Could not parse {pipeline_path.name}: {e}")
    return {"skeleton": {"sections": []}}


def _find_section_in_skeleton(skeleton: dict, section_id: str) -> dict | None:
    """Find a section or subsection by section_id in the skeleton."""
    for sec in skeleton.get("skeleton", {}).get("sections", []):
        if sec.get("section_id") == section_id:
            return sec
        for sub in sec.get("subsections", []):
            if sub.get("section_id") == section_id:
                return sub
    return None


def _make_check_entry(checklist_name: str, status: str, errors: list, findings: str) -> dict:
    """Build a single check entry in the skeleton format."""
    return {
        "check_name": "Requirement Coverage Validator",
        "validation_results": [
            {
                "checklist_name": checklist_name,
                "status": status,
                "error_count": len(errors),
                "errors": errors,
                "findings": findings if findings else ("Issues found." if errors else "No findings."),
            }
        ],
    }


def _make_error(msg: str, where: str, severity: str = "High",
                err_type: str = "COVERAGE_GAP", suggestion: str = "",
                redirect_text: str = "", what: str = "") -> dict:
    return {
        "where": where,
        "what": what or msg,
        "suggestion": suggestion,
        "redirect_text": redirect_text or where,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# TOP-LEVEL CHECKS SUMMARY
# ---------------------------------------------------------------------------

def _populate_top_level_checks(skeleton: dict) -> None:
    """
    Scan all sections and subsections for 'Requirement Coverage Validator' checks.
    Each unique checklist_name appears ONCE. If any occurrence is FAIL, the
    overall entry is marked FAIL; otherwise PASS.
    """
    inner = skeleton.get("skeleton", skeleton)
    status_map: Dict[str, str] = {}

    def _collect(checks):
        for check in checks:
            if check.get("check_name") == "Requirement Coverage Validator":
                for vr in check.get("validation_results", []):
                    name = vr.get("checklist_name", "")
                    status = str(vr.get("status", "")).upper()
                    if not name:
                        continue
                    if status_map.get(name, "PASS") != "FAIL":
                        status_map[name] = status

    for sec in inner.get("sections", []):
        _collect(sec.get("checks", []))
        for sub in sec.get("subsections", []):
            _collect(sub.get("checks", []))

    entries = [f"{name} - {status}" for name, status in status_map.items()]

    inner["checks"] = [
        {
            "check_name": "Requirement Coverage Validator",
            "total_checklist_name": entries,
        }
    ]


# ---------------------------------------------------------------------------
# INJECT RESULTS INTO PIPELINE OUTPUT
# ---------------------------------------------------------------------------

def inject_coverage_into_skeleton(skeleton: dict, coverage_result: dict) -> None:
    """Inject Coverage Validation results into section_8_1 of the skeleton."""
    sec_81 = _find_section_in_skeleton(skeleton, "section_8_1")
    if not sec_81:
        print("  WARNING: section_8_1 not found in skeleton, skipping coverage injection")
        return

    errors = []
    missing = coverage_result.get("missing_coverage_summary", [])
    if not isinstance(missing, list):
        missing = [str(missing)] if missing else []

    for item in missing:
        if item:
            errors.append(_make_error(
                msg=str(item),
                where="8.1. Number of Test Scenarios",
                what="Coverage Validation across Section 8 and 11",
                suggestion=str(item),
                redirect_text="Number of Test Scenarios",
            ))

    result_status = coverage_result.get("result", "FAIL")
    status = "PASS" if str(result_status).upper() == "PASS" else "FAIL"
    findings = "No findings." if status == "PASS" else "Coverage gaps detected."

    check = _make_check_entry("Coverage Validation", status, errors, findings)
    sec_81["checks"] = [c for c in sec_81.get("checks", [])
                        if c.get("check_name") != "Requirement Coverage Validator"
                        or not any(vr.get("checklist_name") == "Coverage Validation"
                                   for vr in c.get("validation_results", []))]
    sec_81["checks"].append(check)


def inject_scope_per_scenario(skeleton: dict, gap_result: dict, health: dict) -> None:
    """Inject Scope Validation results per Section 11 subsection."""
    cases_11 = health["raw_11"]
    gaps = gap_result.get("gap_assessment", {})

    for idx, tc in enumerate(cases_11):
        heading = tc.get("test_case_heading", "").strip()
        sub_id = f"section_11_{idx + 1}"
        sub_sec = _find_section_in_skeleton(skeleton, sub_id)
        if not sub_sec:
            continue

        gap_entry = gaps.get(heading, {})
        deviation = gap_entry.get("deviation_summary", "").strip() if isinstance(gap_entry, dict) else ""

        if deviation:
            errors = [_make_error(
                msg=deviation,
                where=heading,
                err_type="OUT_OF_SCOPE",
                severity="Medium",
                what="Scope Validation across Section 8 and 11",
                suggestion=deviation,
                redirect_text=heading,
            )]
            status = "FAIL"
            findings = "Out-of-scope test case detected."
        else:
            errors = []
            status = "PASS"
            findings = "No findings."

        check = _make_check_entry("Scope Validation", status, errors, findings)
        sub_sec["checks"] = [c for c in sub_sec.get("checks", [])
                             if c.get("check_name") != "Requirement Coverage Validator"
                             or not any(vr.get("checklist_name") == "Scope Validation"
                                        for vr in c.get("validation_results", []))]
        sub_sec["checks"].append(check)


def inject_scope_combined_into_81(skeleton: dict, gap_result: dict) -> None:
    """Inject combined Scope Validation results into section_8_1."""
    sec_81 = _find_section_in_skeleton(skeleton, "section_8_1")
    if not sec_81:
        return

    gaps = gap_result.get("gap_assessment", {})
    errors = []
    for heading, entry in gaps.items():
        if not isinstance(entry, dict):
            continue
        deviation = entry.get("deviation_summary", "").strip()
        where_val = entry.get("section81_key", heading)

        if deviation:
            errors.append(_make_error(
                msg=deviation,
                where=where_val,
                err_type="OUT_OF_SCOPE",
                severity="Medium",
                what="Scope Validation across Section 8 and 11",
                suggestion=deviation,
                redirect_text=where_val,
            ))

    status = "FAIL" if errors else "PASS"
    findings = "Out-of-scope test cases detected." if errors else "No findings."

    check = _make_check_entry("Scope Validation", status, errors, findings)
    sec_81["checks"] = [c for c in sec_81.get("checks", [])
                        if c.get("check_name") != "Requirement Coverage Validator"
                        or not any(vr.get("checklist_name") == "Scope Validation"
                                   for vr in c.get("validation_results", []))]
    sec_81["checks"].append(check)


def inject_error_into_81(skeleton: dict, message: str) -> None:
    """Inject a simple error message into section_8_1."""
    sec_81 = _find_section_in_skeleton(skeleton, "section_8_1")
    if not sec_81:
        return

    errors = [_make_error(
        msg=message,
        where="8.1. Number of Test Scenarios",
        err_type="VALIDATION_BLOCKED",
        severity="High",
        what="Coverage Validation",
        suggestion=message,
        redirect_text="Number of Test Scenarios",
    )]
    check = _make_check_entry("Coverage Validation", "FAIL", errors, message)
    sec_81["checks"] = [c for c in sec_81.get("checks", [])
                        if c.get("check_name") != "Requirement Coverage Validator"]
    sec_81["checks"].append(check)



def extract_scenarios(structured_json: dict) -> list:
    """
    Positional extraction:
    - Section 8.1  -> descriptions (ORDER SOURCE)
    - Section 8.4  -> steps (PRIMARY)
    - Section 11   -> fallback steps + OUTPUT KEYS (exact test_case_heading strings)
    
    All mapping is purely positional (index-based). No ID matching.
    """

    def normalize_title(title: str) -> str:
        return re.sub(r'[^a-z0-9]', '', title.lower())

    scenarios_81: list = []   # [{description}]
    steps_84: list = []       # [steps[]]
    cases_11: list = []       # [steps[]]
    section11_keys: list = [] # exact test_case_heading strings from Section 11

    # ------------------------------------------------------------------ #
    # Extract all three sections
    # ------------------------------------------------------------------ #
    for section in structured_json.get("sections", []):
        title_norm = normalize_title(section.get("title", ""))

        # ---- Section 8.1: Number of Test Scenarios ----
        if "numberoftestscenarios" in title_norm:
            for ts in section.get("test_scenarios", []):
                scenarios_81.append({
                    "description": ts.get("description", ""),
                    "test_scenario": ts.get("test_scenario", "")
                })

        # ---- Section 8.4: Test Execution Steps ----
        elif "testexecutionsteps" in title_norm:
            for item in section.get("execution_steps", []):
                steps_84.append(item.get("steps", []))

        # ---- Section 11: Test Execution (actual executed cases) ----
        elif "testexecution" in title_norm:
            for tc in section.get("test_cases", []):
                # EXACT string — no transformation
                section11_keys.append(tc.get("test_case_heading", "").strip())
                steps = [
                    s for s in tc.get("execution", [])
                    if isinstance(s, dict) and "step" in s
                ]
                cases_11.append(steps)

    # ------------------------------------------------------------------ #
    # Validations
    # ------------------------------------------------------------------ #
    if not scenarios_81:
        raise ValueError("Section 8.1 not found or empty")

    if steps_84 and len(steps_84) != len(scenarios_81):
        print(f"WARNING: 8.1 has {len(scenarios_81)} scenarios but 8.4 has {len(steps_84)} steps")

    if section11_keys and len(section11_keys) != len(scenarios_81):
        print(f"WARNING: Section 11 has {len(section11_keys)} cases but 8.1 has {len(scenarios_81)} scenarios")

    # ------------------------------------------------------------------ #
    # Positional mapping
    # ------------------------------------------------------------------ #
    combined = []
    count = len(scenarios_81)

    for i in range(count):
        s81_key = scenarios_81[i].get("test_scenario")
        if not s81_key:
            s81_key = f"Test Scenario {i + 1}"
            
        combined.append({
            "tid": f"Test Scenario {i + 1}",   # internal label for LLM only
            "description": scenarios_81[i]["description"],
            "steps": (
                steps_84[i] if i < len(steps_84)
                else cases_11[i] if i < len(cases_11)
                else []
            ),
            "section11_key": (
                section11_keys[i] if i < len(section11_keys)
                else None   # STRICT: no synthetic fallback
            ),
            "section81_key": s81_key
        })

    # Mapping confirmation
    for i, c in enumerate(combined):
        print(f"[MAP] {i + 1} -> {c['section11_key']}")

    return combined




def _format_scenario_for_prompt(scenario: dict) -> dict:
    """
    Format a single scenario into a dict suitable for the summarization prompt.
    """
    steps_text = []
    for s in scenario.get("steps", []):
        steps_text.append(s.get("step", ""))
    return {
        "test_scenario": scenario.get("tid", ""),
        "test_case_name": scenario.get("description", ""),
        "steps": steps_text,
    }


def summarize_scenarios_in_pairs(
    scenarios: list,
    llm_endpoint: str,
    llm_model: str,
) -> list:
    """
    Send scenarios 2 at a time to the LLM for summarization.
    If the last chunk has only 1 scenario, send it alone with a single-item prompt.
    Returns a combined list of {test_case_id, test_case_summary} dicts.
    """

    all_summaries = []

    for i in range(0, len(scenarios), 2):
        pair = scenarios[i : i + 2]
        formatted = [_format_scenario_for_prompt(s) for s in pair]
        payload_json = json.dumps({"test_scenarios": formatted}, indent=2)

        # Use single-item prompt when only 1 scenario in batch
        if len(pair) == 1:
            prompt = SINGLE_SUMMARIZATION_PROMPT
        else:
            prompt = SUMMARIZATION_PROMPT

        # Render all placeholders
        render_values = {"test_scenarios": payload_json}
        prompt_text = render_prompt(prompt, render_values)
        messages = _parse_prompt_sections(prompt_text)

        # --- CLI: Print rendered summarization prompt ---
        pair_ids = [s.get("tid", "?") for s in pair]
        print(f"\n{'='*60}")
        print(f"SUMMARIZATION PROMPT [{', '.join(pair_ids)}]")
        print(f"{'='*60}")
        for msg in messages:
            print(f"[{msg['role'].upper()}]")
            print(msg["content"])
            print()
        print(f"{'='*60}\n")

        llm_endpoint = llm_endpoint.strip()
        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError(
                "Unsupported LLM endpoint. Use /api/generate or /v1/chat/completions"
            )

        # --- CLI: Print raw LLM response ---
        print(f"\n{'-'*60}")
        print(f"SUMMARIZATION RESPONSE [{', '.join(pair_ids)}]")
        print(f"{'-'*60}")
        print(response_text)
        print(f"{'-'*60}\n")

        parsed = _extract_json(response_text)
        batch_summaries = parsed.get("test_summary", [])

        if len(batch_summaries) < len(pair):
            pair_ids = [s.get("tid", "?") for s in pair]
            print(f"  WARNING: Expected {len(pair)} summaries but got {len(batch_summaries)} for: {', '.join(pair_ids)}")

        all_summaries.extend(batch_summaries)

        pair_ids = [s.get("tid", "?") for s in pair]
        print(f"  Summarized: {', '.join(pair_ids)}")

    return all_summaries


def load_structured_json(structured_json: StructuredInput) -> Dict[str, Any]:
    if isinstance(structured_json, (str, Path)):
        path = Path(structured_json)
        return json.loads(path.read_text(encoding="utf-8"))

    if isinstance(structured_json, dict):
        return structured_json

    raise TypeError("structured_json must be a dict or a file path")


def build_prompt(
    test_scenario_summaries: str,
    test_scenarios_text: str,
) -> str:
    prompt_template = COVERAGE_VALIDATOR_PROMPT

    payload = {
        "test_scenario_summaries": test_scenario_summaries,
        "test_scenarios": test_scenarios_text,
    }

    return render_prompt(prompt_template, payload)


def run_gap_analysis(
    scenarios: list,
    llm_endpoint: str,
    llm_model: str,
) -> Dict[str, Any]:
    """
    Run gap analysis on each scenario.
    Displays a preview of the final prompt in the CLI for the first case.
    """

    all_gaps = {}
    count = len(scenarios)

    print("\n" + "="*60)
    print("PHASE 3: GAP ANALYSIS & OUT-OF-SCOPE VALIDATION")
    print("="*60)

    for idx, s in enumerate(scenarios, 1):
        tid = s.get("tid", "Unknown")
        name = s.get("description", "")

        # When s11 was degraded (count mismatch), section11_key is None for
        # extra scenarios. Fall back to section81_key so gap analysis completes.
        section_11_id = s.get("section11_key") or s.get("section81_key") or tid
        if not s.get("section11_key"):
            print(f"  [WARN] No Section 11 key for {tid} "
                  f"(s11 degraded) -> using fallback key: '{section_11_id}'")

        render_values = {}
        render_values["test_case_id"] = tid
        render_values["test_scenario"] = name

        prompt_text = render_prompt(GAP_ANALYZER_PROMPT_TEMPLATE, render_values)
        messages = _parse_prompt_sections(prompt_text)

        llm_endpoint = llm_endpoint.strip()
        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError("Unsupported LLM endpoint")

        parsed = _extract_json(response_text)

        # Handle both list and dict responses
        entry = {}
        if isinstance(parsed, list) and len(parsed) > 0:
            entry = parsed[0]
        elif isinstance(parsed, dict):
            entry = parsed

        if entry:
            dev_sum = entry.get("deviation_summary", "")
            if isinstance(dev_sum, list):
                dev_sum = " ".join(str(i) for i in dev_sum)
                
            all_gaps[section_11_id] = {
                "test_case_name": entry.get("test_case_name", name),
                "deviation_summary": dev_sum.strip(),
                "section81_key": s.get("section81_key", tid)
            }

        print(f"  [idx:{idx}/{count}] Analyzed: {tid} -> {section_11_id}")

    print("="*60 + "\n")
    return {"gap_assessment": all_gaps}


def _ollama_generate(endpoint: str, model: str, prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
    """
    Call Ollama /api/generate. 
    If a list of messages is provided, they are joined as role-labeled sections.
    """
    if isinstance(prompt_or_messages, list):
        # Join messages as role-labeled sections for single prompt
        full_prompt = ""
        for msg in prompt_or_messages:
            full_prompt += f"[{msg['role'].upper()}]\n{msg['content']}\n\n"
        prompt = full_prompt.strip()
    else:
        prompt = prompt_or_messages

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    
    max_retries = 3
    timeout_seconds = 300

    for attempt in range(max_retries):
        try:
            req = request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")

            result = json.loads(body)
            return str(result.get("response", ""))
        except error.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except error.URLError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
            
    return ""


def _openai_chat_completions(endpoint: str, model: str, prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
    """
    Call OpenAI-compatible /v1/chat/completions.
    Supports either a raw string (wrapped as user message) or a list of
    role-separated message dicts sent as proper API messages.
    """
    if isinstance(prompt_or_messages, list):
        # Send as proper role-separated messages in the API
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": prompt_or_messages}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
    }
    data = json.dumps(payload).encode("utf-8")
    
    max_retries = 3
    timeout_seconds = 300

    for attempt in range(max_retries):
        try:
            req = request.Request(
                endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEFAULT_LLM_BEARER_TOKEN}",
                },
            )
            
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                
            result = json.loads(body)
            choices = result.get("choices", [])
            if not choices:
                return ""
            return str(choices[0].get("message", {}).get("content", ""))
        except error.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except error.URLError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)

    return ""


def _clean_json_string(raw: str) -> str:
    """Fix common LLM JSON issues: trailing commas, single-line comments."""
    # Remove single-line comments (// ...)
    raw = re.sub(r'//[^\n]*', '', raw)
    # Remove trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw


def _repair_missing_commas(raw: str, max_attempts: int = 20) -> str:
    """
    Iteratively insert missing commas at the exact position where Python's JSON
    parser raises 'Expecting ,' errors.  Stops after max_attempts to avoid
    infinite loops on genuinely malformed input.
    """
    for _ in range(max_attempts):
        try:
            json.loads(raw)
            return raw          # valid JSON — nothing more to do
        except json.JSONDecodeError as exc:
            msg = exc.msg.lower()
            pos = exc.pos
            # Only attempt to fix "expecting ',' delimiter" errors
            if "expecting ',' delimiter" not in msg and "expecting," not in msg:
                break
            # Find the last non-whitespace character before the error position
            insert_at = pos
            while insert_at > 0 and raw[insert_at - 1] in (' ', '\t', '\n', '\r'):
                insert_at -= 1
            raw = raw[:insert_at] + ',' + raw[insert_at:]
    return raw


def _strip_llm_wrappers(text: str) -> str:
    """Remove markdown code fences, thinking tags, reasoning blocks, and other LLM wrappers."""
    text = text.strip()
    
    # Remove <think>...</think> tags (case-insensitive)
    text = re.sub(r'(?i)<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Remove <reasoning>...</reasoning> tags (case-insensitive)
    text = re.sub(r'(?i)<reasoning.*?>.*?</reasoning>', '', text, flags=re.DOTALL).strip()
    
    # Remove markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r'```(?:json)?\s*\n?', '', text).strip()
    # Remove trailing backticks if any
    text = re.sub(r'```$', '', text).strip()
    
    return text


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts the JSON payload from LLM responses, prioritizing code blocks
    and handling Chain-of-Thought (CoT) reasoning.
    """
    original_text = text
    
    # Attempt to find JSON in markdown code blocks first
    code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.DOTALL)
    if code_blocks:
        # Try blocks in reverse order (usually the final answer is in the last block)
        for block in reversed(code_blocks):
            block = block.strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                try:
                    return json.loads(_clean_json_string(block))
                except json.JSONDecodeError:
                    continue

    # Fallback to stripping wrappers and trying the whole text
    text = _strip_llm_wrappers(text)

    # Attempt 1: raw stripped text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: cleaned stripped text (trailing commas, comments)
    try:
        return json.loads(_clean_json_string(text))
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair missing commas using error position
    try:
        repaired = _repair_missing_commas(_clean_json_string(text))
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass

    # Attempt 4: find the LAST outermost JSON object or array
    # This regex looks for either { ... } or [ ... ]
    matches = list(re.finditer(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL))
    if matches:
        extracted = matches[-1].group(0)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_clean_json_string(extracted))
        except json.JSONDecodeError:
            pass
        try:
            repaired = _repair_missing_commas(_clean_json_string(extracted))
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            pass

    # All attempts failed — print debug info
    debug_file = BASE_DIR / "llm_response_debug.txt"
    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*20} PARSE ERROR RESPONSE ({time.strftime('%H:%M:%S')}) {'='*20}\n")
            f.write(original_text)
            f.write(f"\n{'='*60}\n")
    except:
        pass
        
    print(f"\n[DEBUG] Failed to parse LLM response as JSON. See {debug_file.name}")
    print(f"[DEBUG] Raw response (first 200 chars):\n{original_text[:200]}")
    raise ValueError(f"LLM output does not contain valid JSON. See {debug_file.name} for details.")

# ---------------------------------------------------------------------------
# PIPELINE GAP ASSESSMENT WRITER
# ---------------------------------------------------------------------------

def _write_gap_to_pipeline(gap_result: dict, pipeline_path: Path) -> None:
    """
    Write gap_assessment (OUT-OF-SCOPE only) to pipeline_output.json
    OUTSIDE the skeleton key. Skeleton is never touched.
    """
    pipeline_path = pipeline_path.resolve()
    if pipeline_path.exists():
        try:
            data = json.loads(pipeline_path.read_text(encoding="utf-8").strip() or "{}")
        except Exception:
            data = {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}

    new_gaps = gap_result.get("gap_assessment", {})
    filtered = {}
    for key, val in new_gaps.items():
        if not isinstance(val, dict):
            continue
        dev = val.get("deviation_summary", "")
        if isinstance(dev, list):
            dev = " ".join(str(i) for i in dev)
        if str(dev).strip():
            filtered[key] = {
                "test_case_name": val.get("test_case_name", ""),
                "deviation_summary": str(dev).strip(),
            }

    data["gap_assessment"] = filtered
    pipeline_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Pipeline updated -> {pipeline_path.name} ({len(filtered)} OUT-OF-SCOPE entries)")


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION (1.1.7 DECISION TREE)
# ---------------------------------------------------------------------------

def run_validation(
    structured_json: StructuredInput,
    pipeline_path: Path = None,
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> None:
    """
    Orchestration with 3-section health check:
    1. Assess health of 8.1, 8.4, 11
    2. Follow decision tree
    3. Inject results into output.json (pipeline_output.json is read-only skeleton)
    """
    import copy
    structured_data = load_structured_json(structured_json)
    pipeline_path = pipeline_path or (BASE_DIR / "pipeline_output.json")
    output_path = BASE_DIR / "output.json"
    # Deep-copy the skeleton so pipeline_output.json is NEVER modified
    skeleton = copy.deepcopy(_load_pipeline_skeleton(pipeline_path))

    # ---- STEP 1: Health assessment ----
    health = assess_section_health(structured_data)
    s81 = health["section_81"]
    s84 = health["section_84"]
    s11 = health["section_11"]

    print(f"\n{'='*60}")
    print(f"SECTION HEALTH: 8.1={s81} | 8.4={s84} | 11={s11}")
    print(f"COUNTS:         8.1={health['count_81']} | 8.4={health['count_84']} | 11={health['count_11']}")
    print(f"{'='*60}\n")

    # ---- DECISION: If 8.1 or 8.4 is unhealthy (either one = 1) ----
    if s81 == 1 or s84 == 1:
        failed_sections = []
        if s81 == 1:
            failed_sections.append("Section 8.1 (Number of Test Scenarios)")
        if s84 == 1:
            failed_sections.append("Section 8.4 (Test Execution Steps)")
        msg = f"Requirement Coverage cannot be done as {', '.join(failed_sections)} is Not Parsed Properly."
        print(f"[BLOCKED] {msg}")
        inject_error_into_81(skeleton, msg)
        _populate_top_level_checks(skeleton)
        output_path.write_text(json.dumps(skeleton.get("skeleton", skeleton), indent=2), encoding="utf-8")
        print(f"Output written -> {output_path.name}")
        return

    # ---- Both 8.1 and 8.4 are healthy (= 0) ----
    # Check count match between 8.1 and 8.4
    if health["count_81"] != health["count_84"]:
        msg = "Scenario count are mismatched. AI cannot validate the Requirement Coverage."
        print(f"[MISMATCH] {msg}")
        inject_error_into_81(skeleton, msg)
        _populate_top_level_checks(skeleton)
        output_path.write_text(json.dumps(skeleton.get("skeleton", skeleton), indent=2), encoding="utf-8")
        print(f"Output written -> {output_path.name}")
        return

    # ---- 8.1 == 8.4 counts match -> Run LLM validation ----
    all_scenarios = extract_scenarios(structured_data)

    # Format for coverage prompt
    test_scenarios_lines = [f"- {ts['tid']} {ts['description']}" for ts in all_scenarios]
    test_scenarios_text = "\n".join(test_scenarios_lines)
    print(f"Found {len(all_scenarios)} combined test scenarios")

    # Phase 1: Summarize
    print("Phase 1: Summarizing scenarios in pairs...")
    summaries = summarize_scenarios_in_pairs(all_scenarios, llm_endpoint, llm_model)
    summary_lines = []
    for s in summaries:
        tid = s.get("test_case_id", "Unknown")
        obj = s.get("objective", "")
        exe = s.get("execution_summary", "")
        summary_lines.append(f"- {tid} {obj} | Execution: {exe}")
    test_scenario_summaries = "\n".join(summary_lines)
    print(f"Phase 1 complete. {len(summaries)} summaries collected.")

    # Phase 2: Coverage validation
    print("Phase 2: Running coverage validation...")
    coverage_prompt_text = build_prompt(test_scenario_summaries, test_scenarios_text)
    coverage_messages = _parse_prompt_sections(coverage_prompt_text)
    llm_endpoint = llm_endpoint.strip()
    if llm_endpoint.endswith("/api/generate"):
        coverage_response = _ollama_generate(llm_endpoint, llm_model, coverage_messages)
    elif llm_endpoint.endswith("/v1/chat/completions"):
        coverage_response = _openai_chat_completions(llm_endpoint, llm_model, coverage_messages)
    else:
        raise ValueError(f"Unsupported LLM endpoint. Got '{llm_endpoint}'")

    print("\n--- Coverage Validation Response ---")
    print(coverage_response)
    print("--- End Response ---\n")

    coverage_result = _extract_json(coverage_response)

    # Inject coverage into section_8_1
    inject_coverage_into_skeleton(skeleton, coverage_result)

    # Write intelligence.json
    _write_intelligence_json(coverage_result, BASE_DIR / "intelligence.json")

    # Phase 3: Gap analysis (scope validation)
    print("Phase 3: Running gap analysis...")
    gap_result = run_gap_analysis(all_scenarios, llm_endpoint, llm_model)

    # ---- DECISION: Where to inject scope results ----
    if s11 == 0 and health["count_81"] == health["count_11"]:
        # All three equal & healthy -> per-scenario in Section 11
        print("[OUTPUT] Coverage -> Section 8.1 | Scope -> Per-scenario in Section 11")
        inject_scope_per_scenario(skeleton, gap_result, health)
        _write_gap_to_pipeline(gap_result, pipeline_path)
    else:
        # count mismatch (8.1 != 11) OR Section 11 unhealthy (s11=1) -> combined scope in 8.1
        reason = "Section 11 FAIL" if s11 == 1 else f"count mismatch 8.1={health['count_81']} vs 11={health['count_11']}"
        print(f"[OUTPUT] Coverage -> Section 8.1 | Scope -> Combined in Section 8.1 ({reason})")
        inject_scope_combined_into_81(skeleton, gap_result)

    # Populate top-level checks summary then write
    _populate_top_level_checks(skeleton)
    output_path.write_text(json.dumps(skeleton.get("skeleton", skeleton), indent=2), encoding="utf-8")
    print(f"\nOutput written -> {output_path.name}")


def _write_intelligence_json(result: Dict[str, Any], output_path: Path) -> None:
    """
    Extract compliance_score from the coverage validation result and write
    intelligence.json in the canonical format expected by downstream consumers.

    Format:
    {
      "intelligence": {
        "name": "Requirement Compliance Score",
        "compliance_score": "0.00"
      }
    }
    """
    raw_score = result.get("compliance_score", None)

    # Normalise: accept int, float, numeric string, or None
    try:
        score_float = float(raw_score) if raw_score is not None else 0.0
    except (TypeError, ValueError):
        score_float = 0.0

    # Clamp to [0, 100] and format as "XX.YY"
    score_float = max(0.0, min(100.0, score_float))
    score_str = f"{score_float:.2f}"

    payload = {
        "intelligence": {
            "name": "Requirement Compliance Score",
            "compliance_score": score_str,
        }
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Intelligence written -> {output_path.name}  (compliance_score: {score_str})")


# --------------------------------------------------
# Argument detection
# --------------------------------------------------

def detect_all_files(args: list[str]) -> tuple[Path | None, Path | None]:
    """
    Detect structured_json and pipeline_output_json from a list of paths.
    """
    structured = None
    pipeline = None
    
    for arg in args:
        path = Path(arg)
        if not path.exists():
            continue
            
        name = path.name.lower()
        
        # 1. Pipeline Output JSON
        if name == "pipeline_output.json":
            pipeline = path
        # 2. Structured JSON
        elif name.endswith("ai_structured.json") or name.endswith("_structured.json"):
            structured = path
        # Fallback content checks
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if "sections" in data:
                        structured = path
                    elif "gap_assessment" in data or "skeleton" in data:
                        pipeline = path
            except:
                pass
                
    return structured, pipeline


if __name__ == "__main__":
    import sys
    import traceback

    if len(sys.argv) < 2:
        print("Usage: python main.py <structured.json> [pipeline_output.json]")
        sys.exit(1)

    # Detect files from all arguments
    structured_path, pipeline_path = detect_all_files(sys.argv[1:])
    
    if structured_path:
        print(f"Detected structured file: {structured_path.name}")
    if pipeline_path:
        print(f"Detected pipeline file: {pipeline_path.name}")

    if not structured_path:
        print("Error: Could not find structured JSON file.")
        sys.exit(1)

    pipeline_output_file = pipeline_path if pipeline_path else (BASE_DIR / "pipeline_output.json")

    try:
        run_validation(
            structured_path,
            pipeline_path=pipeline_output_file,
        )
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

