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
DEFAULT_LLM_ENDPOINT = os.environ.get("DYNAMIC_LLM_URL") or "http://96.28.88.208:40201/v1/chat/completions"
DEFAULT_LLM_MODEL = os.environ.get("DYNAMIC_LLM_MODEL") or "sorc/qwen3.5-instruct:latest"
DEFAULT_VL_MODEL = os.environ.get("DYNAMIC_VLM_MODEL") or os.environ.get("DYNAMIC_LLM_MODEL") or "sorc/qwen3.5-instruct:latest"
DEFAULT_LLM_BEARER_TOKEN = os.environ.get("DYNAMIC_LLM_BEARER_TOKEN") or os.environ.get("DYNAMIC_LLM_API_KEY") or "99203031b9516a1f36c85d6f9232f8420bb059dcd92c580653304121c72754f0"
StructuredInput = Union[Dict[str, Any], str, Path]


GAP_ANALYZER_PROMPT_TEMPLATE = """
--- OUT_OF_SCOPE_VALIDATOR_PROMPT ---
SYSTEM
======
You are a TELECOM TEST PLAN SCOPE VALIDATOR for requirement 1.2.6:

  "Password Change facility, 1st Installation / Factory Reset"

Determine whether the given TEST CASE NAME is
IN-SCOPE or OUT-OF-SCOPE.

Evaluate ONLY the Test Case Name.

Do NOT use:
* execution results
* logs
* observations
* assumptions
* inferred meaning
* implementation behavior

=================================================
INPUTS
======

Access Methods: {access_methods}

Controller Support: {controller_support}

Controller Name: {auth_controller_name}

Test Case ID: {test_case_id}

Test Case Name: {test_case_name}

=================================================
INTENT EVALUATION
=================================================

C1 — Password Change Enforcement During Initial Setup (MANDATORY)
Intent:
Ensure the test case explicitly validates forced password
change during first installation configuration or initial login.

A test case satisfies this intent ONLY if it explicitly verifies:
- forced password change during first login
- initial setup password modification
- mandatory password update during onboarding

Do NOT consider:
- normal password change
- password expiry
- password reset only
- password complexity

If the PRIMARY purpose of the test case is initial setup
password enforcement:
    password_change_enforcement = PASS
else:
    password_change_enforcement = FAIL

-------------------------------------------------

C2 — Password Change Enforcement After Factory Reset (MANDATORY)
Intent:
Ensure the test case explicitly validates forced password
change after factory reset condition.

A test case satisfies this intent ONLY if it explicitly verifies:
- password change enforcement after reset
- mandatory password update after factory reset
- first login password change following reset

Do NOT consider:
- reboot validation
- configuration restore
- reset without password enforcement
- password recovery testing

If the PRIMARY purpose of the test case is factory reset
password enforcement:
    password_change_factory_reset = PASS
else:
    password_change_factory_reset = FAIL

-------------------------------------------------

C3 — User Password Change Functionality (MANDATORY)
Intent:
Ensure the test case explicitly validates that user/admin
can change password or authentication attribute.

A test case satisfies this intent ONLY if it explicitly verifies:
- user password modification
- admin password change
- password update functionality
- authentication attribute change

Do NOT consider:
- password expiry
- account lockout
- authentication failure
- login testing only

If the PRIMARY purpose of the test case is password change
functionality:
    password_change = PASS
else:
    password_change = FAIL

-------------------------------------------------

C4 — Password History Restriction (MANDATORY)
Intent:
Ensure the test case explicitly validates restriction on
reuse of previously used passwords.

A test case satisfies this intent ONLY if it explicitly verifies:
- old password reuse restriction
- password history enforcement
- rejection of previously used passwords

Do NOT consider:
- password complexity
- password strength
- password expiry
- password storage validation

If the PRIMARY purpose of the test case is password history
restriction:
    password_history_restriction = PASS
else:
    password_history_restriction = FAIL

-------------------------------------------------

C5 — Controller Password Change Functionality (MANDATORY / NA)

APPLICABILITY CHECK:
  If <<CONTROLLER_SUPPORT>> = NO:
      applicability = NA
      result = NA
      Skip evaluation.

  If <<CONTROLLER_SUPPORT>> = YES:
      applicability = MANDATORY
      Proceed with evaluation below.

Intent:
Ensure the test case explicitly validates password
change functionality through controller management.

A test case satisfies this intent ONLY if it explicitly verifies:
- password change through controller
- controller-managed password modification
- password update using centralized controller

Do NOT consider:
- AP local password change only
- controller connectivity
- controller onboarding only

If the PRIMARY purpose of the test case is controller-based
password change:
    controller_password_change = PASS
else:
    controller_password_change = FAIL

-------------------------------------------------

C6 — Controller Password History Restriction (MANDATORY / NA)

APPLICABILITY CHECK:
  If <<CONTROLLER_SUPPORT>> = NO:
      applicability = NA
      result = NA
      Skip evaluation.

  If <<CONTROLLER_SUPPORT>> = YES:
      applicability = MANDATORY
      Proceed with evaluation below.

Intent:
Ensure the test case explicitly validates password history
restriction through controller management.

A test case satisfies this intent ONLY if it explicitly verifies:
- password history enforcement through controller
- old password rejection through controller
- centralized password history restriction

Do NOT consider:
- AP-only password history validation
- controller synchronization
- password complexity validation

If the PRIMARY purpose of the test case is controller-based
password history restriction:
    controller_password_history_restriction = PASS
else:
    controller_password_history_restriction = FAIL

=================================================
ACCESS METHOD VALIDATION
=================================================

This validation applies ONLY to C1, C2, C3, and C4.
C5 and C6 are NOT affected by access method scope.

Evaluate ONLY the Test Case Name.

Do NOT:
* infer hidden interfaces
* normalize interface names
* assume equivalent protocols
* infer DUT-wide applicability

-------------------------------------------------
DECISION LOGIC
-------------------------------------------------

CASE 1:
If the test case name does NOT explicitly mention
any authentication interface or access method:
    access_method_scope = PASS

CASE 2:
If the test case name explicitly mentions one or more
interfaces or access methods:
  * Extract ALL referenced interfaces from the test case name.
  * Compare STRICTLY against the declared access methods
    in <<ACCESS_METHODS>>.

  If ALL referenced interfaces exist in <<ACCESS_METHODS>>:
      access_method_scope = PASS
  Else:
      access_method_scope = FAIL

-------------------------------------------------
STRICT MATCHING RULES
-------------------------------------------------

1. Matching MUST be exact semantic interface matching.
2. Do NOT infer:
   * CLI = SSH
   * GUI = WebUI
   * Console = CLI
   * Controller = GUI
3. If any referenced interface is absent from <<ACCESS_METHODS>>:
      access_method_scope = FAIL
4. Generic wording such as "authentication", "management access",
   or "login interface" without explicit interface naming:
      access_method_scope = PASS

=================================================
INTENT MATCHING RULES (CRITICAL)
=================================

1. A test case satisfies an intent ONLY if its PRIMARY
   purpose directly matches that intent.
2. Do NOT infer intent from related words.
3. Do NOT map one scenario to multiple intents unless
   each intent is explicitly validated.
4. The following are INVALID mappings:
   * password expiry ≠ password change functionality
   * password complexity ≠ password history restriction
   * reset validation ≠ password enforcement after reset
   * login validation ≠ password change enforcement
   * controller onboarding ≠ controller password management
5. If intent is unclear, partial, or indirect → mark FAIL
6. Evaluate C1, C2, C3, and C4 on their own intent criteria
   independently. Do NOT modify their results based on
   access_method_scope.

=================================================
SCOPE DECISION RULE
===================

Step 1 — Access Method Gate:
  If access_method_scope = FAIL:
      final_scope_result = OUT-OF-SCOPE
      Stop. Do not proceed to Step 2.

Step 2 — Intent Gate:
  Collect results for all APPLICABLE intents only
  (exclude NA intents from this check).

  If at least ONE applicable intent result = PASS:
      final_scope_result = IN-SCOPE
  Else:
      final_scope_result = OUT-OF-SCOPE

=================================================
DEVIATION SUMMARY RULE
======================

IF final_scope_result = IN-SCOPE:
    deviation_summary = ""

IF final_scope_result = OUT-OF-SCOPE due to access_method_scope = FAIL:
    Write 1–2 concise telecom audit-style lines:
      * What interface the test case references
      * Why it falls outside the declared access methods
        for requirement 1.2.6

IF final_scope_result = OUT-OF-SCOPE due to all intents failing:
    Write 1–2 concise telecom audit-style lines:
      * What the test case validates
      * Why it does not align with requirement 1.2.6

Do NOT mention:
* internal intent names (C1, C2, etc.)
* validator logic
* PASS/FAIL reasoning

=================================================
OUTPUT FORMAT (STRICT JSON — RETURN ONLY JSON)
==============================================

{
  "test_case_id": "<value of <<TEST_CASE_ID>>>",
  "test_case_name": "<value of <<TEST_CASE_NAME>>>",

  "access_method_scope": {
    "result": "<PASS or FAIL>"
  },

  "access_method_bound_intents": [
    {
      "password_change_enforcement": {
        "applicability": "MANDATORY",
        "result": "<PASS or FAIL>"
      }
    },
    {
      "password_change_factory_reset": {
        "applicability": "MANDATORY",
        "result": "<PASS or FAIL>"
      }
    },
    {
      "password_change": {
        "applicability": "MANDATORY",
        "result": "<PASS or FAIL>"
      }
    },
    {
      "password_history_restriction": {
        "applicability": "MANDATORY",
        "result": "<PASS or FAIL>"
      }
    }
  ],

  "controller_intents": [
    {
      "controller_password_change": {
        "applicability": "<MANDATORY or NA>",
        "result": "<PASS or FAIL or NA>"
      }
    },
    {
      "controller_password_history_restriction": {
        "applicability": "<MANDATORY or NA>",
        "result": "<PASS or FAIL or NA>"
      }
    }
  ],

  "final_scope_result": "<IN-SCOPE or OUT-OF-SCOPE>",
  "deviation_summary": "<empty string if IN-SCOPE, audit summary if OUT-OF-SCOPE>"
}

CRITICAL:
* Return ONLY the JSON object above.
* Do NOT include explanation, commentary, or markdown outside the JSON.
* Do NOT skip any intent field in either block.
* access_method_bound_intents — C1 to C4: evaluated independently
  on intent merit; access_method_scope feeds the scope decision,
  NOT these results.
* controller_intents — C5 and C6: NOT affected by access_method_scope.
* Resolve all values before outputting — do NOT echo placeholder names.
"""

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


def load_answers(answers_path: Path) -> Dict[str, Any]:
    return json.loads(answers_path.read_text(encoding="utf-8"))


def render_prompt(template: str, values: Dict[str, str]) -> str:
    for key, val in values.items():
        template = template.replace("{" + key + "}", str(val))
        # Also substitute <<KEY>> style placeholders used by GAP_ANALYZER_PROMPT_TEMPLATE
        template = template.replace("<<" + key.upper() + ">>", str(val))
    return template


def _parse_chatml(text: str) -> List[Dict[str, str]]:
    """
    Parse a string containing ChatML tags into a list of messages.
    Supports: <|im_start|>role\ncontent<|im_end|>
    Also supports content until the next <|im_start|> or end of text.
    """
    messages = []
    # Pattern to find structured ChatML blocks
    # Role is immediately after start tag, then everything until end of that line
    # Then content until the next im_start, im_end, or end of string
    pattern = r"<\|im_start\|>(\w+).*?\n(.*?)(?=<\|im_start\|>|<\|im_end\|>|$)"
    
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        # Fallback if no tags found: treat everything as a user message
        return [{"role": "user", "content": text.strip()}]
    
    for match in matches:
        role = match.group(1).strip()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})
        
    return messages


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "true", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def ensure_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(answers)

    methods = _normalize_list(answers.get("access_methods"))

    # Deduplicate while preserving order.
    deduped_methods: list[str] = []
    seen_methods = set()
    for method in methods:
        key = method.lower()
        if key not in seen_methods:
            deduped_methods.append(method)
            seen_methods.add(key)

    normalized["access_methods"] = deduped_methods

    # Normalize controller_support -> bool
    normalized["controller_support"] = _normalize_bool(answers.get("controller_support", False))

    # Store auth_controller_name as string
    normalized["auth_controller_name"] = str(answers.get("auth_controller_name", "")).strip()

    return normalized


def load_questions(questions_path: Path) -> Dict[str, Any]:
    if not questions_path.exists():
        return {"questions": []}
    return json.loads(questions_path.read_text(encoding="utf-8"))


def validate_answers_against_questions(
    answers: Dict[str, Any], questions_data: Dict[str, Any]
) -> List[str]:
    missing_messages: List[str] = []
    questions = questions_data.get("questions", [])

    for q in questions:
        qid = q.get("id")
        label = q.get("label", qid)
        required = bool(q.get("required", False))
        show_if = q.get("show_if", {})

        # Respect conditional questions (show_if).
        should_validate = True
        if isinstance(show_if, dict):
            for key, expected in show_if.items():
                if answers.get(key) != expected:
                    should_validate = False
                    break

        if not required or not should_validate:
            continue

        value = answers.get(qid)
        is_missing = False

        if value is None:
            is_missing = True
        elif isinstance(value, str) and not value.strip():
            is_missing = True
        elif isinstance(value, list) and len([v for v in value if str(v).strip()]) == 0:
            is_missing = True

        if is_missing:
            missing_messages.append(f'"{label}" is not mentioned in the input.')

    return missing_messages


# Pre-load prompts once
PROMPTS = load_prompts()
SUMMARIZATION_PROMPT = PROMPTS.get("SUMMARIZATION_PROMPT", "")
SINGLE_SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT.replace("test scenarios", "test scenario").replace("each test case", "the test case")
GAP_ANALYZER_PROMPT = GAP_ANALYZER_PROMPT_TEMPLATE # Use hardcoded latest version
COVERAGE_VALIDATOR_PROMPT = PROMPTS.get("COVERAGE_VALIDATOR_PROMPT", "")


def _normalize_title(title: str) -> str:
    return re.sub(r'[^a-z0-9]', '', title.lower())


# ---------------------------------------------------------------------------
# SECTION HEALTH ASSESSMENT
# ---------------------------------------------------------------------------

def assess_section_health(structured_data: dict) -> dict:
    """
    Evaluate 3 sections from AI structured JSON.
    Returns dict with keys: section_81, section_84, section_11
    Each value is 0 (healthy) or 1 (unhealthy / has FAIL status).
    Also returns raw lists for downstream use.
    """
    scenarios_81 = []
    steps_84 = []
    cases_11 = []

    sec_81_found = False
    sec_84_found = False
    sec_11_found = False

    status_81 = ""
    status_84 = ""
    status_11 = ""

    for section in structured_data.get("sections", []):
        tn = _normalize_title(section.get("title", ""))

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
        # If the section wasn't even in the JSON, it's missing (unhealthy)
        if not found:
            return 1
        # If the section explicitly says FAIL
        if sec_status == "FAIL":
            return 1
        # If any item inside explicitly says FAIL
        if any(isinstance(item, dict) and str(item.get("status", "")).upper() == "FAIL" for item in items):
            return 1
        # Otherwise, it's healthy (0). Even if items are empty, it's valid (e.g. 0 scenarios).
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
# EXTRACT & COMBINE FOR LLM
# ---------------------------------------------------------------------------

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

    for section in structured_json.get("sections", []):
        title_norm = normalize_title(section.get("title", ""))

        if "numberoftestscenarios" in title_norm:
            for ts in section.get("test_scenarios", []):
                scenarios_81.append({
                    "description": ts.get("description", ""),
                    "test_scenario": ts.get("test_scenario", "")
                })

        elif "testexecutionsteps" in title_norm:
            for item in section.get("execution_steps", []):
                steps_84.append(item.get("steps", []))

        elif "testexecution" in title_norm:
            for tc in section.get("test_cases", []):
                section11_keys.append(tc.get("test_case_heading", "").strip())
                steps = [
                    s for s in tc.get("execution", [])
                    if isinstance(s, dict) and "step" in s
                ]
                cases_11.append(steps)

    if not scenarios_81:
        raise ValueError("Section 8.1 not found or empty")

    if steps_84 and len(steps_84) != len(scenarios_81):
        print(f"WARNING: 8.1 has {len(scenarios_81)} scenarios but 8.4 has {len(steps_84)} steps")

    if section11_keys and len(section11_keys) != len(scenarios_81):
        print(f"WARNING: Section 11 has {len(section11_keys)} cases but 8.1 has {len(scenarios_81)} scenarios")

    combined = []
    count = len(scenarios_81)

    for i in range(count):
        s81_key = scenarios_81[i].get("test_scenario")
        if not s81_key:
            s81_key = f"Test Scenario {i + 1}"
            
        combined.append({
            "tid": f"Test Scenario {i + 1}",
            "description": scenarios_81[i]["description"],
            "steps": (
                steps_84[i] if i < len(steps_84)
                else cases_11[i] if i < len(cases_11)
                else []
            ),
            "section11_key": (
                section11_keys[i] if i < len(section11_keys)
                else None
            ),
            "section81_key": s81_key
        })

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
    user_answers: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
) -> list:
    """
    Send scenarios 2 at a time to the LLM for summarization.
    If the last chunk has only 1 scenario, send it alone with a single-item prompt.
    Returns a combined list of {test_case_id, test_case_summary} dicts.
    """
    methods = ", ".join(user_answers.get("access_methods", []))
    answer_values = {
        "access_methods": methods,
    }

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
        render_values = dict(answer_values)
        render_values["test_scenarios"] = payload_json
        prompt_text = render_prompt(prompt, render_values)
        messages = _parse_chatml(prompt_text)

        llm_endpoint = llm_endpoint.strip()
        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError(f"Unsupported LLM endpoint. Got '{llm_endpoint}'")

        pair_ids = [s.get("tid", "?") for s in pair]

        # Print raw LLM response for this batch
        print(f"\n--- Summarization Response: {', '.join(pair_ids)} ---")
        print(response_text)
        print(f"--- End Response ---\n")

        parsed = _extract_json(response_text)
        batch_summaries = parsed.get("test_summary", [])

        if len(batch_summaries) < len(pair):
            print(f"  WARNING: Expected {len(pair)} summaries but got {len(batch_summaries)} for: {', '.join(pair_ids)}")

        all_summaries.extend(batch_summaries)

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
    user_answers: Dict[str, str],
    test_scenario_summaries: str,
    test_scenarios_text: str,
) -> str:
    prompt_template = COVERAGE_VALIDATOR_PROMPT

    methods_value = user_answers.get("access_methods", "")
    if isinstance(methods_value, list):
        methods = ",".join(
            str(item).strip() for item in methods_value if str(item).strip()
        )
    else:
        methods = str(methods_value).strip()

    controller_support = user_answers.get("controller_support", False)
    controller_support_str = "YES" if controller_support else "NO"

    auth_controller_name = str(user_answers.get("auth_controller_name", "")).strip()
    auth_controller_name_str = auth_controller_name if auth_controller_name else "N/A"

    # Strict coverage payload: only the fields defined in answer.json for coverage context.
    payload = {
        "test_scenario_summaries": test_scenario_summaries,
        "test_scenarios": test_scenarios_text,
        "access_methods": methods,
        "controller_support": controller_support_str,
        "auth_controller_name": auth_controller_name_str,
    }

    return render_prompt(prompt_template, payload)


def run_gap_analysis(
    scenarios: list,
    user_answers: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
) -> Dict[str, Any]:
    """
    Run gap analysis on each scenario.
    Populates <<ACCESS_METHODS>>, <<CONTROLLER_SUPPORT>>, <<CONTROLLER_NAME>>,
    <<TEST_CASE_ID>>, and <<TEST_CASE_NAME>> from answer.json and the scenario.
    """
    methods = user_answers.get("access_methods", [])
    if isinstance(methods, list):
        methods_str = ", ".join(methods)
    else:
        methods_str = str(methods).strip()

    controller_support = user_answers.get("controller_support", False)
    controller_support_str = "YES" if controller_support else "NO"

    auth_controller_name = str(user_answers.get("auth_controller_name", "")).strip()
    auth_controller_name_str = auth_controller_name if auth_controller_name else "N/A"

    base_payload = {
        "access_methods": methods_str,
        "controller_support": controller_support_str,
        "auth_controller_name": auth_controller_name_str,
    }

    all_gaps = {}
    count = len(scenarios)

    print("\n" + "="*60)
    print("PHASE 3: GAP ANALYSIS & OUT-OF-SCOPE VALIDATION")
    print("="*60)
    print(f"  Access Methods     : {methods_str}")
    print(f"  Controller Support : {controller_support_str}")
    print(f"  Controller Name    : {auth_controller_name_str}")

    for idx, s in enumerate(scenarios, 1):
        tid = s.get("tid", "Unknown")
        name = s.get("description", "")

        # When s11 was degraded (count mismatch), section11_key is None for
        # extra scenarios. Fall back to section81_key so gap analysis completes.
        section_11_id = s.get("section11_key") or s.get("section81_key") or tid
        if not s.get("section11_key"):
            print(f"  [WARN] No Section 11 key for {tid} "
                  f"(s11 degraded) -> using fallback key: '{section_11_id}'")

        render_values = dict(base_payload)
        # <<TEST_CASE_ID>> and <<TEST_CASE_NAME>> placeholders in GAP_ANALYZER_PROMPT_TEMPLATE
        render_values["test_case_id"] = tid
        render_values["test_case_name"] = name

        prompt_text = render_prompt(GAP_ANALYZER_PROMPT_TEMPLATE, render_values)
        messages = _parse_chatml(prompt_text)

        llm_endpoint = llm_endpoint.strip()
        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError(f"Unsupported LLM endpoint. Got: '{llm_endpoint}'")

        print(f"\n--- Gap Analysis Response: {tid} ---")
        print(response_text)
        print("--- End Response ---\n")

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
    If a list of messages is provided, they are joined with ChatML tags.
    """
    if isinstance(prompt_or_messages, list):
        # Join messages with ChatML tags to pass as single prompt to /api/generate
        full_prompt = ""
        for msg in prompt_or_messages:
            full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt = full_prompt.strip()
    else:
        prompt = prompt_or_messages

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9, # Note: Ollama top_p is usually 0.0-1.0
            "num_predict": 512,
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

    If the last message has role "assistant" (a prefill/primer, e.g. "{"),
    it is prepended back to the model response since the API does not echo it.
    """
    if isinstance(prompt_or_messages, list):
        # Send as proper role-separated messages in the API
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": prompt_or_messages}]

    # Detect assistant prefill: the model continues from this prefix
    # but does NOT include it in the returned content — prepend it back.
    assistant_prefix = ""
    if messages and messages[-1].get("role") == "assistant":
        assistant_prefix = messages[-1].get("content", "")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 11000,
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
            content = str(choices[0].get("message", {}).get("content", ""))
            # Reconstruct full response when an assistant prefix was used
            if assistant_prefix and not content.lstrip().startswith(assistant_prefix.strip()):
                content = assistant_prefix + content
            return content
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
      [{"check_name": "Requirement Coverage Validator",
        "total_checklist_name": ["Coverage Validation - PASS",
                                  "Scope Validation - FAIL"]}]
    """
    inner = skeleton.get("skeleton", skeleton)
    # Track: checklist_name -> worst status (FAIL beats PASS)
    status_map: Dict[str, str] = {}

    def _collect(checks):
        for check in checks:
            if check.get("check_name") == "Requirement Coverage Validator":
                for vr in check.get("validation_results", []):
                    name = vr.get("checklist_name", "")
                    status = str(vr.get("status", "")).upper()
                    if not name:
                        continue
                    # FAIL wins over PASS
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
        if item:  # skip empty strings (PASS case)
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
        # Fallback to the heading (from Section 11) if section81_key is somehow missing
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
# MAIN ORCHESTRATION (NEW DECISION TREE)
# ---------------------------------------------------------------------------

def run_validation(
    structured_json: StructuredInput,
    user_answers: Dict[str, str],
    pipeline_path: Path = None,
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> None:
    """
    New orchestration with 3-section health check:
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
        print(f"Output written → {output_path.name}")
        return

    # ---- Both 8.1 and 8.4 are healthy (= 0) ----
    # Check count match between 8.1 and 8.4
    if health["count_81"] != health["count_84"]:
        msg = "Scenario count are mismatched. AI cannot validate the Requirement Coverage."
        print(f"[MISMATCH] {msg}")
        inject_error_into_81(skeleton, msg)
        _populate_top_level_checks(skeleton)
        output_path.write_text(json.dumps(skeleton.get("skeleton", skeleton), indent=2), encoding="utf-8")
        print(f"Output written → {output_path.name}")
        return

    # ---- 8.1 == 8.4 counts match → Run LLM validation ----
    all_scenarios = extract_scenarios(structured_data)

    # Format for coverage prompt
    test_scenarios_lines = [f"- {ts['tid']} {ts['description']}" for ts in all_scenarios]
    test_scenarios_text = "\n".join(test_scenarios_lines)
    print(f"Found {len(all_scenarios)} combined test scenarios")

    # Phase 1: Summarize
    print("Phase 1: Summarizing scenarios in pairs...")
    summaries = summarize_scenarios_in_pairs(all_scenarios, user_answers, llm_endpoint, llm_model)
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
    coverage_prompt_text = build_prompt(user_answers, test_scenario_summaries, test_scenarios_text)
    coverage_messages = _parse_chatml(coverage_prompt_text)
    llm_endpoint = llm_endpoint.strip()
    if llm_endpoint.endswith("/api/generate"):
        coverage_response = _ollama_generate(llm_endpoint, llm_model, coverage_messages)
    elif llm_endpoint.endswith("/v1/chat/completions"):
        coverage_response = _openai_chat_completions(llm_endpoint, llm_model, coverage_messages)
    else:
        raise ValueError(f"Unsupported LLM endpoint. Got: '{llm_endpoint}'")

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
    gap_result = run_gap_analysis(all_scenarios, user_answers, llm_endpoint, llm_model)

    # ---- DECISION: Where to inject scope results ----
    if s11 == 0 and health["count_81"] == health["count_11"]:
        # All three equal & healthy -> per-scenario in Section 11
        print("[OUTPUT] Coverage -> Section 8.1 | Scope -> Per-scenario in Section 11")
        inject_scope_per_scenario(skeleton, gap_result, health)
        _write_gap_to_pipeline(gap_result, pipeline_path)
    elif s11 == 0 and health["count_81"] != health["count_11"]:
        # 8.1==8.4 but !=11 -> combined scope in 8.1
        print("[OUTPUT] Coverage -> Section 8.1 | Scope -> Combined in Section 8.1")
        inject_scope_combined_into_81(skeleton, gap_result)
    elif s11 == 1:
        # Section 11 unhealthy -> combined scope in 8.1
        print("[OUTPUT] Coverage -> Section 8.1 | Scope -> Combined in Section 8.1 (Section 11 FAIL)")
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

def detect_all_files(args: list[str]) -> tuple[Path | None, Path | None, Path | None]:
    """
    Detect structured_json, answer_json, and pipeline_output_json from a list of paths.
    """
    structured = None
    answer = None
    pipeline = None
    
    for arg in args:
        path = Path(arg)
        if not path.exists():
            continue
            
        name = path.name.lower()
        
        # 1. Answer JSON
        if name == "answer.json":
            answer = path
        # 2. Pipeline Output JSON
        elif name == "pipeline_output.json":
            pipeline = path
        # 3. Structured JSON
        elif name.endswith("ai_structured.json") or name.endswith("_structured.json"):
            structured = path
        # Fallback content checks
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if "sections" in data:
                        structured = path
                    elif "access_methods" in data:
                        answer = path
                    elif "gap_assessment" in data or "Protocols" in data:
                        pipeline = path
            except:
                pass
                
    return structured, answer, pipeline


if __name__ == "__main__":
    import sys
    import traceback

    if len(sys.argv) < 2:
        print("Usage: python main.py <file1.json> [file2.json] [file3.json]")
        sys.exit(1)

    # Detect files from all arguments
    structured_path, answers_path, pipeline_path = detect_all_files(sys.argv[1:])
    
    if structured_path:
        print(f"Detected structured file: {structured_path.name}")
    if answers_path:
        print(f"Detected answer file: {answers_path.name}")
    if pipeline_path:
        print(f"Detected pipeline file: {pipeline_path.name}")

    if not structured_path or not answers_path:
        print("Error: Could not find both 'structured' and 'answer' files.")
        sys.exit(1)

    user_answers = ensure_answers(load_answers(answers_path))
    pipeline_output_file = pipeline_path if pipeline_path else (BASE_DIR / "pipeline_output.json")

    questions_data = load_questions(BASE_DIR / "questions.json")
    missing_answer_messages = validate_answers_against_questions(user_answers, questions_data)
    
    if missing_answer_messages:
        skeleton = _load_pipeline_skeleton(pipeline_output_file)
        for msg in missing_answer_messages:
            inject_error_into_81(skeleton, msg)
        _populate_top_level_checks(skeleton)
        output_path = BASE_DIR / "output.json"
        output_path.write_text(json.dumps(skeleton.get("skeleton", skeleton), indent=2), encoding="utf-8")
        for msg in missing_answer_messages:
            print(f"[BLOCKED] {msg}")
        sys.exit(1)

    try:
        run_validation(
            structured_path,
            user_answers,
            pipeline_path=pipeline_output_file,
        )
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)