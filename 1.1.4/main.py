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
DEFAULT_LLM_ENDPOINT = os.environ.get("DYNAMIC_LLM_URL", "")
DEFAULT_LLM_MODEL = os.environ.get("DYNAMIC_LLM_MODEL", "")
DEFAULT_VL_MODEL = os.environ.get("DYNAMIC_VLM_MODEL") or os.environ.get("DYNAMIC_LLM_MODEL") or ""
DEFAULT_LLM_BEARER_TOKEN = os.environ.get("DYNAMIC_LLM_BEARER_TOKEN") or os.environ.get("DYNAMIC_LLM_API_KEY") or ""

#DEFAULT_LLM_ENDPOINT = "http://167.179.138.57:40381/v1/chat/completions"
#DEFAULT_LLM_MODEL = "sorc/qwen3.5-instruct:latest"
#DEFAULT_VL_MODEL = "sorc/qwen3.5-instruct:latest"
#DEFAULT_LLM_BEARER_TOKEN = "ab8afd3ddee9ced8bd04b3d708278e84a03120c74327aae5389efd81ec8e61aa"

StructuredInput = Union[Dict[str, Any], str, Path]

GAP_ANALYZER_PROMPT_TEMPLATE = """
--- OUT_OF_SCOPE_VALIDATOR_PROMPT ---

SYSTEM ROLE:
You are a TELECOM TEST PLAN OUT-OF-SCOPE VALIDATOR.

Your task is to determine whether a SINGLE GIVEN TEST CASE
is IN-SCOPE or OUT-OF-SCOPE for the requirement under validation.

You must evaluate based ONLY on the TEST CASE NAME and the
declared access-method context provided in the inputs.

You must perform a strict design-intent validation.

You must NOT evaluate:
* Test execution result
* Product behavior during execution
* Whether the product actually supports the interface
* Whether the test passed or failed
* Coverage completeness of the entire document
* Any information outside the Test Case Name

=================================================
INPUTS
=================================================

Local Management Access Methods:
{local_management_access_methods}

Remote Management Access Methods:
{remote_management_access_methods}

Controller Present:
{controller_present}

Controller Name:
{controller_name}

Test Case ID:
{test_case_id}

Test Case Name:
{test_case_name}

=================================================
PRIMARY VALIDATION OBJECTIVE
=================================================

Determine whether the Test Case Name clearly validates at least
one requirement-relevant authentication obligation AND whether
the stated validation target is relevant to the declared access scope.

A test case is IN-SCOPE only when it validates a requirement-relevant
authentication obligation in a valid scope context.

A test case is OUT-OF-SCOPE when:
* it does not validate any requirement-relevant authentication obligation, OR
* it targets an invalid or undeclared access path and no exception applies.

=================================================
REQUIREMENT-RELEVANT OBLIGATIONS
=================================================

The test case is requirement-relevant only if it validates at least one
of the following obligations:

O1. Authentication is required before access is granted
O2. Access without valid authentication is denied or not possible
O3. Authenticated user or machine is uniquely and unambiguously identifiable
O4. Authentication between DUT and controller is validated, when controller is present
O5. Authentication bypass, circumvention, or brute-force success is prevented

A test case may validate one obligation only and still be IN-SCOPE.

Failure to validate one obligation does NOT make the test case OUT-OF-SCOPE
if another valid obligation is clearly validated.

Unambiguous identification is one valid obligation, but it is NOT a mandatory
gate for all authentication-related tests.

=================================================
EVIDENCE RULE
=================================================

Use ONLY the Test Case Name as evidence.

Do not infer missing intent.
Do not assume unstated meaning.
Do not use vendor names, product names, model names, or protocol names
as evidence unless they contribute to explicit authentication intent.

All evidence_phrase values must be copied exactly from the Test Case Name.
If exact evidence is not available, use empty string.

=================================================
TECHNICAL INTERPRETATION RULES
=================================================

1. Positive and negative authentication behavior are both valid.
2. Authentication-required checks and unauthenticated-access-denied checks
   are valid and requirement-relevant.
3. Identity attribution is valid only when the intent explicitly targets
   identifiable authenticated subject traceability.
4. Role separation, access-level separation, privilege separation,
   or authorization-only behavior are NOT sufficient by themselves.
5. Controller authentication is valid only when Controller Present = YES.
6. Bypass validation is valid only when it targets:
   a) a declared access path, OR
   b) a generic interface-independent authentication bypass statement
      that does not explicitly conflict with undeclared scope.
7. If the test case explicitly names an undeclared access path, that path
   must be treated as out of scope unless it clearly maps to a declared access path.
8. Generic interface wording may be accepted only when it can be reasonably
   mapped to declared management access scope without contradiction.

BYPASS RULE

Authentication bypass validation is IN-SCOPE only when it targets:
- a declared access method, or
- a generic access path that maps to declared access methods, or
- all access methods / all interfaces / all combinations of interface and
  authentication method, provided it does not conflict with declared access scope.

Authentication bypass validation is OUT-OF-SCOPE when it explicitly targets
an undeclared or conflicting access path.

=================================================
ACCESS SCOPE RESOLUTION
=================================================

Look ONLY in the Test Case Name.

Resolve whether the test case targets a valid access scope.

Set access_scope = PASS only if ANY ONE of the following is true:

A. Declared Access Match
The test case explicitly mentions an access method that appears in:
- {local_management_access_methods}
- {remote_management_access_methods}

B. Generic Access Match
The test case uses a generic local or remote management access expression
that clearly maps to one or more declared access methods.

C. Generic Bypass Exception
The test case validates authentication bypass in a generic,
interface-independent manner and does NOT explicitly target an
undeclared access path.

Set access_scope = FAIL if ANY ONE of the following is true and no PASS rule applies:

1. The test case explicitly targets an undeclared interface, undeclared
   access path, or undeclared management channel.

2. The test case targets a path that cannot be reasonably mapped to
   any declared local or remote management access method.

3. The test case is not a valid generic bypass test.

=================================================
CONTROLLER-TO-DUT VALIDATION
=================================================

This section applies ONLY when Controller Present = YES.

Principle:
When a controller is present, authentication between the DUT and
the controller is a valid requirement-relevant authentication path.

This controller-mediated authentication path is independent of the
declared local and remote management access methods.

Therefore, a controller-authentication test case must NOT be rejected
only because the controller path is not listed among the declared
management access methods.

Controller-to-DUT authentication is IN-SCOPE if the Test Case Name
clearly validates authentication behavior between the DUT and the
controller, either positive or negative.

Accept as valid controller-authentication intent when the test case
clearly validates any of the following:

1. DUT authenticates through controller
2. DUT authentication succeeds through controller
3. DUT authentication fails through controller
4. DUT access, onboarding, or join is granted through authenticated controller flow
5. DUT access, onboarding, or join is denied due to failed or missing authentication in controller flow
6. DUT authorization is denied within a controller-mediated authentication path

Product names, controller names, DUT names, model names, and vendor-specific
terms must be treated only as entity labels.

They must NOT be treated as undeclared access paths by themselves.

A controller-to-DUT test case is OUT-OF-SCOPE only when:
1. it mentions controller interaction but does not validate authentication intent, or
2. controller_present = NO

=================================================
OBLIGATION CHECKS
=================================================

Look ONLY in the Test Case Name.

T1 — authentication_required
Question:
Does the test case validate that authentication is required before access is granted?

If YES:
    authentication_required = PASS
Else:
    authentication_required = FAIL

T2 — unauthenticated_access_denied
Question:
Does the test case validate that access without valid authentication is denied,
blocked, rejected, or not possible?

If YES:
    unauthenticated_access_denied = PASS
Else:
    unauthenticated_access_denied = FAIL

T3 — unambiguous_identification
Question:
Does the test case explicitly validate that the authenticated user or machine
is uniquely and unambiguously identifiable?

Accept only identity-traceability intent.

Reject if the test case validates only:
* role separation
* privilege separation
* authorization outcome
* admin vs guest distinction
* account-type behavior
* access-level segregation

If YES:
    unambiguous_identification = PASS
Else:
    unambiguous_identification = FAIL

T4 — controller_authentication
Execute ONLY when Controller Present = YES.
If Controller Present = NO:
    controller_authentication = NA

Question:
Does the Test Case Name clearly validate authentication behavior
between the DUT and the controller?

Accept BOTH:
- positive authentication behavior
- negative authentication behavior

If YES:
    controller_authentication = PASS
Else:
    controller_authentication = FAIL

T5 — brute_force_or_bypass
Look ONLY in the Test Case Name.

Question:
Does the test case validate authentication bypass resistance,
authentication circumvention resistance, or repeated authentication
attack behavior?

If NO:
    brute_force_or_bypass = FAIL

If YES:
     Apply scope validation below.

BYPASS SCOPE VALIDATION

Set brute_force_or_bypass = PASS only if ANY ONE of the following is true:

1. The bypass validation explicitly targets a declared local or remote
    management access method.

2. The bypass validation targets a generic access expression that clearly
    maps to one or more declared access methods and does not conflict with them.

3. The bypass validation is explicitly written as applying to all access methods,
    all interfaces, or all combinations of interface and authentication method,
    and does not explicitly introduce an undeclared conflicting access path.

Set brute_force_or_bypass = FAIL if ANY ONE of the following is true:

1. The bypass validation explicitly targets an undeclared access method,
    undeclared interface, or undeclared management path.

2. The bypass validation introduces an access path that conflicts with the
    declared local or remote management access methods.

3. The bypass validation is generic in wording but also explicitly names an
    undeclared conflicting access path.

=================================================
INTENT RELEVANCE CHECK
=================================================

Set intent_relevance = PASS if ANY ONE of the following is PASS:
* authentication_required
* unauthenticated_access_denied
* unambiguous_identification
* controller_authentication
* brute_force_or_bypass

Else:
* intent_relevance = FAIL

=================================================
STRICT DECISION LOGIC
=================================================

STEP 1 — BASIC RELEVANCE

If intent_relevance = FAIL:
    FINAL = OUT-OF-SCOPE
    SKIP remaining steps

STEP 2 — CONTROLLER PRIORITY

If controller_authentication = PASS:
    FINAL = IN-SCOPE
    SKIP remaining steps

STEP 3 — GENERIC BYPASS PRIORITY

If brute_force_or_bypass = PASS:
    FINAL = IN-SCOPE
    SKIP remaining steps

STEP 4 — DECLARED ACCESS AUTHENTICATION CHECK

If access_scope = PASS AND ANY ONE of the following is PASS:
* authentication_required
* unauthenticated_access_denied
* unambiguous_identification

Then:
    FINAL = IN-SCOPE
Else:
    FINAL = OUT-OF-SCOPE

=================================================
STRICT OUT-OF-SCOPE CONDITIONS
=================================================

Mark the test case OUT-OF-SCOPE when ANY ONE of the following is true
and no earlier IN-SCOPE rule applies:

1. The test case does not validate any requirement-relevant
   authentication obligation.

2. The test case validates only authorization, role separation,
   privilege separation, or account distinction.

3. The test case explicitly targets an undeclared interface,
   undeclared access path, or undeclared management channel.

4. The test case uses generic wording that cannot be clearly mapped
   to declared management access scope.

5. The test case validates bypass or authentication behavior only on
   an undeclared path and does not qualify for the generic bypass exception.

=================================================
STRICT IN-SCOPE CONDITIONS
=================================================

Mark the test case IN-SCOPE only when ANY ONE of the following is true:

1. It validates controller authentication and Controller Present = YES.

2. It validates authentication bypass resistance and the targeted scope
   is valid under access-scope resolution.

3. It validates authentication-required behavior, denial of unauthenticated
   access, or unambiguous identification on a declared or validly mappable
   management access path.

=================================================
OUTPUT FORMAT
=================================================

Return ONLY valid JSON.

{
  "test_case_id": "{test_case_id}",
  "test_case_name": "{test_case_name}",

  "Access": {
    "access_scope": "PASS | FAIL",
    "evidence_phrase": ""
  },

  "Intent": {
    "authentication_required": {
      "result": "PASS | FAIL",
      "evidence_phrase": ""
    },
    "unauthenticated_access_denied": {
      "result": "PASS | FAIL",
      "evidence_phrase": ""
    },
    "unambiguous_identification": {
      "result": "PASS | FAIL",
      "evidence_phrase": ""
    },
    "controller_authentication": {
      "result": "PASS | FAIL | NA",
      "evidence_phrase": ""
    },
    "brute_force_or_bypass": {
      "result": "PASS | FAIL",
      "evidence_phrase": ""
    }
  },

  "intent_relevance": "PASS | FAIL",

  "final_scope": "IN-SCOPE | OUT-OF-SCOPE",

  "deviation_summary": "<A single-line audit statement ONLY if OUT-OF-SCOPE else empty string>"
}

=================================================
DEVIATION SUMMARY RULE
=================================================

If final_scope = IN-SCOPE:
    deviation_summary = ""

If final_scope = OUT-OF-SCOPE:
    deviation_summary must:
    * be a single-line audit statement
    * start with "Test case ..."
    * state only the external reason why the test case is OUT-OF-SCOPE
    * be generic and concise
    * not mention internal checks
    * not mention PASS/FAIL terms
    * not explain reasoning

=================================================
FINAL RESPONSE RULE
=================================================

Return ONLY valid JSON.
Do not return explanations.
Do not return markdown.
Do not return commentary.
Do not return anything before or after the JSON.
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
    required_keys = [
        "local_management_access_methods",
        "remote_management_access_methods",
        "controller_present",
    ]
    missing = [key for key in required_keys if key not in answers]
    if missing:
        raise ValueError(f"Missing required keys in answer.json: {', '.join(missing)}")

    normalized = dict(answers)
    normalized["local_management_access_methods"] = _normalize_list(
        answers.get("local_management_access_methods")
    )
    normalized["remote_management_access_methods"] = _normalize_list(
        answers.get("remote_management_access_methods")
    )
    normalized["controller_present"] = _normalize_bool(answers.get("controller_present"))

    if normalized["controller_present"]:
        controller_keys = ["controller_type"]
        missing_controller = [key for key in controller_keys if key not in normalized]
        if missing_controller:
            raise ValueError(
                "Missing controller fields in answer.json: " + ", ".join(missing_controller)
            )

    return normalized

# Pre-load prompts once
PROMPTS = load_prompts()
SUMMARIZATION_PROMPT = PROMPTS.get("SUMMARIZATION_PROMPT", "")
SINGLE_SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT.replace("test scenarios", "test scenario").replace("each test case", "the test case")
GAP_ANALYZER_PROMPT = GAP_ANALYZER_PROMPT_TEMPLATE # Use hardcoded latest version
COVERAGE_VALIDATOR_PROMPT = PROMPTS.get("COVERAGE_VALIDATOR_PROMPT", "")


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
                    "description": ts.get("description", "")
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
    user_answers: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
) -> list:
    """
    Send scenarios 2 at a time to the LLM for summarization.
    If the last chunk has only 1 scenario, send it alone with a single-item prompt.
    Returns a combined list of {test_case_id, test_case_summary} dicts.
    """
    # Pre-compute answer values for prompt rendering
    local_methods = ", ".join(user_answers.get("local_management_access_methods", []))
    remote_methods = ", ".join(user_answers.get("remote_management_access_methods", []))
    answer_values = {
        "local_management_access_methods": local_methods,
        "remote_management_access_methods": remote_methods,
        "controller_present": "YES" if user_answers.get("controller_present") else "NO",
        "controller_type": user_answers.get("controller_type", ""),
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

        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError(
                "Unsupported LLM endpoint. Use /api/generate or /v1/chat/completions"
            )

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
    user_answers: Dict[str, str],
    test_scenario_summaries: str,
    test_scenarios_text: str,
) -> str:
    prompt_template = COVERAGE_VALIDATOR_PROMPT

    local_methods = ", ".join(user_answers.get("local_management_access_methods", []))
    remote_methods = ", ".join(user_answers.get("remote_management_access_methods", []))

    payload = {
        "test_scenario_summaries": test_scenario_summaries,
        "test_scenarios": test_scenarios_text,
        "local_management_access_methods": local_methods,
        "remote_management_access_methods": remote_methods,
        "controller_present": "YES" if user_answers.get("controller_present") else "NO",
        "controller_name": user_answers.get("controller_type", ""),
        "controller_type": user_answers.get("controller_type", ""),
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
    Displays a preview of the final prompt in the CLI for the first case.
    """
    local_methods = ", ".join(user_answers.get("local_management_access_methods", []))
    remote_methods = ", ".join(user_answers.get("remote_management_access_methods", []))
    
    base_payload = {
        "local_management_access_methods": local_methods,
        "remote_management_access_methods": remote_methods,
        "controller_present": "YES" if user_answers.get("controller_present") else "NO",
        "controller_name": user_answers.get("controller_type", ""),
    }

    all_gaps = {}
    count = len(scenarios)

    print("\n" + "="*60)
    print("PHASE 3: GAP ANALYSIS & OUT-OF-SCOPE VALIDATION")
    print("="*60)

    for idx, s in enumerate(scenarios, 1):
        tid = s.get("tid", "Unknown")
        name = s.get("description", "")

        # STRICT: use exact Section 11 heading from JSON — no synthesis
        if "section11_key" not in s or not s["section11_key"]:
            raise ValueError(f"Missing Section 11 key for scenario index {idx} (tid={tid})")
        section_11_id = s["section11_key"]

        render_values = dict(base_payload)
        render_values["test_case_id"] = tid
        render_values["test_case_name"] = name

        prompt_text = render_prompt(GAP_ANALYZER_PROMPT_TEMPLATE, render_values)
        messages = _parse_chatml(prompt_text)

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
            all_gaps[section_11_id] = {
                "test_case_name": entry.get("test_case_name", name),
                "deviation_summary": entry.get("deviation_summary", "")
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


def run_validation(
    structured_json: StructuredInput,
    user_answers: Dict[str, str],
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> Dict[str, Any]:
    structured_data = load_structured_json(structured_json)

    # 1. Extract and combine scenarios from 8.1 and 8.4
    all_extracted_scenarios = extract_scenarios(structured_data)
    
    # Format for coverage prompt (Phase 1/2 style)
    test_scenarios_lines = []
    for ts in all_extracted_scenarios:
        tid = ts.get("tid", "")
        desc = ts.get("description", "")
        test_scenarios_lines.append(f"- {tid} {desc}")
    test_scenarios_text = "\n".join(test_scenarios_lines)

    print(f"Found {len(all_extracted_scenarios)} combined test scenarios from Sections 8.1 and 8.4")

    # 2. Summarize (keeping this as part of the original pipeline if needed)
    # Note: The original code used scenarios from 8.4 here
    print("Phase 1: Summarizing scenarios in pairs...")
    summaries = summarize_scenarios_in_pairs(all_extracted_scenarios, user_answers, llm_endpoint, llm_model)

    # Format summaries as text block for the final prompt
    summary_lines = []
    for s in summaries:
        tid = s.get("test_case_id", "Unknown")
        obj = s.get("objective", "")
        exe = s.get("execution_summary", "")
        tsum = f"{obj} | Execution: {exe}"
        summary_lines.append(f"- {tid} {tsum}")
    test_scenario_summaries = "\n".join(summary_lines)

    print(f"\nPhase 1 complete. {len(summaries)} summaries collected.")
    print("\nPhase 2: Running coverage validation...")

    # Phase 2: Build final prompt with summaries and validate
    coverage_prompt_text = build_prompt(user_answers, test_scenario_summaries, test_scenarios_text)
    coverage_messages = _parse_chatml(coverage_prompt_text)

    if llm_endpoint.endswith("/api/generate"):
        coverage_response = _ollama_generate(llm_endpoint, llm_model, coverage_messages)
    elif llm_endpoint.endswith("/v1/chat/completions"):
        coverage_response = _openai_chat_completions(llm_endpoint, llm_model, coverage_messages)
    else:
        raise ValueError("Unsupported LLM endpoint")

    coverage_result = _extract_json(coverage_response)

    # --- Phase 3: Run Gap Analysis ---
    print("\nPhase 3: Running gap analysis sequentially...")
    
    gap_result = run_gap_analysis(
        all_extracted_scenarios, user_answers, llm_endpoint, llm_model
    )

    return coverage_result, gap_result


def _write_output_summary(result: Dict[str, Any], output_path: Path) -> None:
    missing = result.get("missing_coverage_summary", [])
    if not isinstance(missing, list):
        missing = [str(missing)] if missing else []

    summary = []
    for item in missing:
        if item:  # skip empty strings (PASS case)
            summary.append({
                "where": "Test Case Coverage Gaps",
                "what": "Section 8.1",
                "suggestion": [str(item)],
                "redirect_text": "Number of Test Scenarios",
                "severity": "Critical",
            })

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _update_pipeline_output(gap_result: Dict[str, Any], user_answers: Dict[str, Any], output_path: Path) -> None:
    """
    Update/Append gap analysis results to the pipeline_output.json.
    Maintains gap_assessment as a DICTIONARY with Section 11 keys.
    Also syncs the Protocols field from answer.json.
    Only entries with a non-empty deviation_summary are written (OUT-OF-SCOPE only).
    """
    # Always resolve to absolute path to ensure the correct file is updated
    output_path = output_path.resolve()

    if output_path.exists():
        try:
            content = output_path.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content)
            else:
                data = {}
        except Exception as e:
            print(f"  Warning: Could not parse existing {output_path.name}: {e}. Creating new.")
            data = {}
    else:
        data = {}
        
    if not isinstance(data, dict):
        data = {}
        
    # Ensure gap_assessment key exists as a dict (do NOT reset if already present)
    if "gap_assessment" not in data or not isinstance(data["gap_assessment"], dict):
        data["gap_assessment"] = {}
        
    # 1. Filter: only keep entries where deviation_summary is non-empty (OUT-OF-SCOPE only)
    new_gaps = gap_result.get("gap_assessment", {})
    filtered_gaps = {
        key: val
        for key, val in new_gaps.items()
        if isinstance(val, dict) and val.get("deviation_summary", "").strip()
    }

    # 2. Merge filtered results into existing gap_assessment (preserve existing keys)
    data["gap_assessment"].update(filtered_gaps)

    # 3. Sync Protocols from answer.json
    remote_methods = user_answers.get("remote_management_access_methods", [])
    if isinstance(remote_methods, list):
        data["Protocols"] = ",".join(remote_methods)
    else:
        data["Protocols"] = str(remote_methods)

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Successfully updated {output_path} with {len(filtered_gaps)} OUT-OF-SCOPE entries (skipped {len(new_gaps) - len(filtered_gaps)} IN-SCOPE).")


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
    print(f"Intelligence written → {output_path.name}  (compliance_score: {score_str})")


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
                    elif "local_management_access_methods" in data or "remote_management_access_methods" in data:
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

    try:
        coverage_result, gap_result = run_validation(structured_path, user_answers)

        _write_output_summary(coverage_result, BASE_DIR / "output.json")
        
        # Write compliance score to intelligence.json (isolated from output.json / pipeline)
        _write_intelligence_json(coverage_result, BASE_DIR / "intelligence.json")
        
        # Update/Append gap analysis results to pipeline_output.json
        _update_pipeline_output(gap_result, user_answers, pipeline_output_file)

        print("\n=== COVERAGE VALIDATION RESULT ===\n")
        print(json.dumps(coverage_result, indent=2))
        
        print("\n=== GAP ANALYSIS RESULT ===\n")
        print(json.dumps(gap_result, indent=2))
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
