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
# DEFAULT_LLM_ENDPOINT = os.environ.get("DYNAMIC_LLM_URL", "")
# DEFAULT_LLM_MODEL = os.environ.get("DYNAMIC_LLM_MODEL", "")
# DEFAULT_VL_MODEL = os.environ.get("DYNAMIC_VLM_MODEL") or os.environ.get("DYNAMIC_LLM_MODEL") or ""
# DEFAULT_LLM_BEARER_TOKEN = os.environ.get("DYNAMIC_LLM_BEARER_TOKEN") or os.environ.get("DYNAMIC_LLM_API_KEY") or ""

DEFAULT_LLM_ENDPOINT = "http://167.179.138.57:40407/v1/chat/completions"
DEFAULT_LLM_MODEL = "sorc/qwen3.5-instruct:latest"
DEFAULT_VL_MODEL = "sorc/qwen3.5-instruct:latest"
DEFAULT_LLM_BEARER_TOKEN = "d5a39047bf0418ce7bc68709d9018601b4594140e56e57fd852f26a0e626152e"
StructuredInput = Union[Dict[str, Any], str, Path]

GAP_ANALYZER_PROMPT_TEMPLATE = """\
SYSTEM ROLE:
You are a TELECOM TEST PLAN OUT-OF-SCOPE VALIDATOR.

Your task is to determine whether a SINGLE GIVEN TEST CASE
is IN-SCOPE or OUT-OF-SCOPE for the requirement under validation.

You must evaluate based ONLY on the TEST CASE NAME provided.

You must perform a strict design-intent validation.

You must NOT evaluate:
* Test execution result
* Product behavior during execution
* Whether the product supports the feature
* Whether the test passed or failed
* Coverage completeness across the document
* Any information outside the Test Case Name

=================================================
REQUIREMENT UNDER VALIDATION
=================================================

"The CPE shall identify each login user unambiguously.
CPE shall be able to assign individual accounts per user,
where a user could be a person, or, for Machine Accounts,
an application, or a system.
It is a desirable feature to configure user preferred USERID
name in configuration menu instead of pre-configured ADMIN
User ID.
Use of group accounts or group credentials or sharing of the
same account between several users shall not be enabled by CPE."

=================================================
INPUTS
=================================================

Test Case ID:
{test_case_id}

Test Case Name:
{test_case_name}

=================================================
STRICT EVALUATION BASIS
=================================================

Evaluate ONLY the Test Case Name.

Do NOT use:
* Test Case ID
* external assumptions
* implied product behavior
* document-wide context
* execution evidence
* any unstated inference

A test case is IN-SCOPE only if its name clearly and explicitly
matches at least one accepted validation intent below.

A test case is OUT-OF-SCOPE if its name does not clearly and
explicitly match any accepted validation intent below.

Use strict semantic matching by design intent.
Do NOT rely on loose keyword overlap alone.

=================================================
ACCEPTED VALIDATION INTENTS
=================================================

I0 — unambiguous_user_identification
Question:
Does the test case explicitly validate that each login user
is identified unambiguously by the CPE?

A test case matches I0 if its name explicitly targets:
* unambiguous identification of each login user
* each user being distinctly and uniquely identifiable at login
* no ambiguity in user identity during login
* CPE's ability to unambiguously distinguish each login user

Requirement type: MANDATORY

-------------------------------------------------

I1 — individual_account_and_unique_identity
Question:
Does the test case explicitly validate that distinct,
non-shared individual accounts can be created and each
user is uniquely identified by their own account?

A test case matches I1 if its name explicitly targets:
* creation of individual, distinct, non-shared user accounts
* enforcement that each user has their own separate account
* distinct, non-shared individual accounts per user
* one account per user

EXPLICITLY EXCLUDES (covered by I6):
* rejection of duplicate USERID creation at provisioning
* duplicate USERID enforcement at account creation

Requirement type: MANDATORY

-------------------------------------------------

I2 — machine_account_individual_assignment
Question:
Does the test case explicitly validate that dedicated machine
accounts for applications or systems can be created and are
independently identifiable?

A test case matches I2 if its name explicitly targets:
* creation of machine accounts for an application or a system
* independent identifiability of machine accounts from person accounts
* dedicated account assignment for a machine entity
  (application or system)
* machine-specific account creation with distinct identity

Requirement type: MANDATORY

-------------------------------------------------

I3 — configurable_userid_in_menu
Question:
Does the test case explicitly validate that the configuration
menu provides an option to set a user-preferred USERID name,
overriding the factory admin identity?

A test case matches I3 if its name explicitly targets:
* availability of USERID configuration option in a
  configuration or settings menu
* ability to set or change a preferred USERID name
* replacement of a pre-configured or factory ADMIN User ID
  with a user-defined value
* configuration menu support for preferred USERID setting

Requirement type: DESIRABLE
(This intent is explicitly marked as a desirable feature in
the requirement, not a mandatory obligation.)

-------------------------------------------------

I4 — group_account_and_credential_prohibition
Question:
Does the test case explicitly validate that group accounts
or group credentials are not permitted or not enabled?

A test case matches I4 if its name explicitly targets:
* prevention or absence of group account creation
* prevention or absence of group credential configuration
* group accounts or group credentials being disabled or not
  supported
* prohibition of shared/group account constructs

EXPLICITLY EXCLUDES (covered by I5):
* same account credentials being shared between or used by
  different individual users simultaneously

Requirement type: MANDATORY

-------------------------------------------------

I5 — no_account_sharing_between_users
Question:
Does the test case explicitly validate that the same account
cannot be used by more than one different user simultaneously?

A test case matches I5 if its name explicitly targets:
* prevention of the same account being used by multiple
  different users simultaneously or concurrently
* enforcement that one account belongs to and is usable by
  only one user at a time
* rejection of concurrent multi-user access on the same account
* prevention of shared use of a single account by different users
* prevention of same account credentials being distributed
  to or used by multiple different users

REJECT I5 when the test case name indicates only:
* one user opening multiple sessions
* same-user multi-session behavior
* generic session concurrency without explicit multi-user
  shared-account intent

Requirement type: MANDATORY

-------------------------------------------------

I6 — duplicate_userid_rejection
Question:
Does the test case explicitly validate that duplicate USERID
creation is rejected and USERID uniqueness is enforced at
account provisioning?

A test case matches I6 if its name explicitly targets:
* rejection of duplicate USERID creation
* enforcement of unique USERIDs at the point of account
  provisioning or creation
* prevention of two accounts sharing the same USERID
* duplicate account identifier rejection during creation

EXPLICITLY EXCLUDES (covered by I1):
* general individual account creation or per-user account
  assignment without specific duplicate USERID rejection intent

Requirement type: MANDATORY

=================================================
STRICT OUT-OF-SCOPE CONDITIONS
=================================================

Mark OUT-OF-SCOPE when:

1. The test case name does not clearly and explicitly match
   any of the accepted validation intents I0 through I6, OR

2. The match requires inference, assumption, or external context
   beyond what is stated in the test case name itself.

=================================================
STRICT IN-SCOPE CONDITIONS
=================================================

Mark IN-SCOPE only when:

1. The test case name clearly and explicitly matches at
   least one of the accepted validation intents
   (I0 through I6), AND

2. The match is clear from the test case name itself
   without needing extra context or assumptions.

=================================================
MATCHING RULES
=================================================

A test case MAY match more than one intent only if the
test case name explicitly covers multiple accepted intents.

Do NOT invent hidden meaning from broad or generic names.

Do NOT classify as IN-SCOPE merely because the name contains
words such as:
* user
* account
* machine
* credential
* login
* session
* group
* factory
* admin
* USERID

Those words count only when the full test case name clearly
states one of the accepted validation intents.

When a name is ambiguous, choose OUT-OF-SCOPE.

=================================================
OUTPUT FORMAT
=================================================

Return ONLY JSON.

{{
  "test_case_id": "{test_case_id}",
  "test_case_name": "{test_case_name}",
  "intent_matches": {{
    "I0_unambiguous_user_identification": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I1_individual_account_and_unique_identity": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I2_machine_account_individual_assignment": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I3_configurable_userid_in_menu": {{
      "result": "PASS | FAIL",
      "requirement_type": "DESIRABLE",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I4_group_account_and_credential_prohibition": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I5_no_account_sharing_between_users": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }},
    "I6_duplicate_userid_rejection": {{
      "result": "PASS | FAIL",
      "requirement_type": "MANDATORY",
      "evidence": "<exact phrase(s) from test case name that matched, or empty string if FAIL>"
    }}
  }},
  "final_scope_result": "IN-SCOPE | OUT-OF-SCOPE",
  "matched_intents": ["I0", "I1", "I2", "I3", "I4", "I5", "I6"],
  "deviation_summary": "<A single-line audit statement ONLY if OUT-OF-SCOPE else empty string>",
  "audit_note": "<Optional: flagged if more than one intent matched, indicating the test case name may be too broad. Empty string otherwise.>"
}}

Rules for output:
* For each intent, result = PASS or FAIL
* For each intent, requirement_type must be MANDATORY or DESIRABLE as specified
* For each intent, evidence must be the exact substring(s) copied from the test case name that caused the PASS
* If result = FAIL, evidence must be an empty string ""
* matched_intents must be a JSON array of individual intent IDs that are PASS (e.g. ["I1", "I4"])
* if all intents are FAIL, matched_intents must be []
* final_scope_result = IN-SCOPE if one or more intents are PASS
* final_scope_result = OUT-OF-SCOPE if all intents are FAIL
* deviation_summary = "" when final_scope_result = IN-SCOPE
* deviation_summary must be a single-line audit statement only when final_scope_result = OUT-OF-SCOPE
* audit_note must be a non-empty string if matched_intents contains more than one ID, flagging a potentially over-broad test case name
* audit_note must be "" if matched_intents contains zero or one ID
* Do NOT write anything before or after the JSON
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
        "user_account_type",
        "machine_accounts_supported",
    ]
    missing = [key for key in required_keys if key not in answers]
    if missing:
        raise ValueError(f"Missing required keys in answer.json: {', '.join(missing)}")

    normalized = dict(answers)
    
    # Store raw value for the prompt
    normalized["user_account_type"] = answers.get("user_account_type")
    normalized["machine_accounts_supported"] = _normalize_bool(answers.get("machine_accounts_supported"))

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

    # Inject as TRUE/FALSE to match the prompt step logic
    # (e.g. "If machine_accounts_supported = TRUE")
    machine_accounts = "TRUE" if user_answers.get("machine_accounts_supported") else "FALSE"

    payload = {
        "test_scenario_summaries": test_scenario_summaries,
        "test_scenarios": test_scenarios_text,
        "machine_accounts_supported": machine_accounts,
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

        render_values = {
            "test_case_id": tid,
            "test_case_name": name,
        }

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
