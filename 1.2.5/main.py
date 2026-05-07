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


DEFAULT_LLM_ENDPOINT = os.environ.get("DYNAMIC_LLM_URL", "http://109.165.141.174:30406/v1/chat/completions")
DEFAULT_LLM_MODEL = os.environ.get("DYNAMIC_LLM_MODEL", "sorc/qwen3.5-instruct:latest")
DEFAULT_VL_MODEL = os.environ.get("DYNAMIC_VLM_MODEL") or os.environ.get("DYNAMIC_LLM_MODEL") or "sorc/qwen3.5-instruct:latest"
DEFAULT_LLM_BEARER_TOKEN = os.environ.get("DYNAMIC_LLM_BEARER_TOKEN") or os.environ.get("DYNAMIC_LLM_API_KEY") or "9f40414a2c1b9762595c198c554163dab08bb34d1181e670d3f892b3cb46eb3a"
StructuredInput = Union[Dict[str, Any], str, Path]


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
    """
    Validate and normalize answer.json for Requirement 1.2.5.
    Accepts either:
      - New format: lan_supported/wifi_supported (yes/no or true/false) + authentication_interfaces
      - Legacy format: user_session_types (list) + authentication_interfaces
    Converts new format into legacy user_session_types list before returning.
    """
    required_keys = [
        "authentication_interfaces",
    ]
    missing = [key for key in required_keys if key not in answers]
    if missing:
        raise ValueError(f"Missing required keys in answer.json: {', '.join(missing)}")

    normalized = dict(answers)

    # Validate authentication_interfaces
    if not isinstance(normalized.get("authentication_interfaces"), str) or not normalized.get("authentication_interfaces").strip():
        raise ValueError("authentication_interfaces must be a non-empty string")

    # Detect new input format (lan_supported / wifi_supported)
    has_new_format = "lan_supported" in answers or "wifi_supported" in answers

    if has_new_format:
        # Normalize boolean true/false (JSON or Python) → "yes"/"no" strings
        for field in ("lan_supported", "wifi_supported"):
            val = normalized.get(field)
            if isinstance(val, bool):
                normalized[field] = "yes" if val else "no"
            elif isinstance(val, str) and val.strip().lower() in ("true", "1"):
                normalized[field] = "yes"
            elif isinstance(val, str) and val.strip().lower() in ("false", "0"):
                normalized[field] = "no"

        if normalized.get("lan_supported") not in ["yes", "no"]:
            raise ValueError("lan_supported must be 'yes', 'no', true, or false")
        if normalized.get("wifi_supported") not in ["yes", "no"]:
            raise ValueError("wifi_supported must be 'yes', 'no', true, or false")

        # Convert new format into legacy session types list
        session_types = []
        if normalized.get("lan_supported") == "yes":
            session_types.append("lan_authenticated_users")
        if normalized.get("wifi_supported") == "yes":
            session_types.append("wifi_authenticated_users")
        # Admin sessions assumed always present
        session_types.append("admin_users")
        normalized["user_session_types"] = session_types

    else:
        # Legacy format: validate user_session_types directly
        if not isinstance(normalized.get("user_session_types"), list):
            raise ValueError("user_session_types must be a list")

    return normalized



def map_session_applicability(answers: dict) -> dict:
    session_types = answers.get("user_session_types", [])
    result = {
        "has_authenticated_sessions": False,
        "has_admin_sessions": False,
        "has_data_sessions": False
    }

    if not isinstance(session_types, list):
        return result

    if "admin_users" in session_types:
        result["has_authenticated_sessions"] = True
        result["has_admin_sessions"] = True

    if "wifi_authenticated_users" in session_types or "lan_authenticated_users" in session_types:
        result["has_authenticated_sessions"] = True
        result["has_data_sessions"] = True

    return result

# Pre-load prompts once
PROMPTS = load_prompts()
SUMMARIZATION_PROMPT = PROMPTS.get("SUMMARIZATION_PROMPT", "")
SINGLE_SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT.replace("test scenarios", "test scenario").replace("each test case", "the test case")
# Requirement 1.2.5 – single prompt for coverage validation (no routing)
COVERAGE_VALIDATOR_PROMPT_1_2_5 = PROMPTS.get("COVERAGE_VALIDATOR_PROMPT_1_2_5", "")
# Requirement 1.2.5 – single prompt for gap/scope analysis (no routing)
GAP_ANALYZER_PROMPT_1_2_5 = PROMPTS.get("SCOPE_VALIDATOR_PROMPT_1_2_5") or PROMPTS.get("GAP_ANALYZER_PROMPT_1_2_5", "")


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

def extract_scenarios_for_llm(health: dict) -> list:
    """
    Positional extraction from raw section data (8.1, 8.4, 11).
    Returns combined list for LLM processing.
    """
    scenarios_81 = health["raw_81"]
    steps_84 = health["raw_84"]
    cases_11 = health["raw_11"]

    combined = []
    count = len(scenarios_81)

    for i in range(count):
        desc = scenarios_81[i].get("description", "") if i < len(scenarios_81) else ""
        
        # Capture the original test scenario label from 8.1
        s81_key = scenarios_81[i].get("test_scenario", f"Test Scenario {i + 1}") if i < len(scenarios_81) else f"Test Scenario {i + 1}"

        steps = []
        if i < len(steps_84):
            steps = steps_84[i].get("steps", [])
        elif i < len(cases_11):
            tc = cases_11[i]
            steps = [s for s in tc.get("execution", []) if isinstance(s, dict) and "step" in s]

        s11_key = None
        if i < len(cases_11):
            s11_key = cases_11[i].get("test_case_heading", "").strip() or None

        combined.append({
            "tid": f"Test Scenario {i + 1}",
            "description": desc,
            "steps": steps,
            "section11_key": s11_key,
            "section81_key": s81_key,
        })

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
    """
    Build the coverage validation prompt for Requirement 1.2.5.
    Always uses COVERAGE_VALIDATOR_PROMPT_1_2_5 — no conditional routing.
    """
    print("[PROMPT ROUTE] → COVERAGE_VALIDATOR_PROMPT_1_2_5")
    prompt_template = COVERAGE_VALIDATOR_PROMPT_1_2_5

    payload = {
        "test_scenario_summaries": test_scenario_summaries,
        "test_scenarios": test_scenarios_text,
        "user_session_types": json.dumps(user_answers.get("user_session_types", [])),
        "authentication_interfaces": user_answers.get("authentication_interfaces", ""),
        "lan_supported": user_answers.get("lan_supported", "no"),
        "wifi_supported": user_answers.get("wifi_supported", "no"),
    }

    return render_prompt(prompt_template, payload)


def run_gap_analysis(
    scenarios: list,
    user_answers: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
) -> Dict[str, Any]:
    """
    Run gap/scope analysis on each scenario for Requirement 1.2.5.
    Always uses GAP_ANALYZER_PROMPT_1_2_5 — no conditional routing.
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

        render_values = {
            "test_case_id": tid,
            "test_case_name": name,
            "authentication_interfaces": user_answers.get("authentication_interfaces", ""),
            "lan_supported": user_answers.get("lan_supported", "no"),
            "wifi_supported": user_answers.get("wifi_supported", "no"),
        }

        prompt_text = render_prompt(GAP_ANALYZER_PROMPT_1_2_5, render_values)
        messages = _parse_chatml(prompt_text)

        if llm_endpoint.endswith("/api/generate"):
            response_text = _ollama_generate(llm_endpoint, llm_model, messages)
        elif llm_endpoint.endswith("/v1/chat/completions"):
            response_text = _openai_chat_completions(llm_endpoint, llm_model, messages)
        else:
            raise ValueError("Unsupported LLM endpoint")

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
            all_gaps[section_11_id] = {
                "test_case_name": entry.get("test_case_name", name),
                "deviation_summary": entry.get("deviation_summary", ""),
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
                what="Coverage Validation",
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
    all_scenarios = extract_scenarios_for_llm(health)

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

    print("\n" + "="*60)
    print("COVERAGE VALIDATOR — FINAL PROMPT SENT TO LLM")
    print("="*60)
    print(coverage_prompt_text)
    print("="*60 + "\n")

    coverage_messages = _parse_chatml(coverage_prompt_text)
    if llm_endpoint.endswith("/api/generate"):
        coverage_response = _ollama_generate(llm_endpoint, llm_model, coverage_messages)
    elif llm_endpoint.endswith("/v1/chat/completions"):
        coverage_response = _openai_chat_completions(llm_endpoint, llm_model, coverage_messages)
    else:
        raise ValueError("Unsupported LLM endpoint")

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
    intent_results = result.get("intent_results", [])

    applicable = [i for i in intent_results if i.get("result") != "NOT_APPLICABLE"]
    passed = [i for i in applicable if i.get("result") == "PASS"]

    score_float = (len(passed) / len(applicable)) * 100 if applicable else 0.0

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
    
    # Task 5 & 6: Applicability mapping and check
    mapped = map_session_applicability(user_answers)
    if not mapped["has_authenticated_sessions"]:
        print("Requirement 1.2.5 NOT APPLICABLE")
        sys.exit(0)

    pipeline_output_file = pipeline_path if pipeline_path else (BASE_DIR / "pipeline_output.json")

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
