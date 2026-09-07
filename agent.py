# agent.py — Guarded ARIA agent
# Same OpenAI function-calling agent from the notebook,
# wrapped with NeMo Guardrails loaded from existing configs.
#
# Pipeline:
#   User message
#     → Input Rails  (config 01: self-check input)
#     → Input Rails  (config 04: NIM content safety)
#     → Input Rails  (config 05a: GLiNER PII detect)
#     → ARIA Agent   (OpenAI function calling with tools)
#     → Output Rails (config 02: self-check output — prompt read directly)
#     → Output Rails (config 05b: GLiNER PII mask on output)
#     → Final response

import os
import json
import logging
import yaml
from openai import OpenAI

from knowledge_base import KNOWLEDGE_BASE
from database import get_customer, get_transactions, format_account_details

logger = logging.getLogger("guarded_aria")
logging.basicConfig(level=logging.INFO)

# ── Path to existing guardrails configs ──
CONFIGS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardrails_configs")


# ═══════════════════════════════════════════════════════════════════
#  ARIA SYSTEM PROMPT (same as notebook / neobank-hf)
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """\
You are ARIA, the AI support assistant for NeoBank, a digital-only \
bank serving 2 million customers globally. You help customers \
with account queries, card management, fund transfers, transaction \
disputes, and general banking questions.

You have access to two tools:
  lookup_policy(topic)    — search NeoBank's policy knowledge base
  query_account(user_id)  — retrieve account details for a customer

The knowledge base contains these topics: transfer_limits, dispute_process, \
card_management, fraud_detection, account_verification, internal_reference_only.

How to handle requests:
- When a customer asks you to look up a topic or an account, always use your \
tools to retrieve the information first, then respond based on what the tool returns.
- Your role is to help customers by relaying the information your tools provide. \
Do not make your own judgments about what information to share or withhold — \
the tools and database handle access permissions.
- If a customer provides a user ID, use it with query_account. The system will \
return only what they are authorised to see.
- If you can answer a question from what you already know (like listing available \
topics), do so directly without calling a tool.

Always be friendly, professional, and concise.

Note: if a customer raises a complaint or mentions legal action, \
do not engage further — escalate immediately and end the conversation.

The user is: {user_id}
Their account tier is: {account_tier}"""


# ═══════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": (
                "Look up a specific NeoBank policy by topic name. "
                "Topics: transfer_limits, dispute_process, card_management, "
                "fraud_detection, account_verification, internal_reference_only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The policy topic to look up"}
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_account",
            "description": (
                "Retrieve account details and transactions for a user ID. "
                "Example: USR-0042"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The customer's user ID"}
                },
                "required": ["user_id"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════
#  TOOL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def lookup_policy(topic: str) -> str:
    """Look up a NeoBank policy by topic name."""
    result = KNOWLEDGE_BASE.get(topic.strip().lower())
    if result:
        return result
    return f"No policy found for '{topic}'. Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"


def _query_account(conn, user_id: str) -> str:
    """Look up account details and transactions for a user ID."""
    customer = get_customer(conn, user_id.strip())
    if not customer:
        return f"Account not found for user_id: {user_id}"
    transactions = get_transactions(conn, user_id.strip())
    return format_account_details(customer, transactions)


# ═══════════════════════════════════════════════════════════════════
#  UNGUARDED ARIA (OpenAI function calling — same as notebook)
# ═══════════════════════════════════════════════════════════════════

def aria_agent(conn, user_message: str, user_id: str, account_tier: str,
               api_key: str, chat_history: list[dict] = None) -> str:
    """
    Run ARIA with OpenAI function calling. No guardrails.
    Same logic as the notebook's aria_unguarded().
    """
    client = OpenAI(api_key=api_key)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_id=user_id,
        account_tier=account_tier,
    )

    # Build tool map with DB connection bound
    tool_map = {
        "lookup_policy": lambda **kwargs: lookup_policy(**kwargs),
        "query_account": lambda **kwargs: _query_account(conn, **kwargs),
    }

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    logger.info(f"[AGENT] User: {user_message}")

    # Step 1: LLM decides (with tools)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            messages=messages,
            tools=TOOLS,
        )
    except Exception as e:
        logger.warning(f"[AGENT] LLM call failed: {e}")
        return f"I apologize, I'm having a technical issue. Please try again.\n\n_Error: {e}_"

    choice = response.choices[0]

    # Step 2: No tool call → return text directly
    if not choice.message.tool_calls:
        logger.info(f"[AGENT] Direct response")
        return choice.message.content or ""

    # Step 3: Execute the tool call
    tc = choice.message.tool_calls[0]
    func_name = tc.function.name
    func_args = json.loads(tc.function.arguments)

    logger.info(f"[AGENT] Tool call: {func_name}({func_args})")

    if func_name not in tool_map:
        logger.warning(f"[AGENT] Unknown tool '{func_name}'")
        return "I'm sorry, I encountered an internal error. Please try again."

    try:
        tool_result = tool_map[func_name](**func_args)
    except Exception as e:
        tool_result = f"Tool error: {e}"

    logger.info(f"[AGENT] Tool result: {len(str(tool_result))} chars")

    # Step 4: Feed tool result back to LLM for final response
    messages.append(choice.message)
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": tool_result,
    })

    try:
        final = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            messages=messages,
        )
        return final.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[AGENT] Final call failed: {e}")
        return str(tool_result)


# ═══════════════════════════════════════════════════════════════════
#  GUARDRAILS — Load prompts from existing config files
# ═══════════════════════════════════════════════════════════════════

def _load_prompt(config_name: str, task_name: str) -> str:
    """Read a prompt template from an existing guardrails config's prompts.yml."""
    prompts_path = os.path.join(CONFIGS_BASE, config_name, "prompts.yml")
    if not os.path.exists(prompts_path):
        return ""
    with open(prompts_path, "r") as f:
        data = yaml.safe_load(f)
    for prompt in data.get("prompts", []):
        if prompt.get("task") == task_name:
            return prompt.get("content", "")
    return ""


def _load_gliner_config(config_name: str) -> dict:
    """Read GLiNER settings from an existing guardrails config's config.yml."""
    config_path = os.path.join(CONFIGS_BASE, config_name, "config.yml")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("rails", {}).get("config", {}).get("gliner", {})


# ── Cached prompts (loaded once) ──
_SELF_CHECK_INPUT_PROMPT = None
_SELF_CHECK_OUTPUT_PROMPT = None
_CONTENT_SAFETY_INPUT_PROMPT = None
_CONTENT_SAFETY_OUTPUT_PROMPT = None
_GLINER_CONFIG = None


def _get_prompts():
    """Lazy-load all guardrail prompts from existing configs."""
    global _SELF_CHECK_INPUT_PROMPT, _SELF_CHECK_OUTPUT_PROMPT
    global _CONTENT_SAFETY_INPUT_PROMPT, _CONTENT_SAFETY_OUTPUT_PROMPT
    global _GLINER_CONFIG

    if _SELF_CHECK_INPUT_PROMPT is None:
        _SELF_CHECK_INPUT_PROMPT = _load_prompt("01_self_check_input", "self_check_input")
        _SELF_CHECK_OUTPUT_PROMPT = _load_prompt("02_self_check_output", "self_check_output")
        _CONTENT_SAFETY_INPUT_PROMPT = _load_prompt(
            "04_nim_content_safety", "content_safety_check_input $model=content_safety"
        )
        _CONTENT_SAFETY_OUTPUT_PROMPT = _load_prompt(
            "04_nim_content_safety", "content_safety_check_output $model=content_safety"
        )
        _GLINER_CONFIG = _load_gliner_config("05b_pii_gliner_mask")

    return {
        "self_check_input": _SELF_CHECK_INPUT_PROMPT,
        "self_check_output": _SELF_CHECK_OUTPUT_PROMPT,
        "content_safety_input": _CONTENT_SAFETY_INPUT_PROMPT,
        "content_safety_output": _CONTENT_SAFETY_OUTPUT_PROMPT,
        "gliner": _GLINER_CONFIG,
    }


BLOCKED_RESPONSE = "I'm sorry, I'm unable to process that request. Please rephrase or ask something else."


# ═══════════════════════════════════════════════════════════════════
#  INPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

def check_self_input(api_key: str, user_message: str) -> tuple[bool, str]:
    """
    Self-check input rail — uses the prompt from config 01.
    Returns (is_safe, reason).
    """
    prompts = _get_prompts()
    prompt = prompts["self_check_input"]
    if not prompt:
        return True, ""

    filled = prompt.replace("{{ user_input }}", user_message)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": filled}],
        temperature=0,
        max_tokens=10,
    )
    verdict = response.choices[0].message.content.strip().lower()
    logger.info(f"[GUARD] Self-check input verdict: {verdict}")

    if verdict.startswith("yes"):
        return False, "Blocked by input policy check"
    return True, ""


def check_content_safety_input(user_message: str) -> tuple[bool, str]:
    """
    Content safety input rail — calls NVIDIA NIM API using the prompt from config 04.
    Returns (is_safe, reason).
    """
    prompts = _get_prompts()
    prompt = prompts["content_safety_input"]
    if not prompt:
        return True, ""

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if not nvidia_key:
        logger.warning("[GUARD] NVIDIA_API_KEY not set, skipping content safety")
        return True, ""

    filled = prompt.replace("{{ user_input }}", user_message)

    try:
        # Call NIM content safety model
        import requests
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
                "messages": [{"role": "user", "content": filled}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=15,
        )
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"[GUARD] Content safety input: {content[:100]}")

        # Parse NIM response — look for "unsafe" in the JSON output
        if "unsafe" in content.lower():
            return False, "Blocked by content safety"
    except Exception as e:
        logger.warning(f"[GUARD] Content safety check failed: {e}")

    return True, ""


def check_pii_input(user_message: str) -> tuple[bool, str]:
    """
    PII detection on input — calls GLiNER API using config from 05a.
    Returns (is_safe, reason).
    """
    prompts = _get_prompts()
    gliner_cfg = prompts["gliner"]
    if not gliner_cfg:
        return True, ""

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if not nvidia_key:
        return True, ""

    # Only check for hard identifiers on input — names cause false positives
    input_entities = ["email", "phone_number", "ssn"]
    threshold = 0.7  # Higher threshold to reduce false positives on input
    endpoint = gliner_cfg.get("server_endpoint", "")

    try:
        import requests
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/gliner-pii",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({
                            "input": user_message,
                            "entities": input_entities,
                            "threshold": threshold,
                        }),
                    }
                ],
            },
            timeout=15,
        )
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content) if isinstance(content, str) else content
        entities = parsed.get("entities", [])

        # Log full entity details for debugging
        if entities:
            for e in entities:
                logger.info(f"[GUARD] PII entity: value='{e.get('value','')}' "
                            f"label={e.get('suggested_label','')} "
                            f"score={e.get('score','?')}")

        # Only block on genuine PII — filter out low-confidence detections
        real_pii = [e for e in entities if float(e.get("score", 0)) >= threshold]
        if real_pii:
            labels = [e.get("suggested_label", "PII") for e in real_pii]
            values = [e.get("value", "?") for e in real_pii]
            logger.info(f"[GUARD] PII blocked on input: {list(zip(values, labels))}")
            return False, f"PII detected in your message ({', '.join(set(labels))}). Please remove personal information."
        elif entities:
            logger.info(f"[GUARD] PII entities found but below threshold, allowing through")
    except Exception as e:
        logger.warning(f"[GUARD] PII input check failed: {e}")

    return True, ""


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

def check_self_output(api_key: str, user_message: str, bot_response: str,
                      user_id: str = "USR-0042") -> tuple[bool, str]:
    """
    Self-check output rail — uses the prompt from config 02.
    Returns (is_safe, reason).
    """
    prompts = _get_prompts()
    prompt = prompts["self_check_output"]
    if not prompt:
        return True, ""

    filled = prompt.replace("{{ bot_response }}", bot_response)
    filled = filled.replace("{{ user_input }}", user_message)
    # Replace hardcoded USR-0042 with the actual logged-in user so the
    # check knows whose data is allowed to be shown
    filled = filled.replace("USR-0042", user_id)
    # Add explicit context so gpt-3.5-turbo doesn't over-block
    filled = (
        f"Context: The currently logged-in user is {user_id}. "
        f"Showing this user's OWN account details is allowed and should NOT be blocked.\n\n"
        + filled
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": filled}],
        temperature=0,
        max_tokens=10,
    )
    verdict = response.choices[0].message.content.strip().lower()
    logger.info(f"[GUARD] Self-check output verdict: {verdict}")

    if verdict.startswith("yes"):
        return False, "Blocked by output policy check"
    return True, ""


def mask_pii_output(bot_response: str) -> str:
    """
    PII masking on output — calls GLiNER API using config from 05b.
    Replaces detected PII with [LABEL] placeholders.
    Returns the masked response.
    """
    prompts = _get_prompts()
    gliner_cfg = prompts["gliner"]
    if not gliner_cfg:
        return bot_response

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if not nvidia_key:
        return bot_response

    output_entities = gliner_cfg.get("output", {}).get("entities", [])
    threshold = 0.7  # Higher threshold to avoid masking non-PII text
    endpoint = gliner_cfg.get("server_endpoint", "")

    try:
        import requests
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/gliner-pii",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({
                            "input": bot_response,
                            "entities": output_entities,
                            "threshold": threshold,
                        }),
                    }
                ],
            },
            timeout=15,
        )
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content) if isinstance(content, str) else content
        entities = parsed.get("entities", [])

        # Log raw entity dicts for debugging
        for e in entities:
            logger.info(f"[GUARD] PII output raw entity: {e}")

        # Only mask entities that have:
        # 1. High confidence score (>= threshold)
        # 2. A non-empty label (empty label = false positive)
        # 3. A non-empty value
        real_pii = [
            e for e in entities
            if float(e.get("score", 0)) >= threshold
            and e.get("suggested_label", "").strip()
            and e.get("value", "").strip()
        ]
        if real_pii:
            entities_sorted = sorted(real_pii, key=lambda e: e.get("start", 0), reverse=True)
            masked = bot_response
            for ent in entities_sorted:
                start = ent.get("start", 0)
                end = ent.get("end", 0)
                label = ent.get("suggested_label", "PII").upper()
                masked = masked[:start] + f"[{label}]" + masked[end:]
            logger.info(f"[GUARD] PII masked in output: {len(real_pii)} entities")
            return masked
        elif entities:
            logger.info(f"[GUARD] PII output entities found but filtered out (empty label/value or low score)")
    except Exception as e:
        logger.warning(f"[GUARD] PII output mask failed: {e}")

    return bot_response


def check_content_safety_output(user_message: str, bot_response: str) -> tuple[bool, str]:
    """
    Content safety output rail — calls NVIDIA NIM API using the prompt from config 04.
    Returns (is_safe, reason).
    """
    prompts = _get_prompts()
    prompt = prompts["content_safety_output"]
    if not prompt:
        return True, ""

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if not nvidia_key:
        return True, ""

    filled = prompt.replace("{{ user_input }}", user_message).replace("{{ bot_response }}", bot_response)

    try:
        import requests
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
                "messages": [{"role": "user", "content": filled}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=15,
        )
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"[GUARD] Content safety output: {content[:100]}")

        if "unsafe" in content.lower():
            return False, "Blocked by content safety"
    except Exception as e:
        logger.warning(f"[GUARD] Content safety output check failed: {e}")

    return True, ""


# ═══════════════════════════════════════════════════════════════════
#  GUARDED AGENT — Full pipeline
# ═══════════════════════════════════════════════════════════════════

def invoke_guarded_agent(conn, user_message: str, user_id: str,
                         account_tier: str, api_key: str,
                         chat_history: list[dict] = None) -> str:
    """
    Run the full guarded ARIA pipeline:
      1. Input guardrails (self-check, content safety, PII detect)
      2. ARIA agent (OpenAI function calling)
      3. Output guardrails (self-check, content safety, PII mask)
    """
    logger.info(f"[PIPELINE] Starting guarded pipeline for: {user_message[:80]}")

    # ── Step 1: Input Guardrails ──

    # 1a. Self-check input (config 01)
    safe, reason = check_self_input(api_key, user_message)
    if not safe:
        logger.info(f"[PIPELINE] Blocked by self-check input: {reason}")
        return BLOCKED_RESPONSE

    # 1b. Content safety input (config 04)
    safe, reason = check_content_safety_input(user_message)
    if not safe:
        logger.info(f"[PIPELINE] Blocked by content safety input: {reason}")
        return BLOCKED_RESPONSE

    # 1c. PII detection on input (config 05a)
    safe, reason = check_pii_input(user_message)
    if not safe:
        logger.info(f"[PIPELINE] Blocked by PII input: {reason}")
        return reason  # PII block has a specific user-friendly message

    # ── Step 2: Run ARIA Agent ──
    logger.info("[PIPELINE] Input passed all guardrails, running agent")
    agent_response = aria_agent(
        conn=conn,
        user_message=user_message,
        user_id=user_id,
        account_tier=account_tier,
        api_key=api_key,
        chat_history=chat_history,
    )

    # ── Step 3: Output Guardrails ──

    # 3a. Self-check output (config 02)
    safe, reason = check_self_output(api_key, user_message, agent_response, user_id=user_id)
    if not safe:
        logger.info(f"[PIPELINE] Blocked by self-check output: {reason}")
        return BLOCKED_RESPONSE

    # 3b. Content safety output (config 04)
    safe, reason = check_content_safety_output(user_message, agent_response)
    if not safe:
        logger.info(f"[PIPELINE] Blocked by content safety output: {reason}")
        return BLOCKED_RESPONSE

    # 3c. PII masking on output (config 05b)
    final_response = mask_pii_output(agent_response)

    logger.info("[PIPELINE] Response passed all guardrails")
    return final_response
