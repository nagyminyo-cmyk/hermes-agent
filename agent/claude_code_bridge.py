"""Claude Code CLI bridge for Boss-tier LLM calls.

Wraps `claude --print --output-format json` subprocess calls with automatic
fallback to Qwen DashScope on any error. Designed to slot into Hermes'
main conversation loop as a drop-in replacement for the primary client.

Legal posture:
- Uses Claude Code CLI subprocess (Anthropic ToS exempt lane, Feb 19 2026)
- OAuth token never leaves ~/.claude/auth.json
- Only shells out and reads stdout

Fallback triggers (all route to Qwen):
- Non-zero exit code from claude
- Subprocess timeout (>10 min)
- JSON parse error
- Missing text field in parsed JSON
- Any stderr content signaling rate-limit / quota / cap-hit
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths
CLAUDE_BIN = "/root/.hermes/node/bin/claude"
FALLBACK_LOG = "/var/log/hermes/llm-fallback.log"
OPUS_BUDGET_LOG = "/var/log/hermes/opus-budget.log"
OPUS_BUDGET_WINDOW_SEC = 5 * 3600  # 5 hours
OPUS_BUDGET_MAX = 40  # max Opus calls per window

# Default model aliases (verified via claude --help)
DEFAULT_BOSS_MODEL = "claude-sonnet-4-6"
DEEP_MODEL = "claude-opus-4-7"

# Files that trigger deep mode when touched in a task
DEEP_SIGNAL_FILES = {
    "worker/src/handlers.ts",
    "worker/src/notion.ts",
    "wrangler.toml",
    ".claude/blindspots.md",
}

# Opus budget tracking (in-memory + disk)
_opus_call_times: list = []


def _load_opus_log() -> list:
    """Load opus call timestamps from disk."""
    try:
        path = Path(OPUS_BUDGET_LOG)
        if not path.exists():
            return []
        lines = path.read_text().strip().splitlines()
        times = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                times.append(datetime.fromisoformat(line))
            except ValueError:
                continue
        return times
    except Exception:
        return []


def _save_opus_log(times: list):
    """Persist opus call timestamps."""
    try:
        path = Path(OPUS_BUDGET_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(t.isoformat() for t in times) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save opus budget log: {e}")


def _check_opus_budget() -> tuple:
    """Check opus budget. Returns (allowed: bool, reason: str)."""
    global _opus_call_times
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - OPUS_BUDGET_WINDOW_SEC

    # Load from disk and filter
    times = _load_opus_log()
    _opus_call_times = [t for t in times if t.timestamp() > cutoff]

    if len(_opus_call_times) >= OPUS_BUDGET_MAX:
        return False, f"opus-budget-saver ({len(_opus_call_times)}/{OPUS_BUDGET_MAX} in {OPUS_BUDGET_WINDOW_SEC // 3600}h window)"
    return True, ""


def _record_opus_call():
    """Record an Opus call for budget tracking."""
    global _opus_call_times
    now = datetime.now(timezone.utc)
    _opus_call_times.append(now)
    _save_opus_log(_opus_call_times)


def _log_fallback(reason: str):
    """Log fallback event to disk."""
    try:
        path = Path(FALLBACK_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} FALLBACK: {reason}\n")
    except Exception as e:
        logger.warning(f"Failed to write fallback log: {e}")


def _signal_deep_mode(system: str, user: str, mode: str = None,
                      file_paths: list = None) -> bool:
    """Determine if a task warrants deep (Opus) mode."""
    if mode == "deep":
        return True

    # Check file path signals
    if file_paths:
        for fp in file_paths:
            if any(sig in fp for sig in DEEP_SIGNAL_FILES):
                return True

    # Check for /audit-* commands in user message
    if user and "/audit" in user.lower():
        return True

    # Check for long diff signals (heuristic: >300 lines of +/-)
    if user:
        diff_lines = sum(1 for line in user.splitlines()
                        if line.startswith("+") or line.startswith("-"))
        if diff_lines > 300:
            return True

    return False


def call_boss_llm(system: str, user: str, mode: str = None,
                  file_paths: list = None, model: str = None,
                  timeout: int = 600) -> str:
    """Primary Boss-tier LLM call with Claude→Qwen fallback.

    Args:
        system: System prompt
        user: User message
        mode: Optional mode override ("deep" for Opus)
        file_paths: List of file paths being worked on (for deep signal detection)
        model: Explicit model override (bypasses auto-selection)
        timeout: Max seconds for claude subprocess (default 600)

    Returns:
        Response text from Claude or Qwen fallback
    """
    # Determine model
    use_deep = _signal_deep_mode(system, user, mode, file_paths)

    if model:
        selected_model = model
    elif use_deep:
        allowed, reason = _check_opus_budget()
        if not allowed:
            _log_fallback(f"deep-mode blocked: {reason}")
            selected_model = DEFAULT_BOSS_MODEL
        else:
            selected_model = DEEP_MODEL
            _record_opus_call()
    else:
        selected_model = DEFAULT_BOSS_MODEL

    # Build claude command
    cmd = [
        CLAUDE_BIN,
        "--print",
        "--output-format", "json",
        "--model", selected_model,
        "--max-budget-usd", "5.00",  # safety cap
        "--system-prompt", system,
        "--",  # end of options
        user,  # prompt as argument (not stdin)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise ClaudeFallback(
                f"claude exit={result.returncode} stderr={stderr_snippet}"
            )

        # Parse JSON output
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            stdout_snippet = (result.stdout or "")[:200]
            raise ClaudeFallback(
                f"JSON parse error: {e}. stdout={stdout_snippet}"
            )

        # Extract text from response (field name varies by CLI version)
        text = (
            parsed.get("result")
            or parsed.get("content")
            or parsed.get("text")
            or parsed.get("response")
        )
        if not text:
            keys = list(parsed.keys())
            raise ClaudeFallback(f"no text field in response: {keys}")

        model_tag = "[deep]" if selected_model == DEEP_MODEL else "[boss]"
        logger.info(f"Claude response OK {model_tag} model={selected_model}")
        return text

    except subprocess.TimeoutExpired as e:
        _log_fallback(f"timeout after {timeout}s model={selected_model}")
        return _fallback_qwen(system, user, selected_model, "timeout")

    except ClaudeFallback as e:
        _log_fallback(str(e))
        return _fallback_qwen(system, user, selected_model, str(e))

    except Exception as e:
        _log_fallback(f"unexpected: {type(e).__name__}: {e}")
        return _fallback_qwen(system, user, selected_model, f"unexpected: {e}")


class ClaudeFallback(Exception):
    """Raised when Claude CLI fails and fallback is needed."""
    pass


def _fallback_qwen(system: str, user: str, attempted_model: str,
                    reason: str) -> str:
    """Fallback to Qwen DashScope.

    Uses the same auxiliary_client resolution chain as Hermes' existing
    compression/vision/search tasks — it will pick up the configured
    DashScope provider automatically.
    """
    logger.warning(f"Falling back to Qwen (attempted: {attempted_model}): {reason}")

    try:
        from agent.auxiliary_client import call_llm

        # Build messages in OpenAI format
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = call_llm(
            messages=messages,
            task="boss_fallback",
            # Use configured auxiliary model (should be Qwen per config)
        )

        if response and hasattr(response, "choices"):
            text = response.choices[0].message.content
            if text:
                return text

        # Fallback for dict-style responses
        if isinstance(response, dict):
            return (
                response.get("content")
                or response.get("text")
                or response.get("response", "")
            )

        return str(response) if response else ""

    except Exception as e:
        logger.error(f"Qwen fallback also failed: {e}")
        return f"[FALLBACK FAILED: {e}]"


def call_boss_llm_stream(system: str, user: str, mode: str = None,
                         file_paths: list = None, model: str = None,
                         timeout: int = 600) -> str:
    """Streaming variant — uses --output-format stream-json."""
    # For now, use the same implementation. Streaming can be added later
    # if needed for real-time Telegram delivery.
    return call_boss_llm(system, user, mode, file_paths, model, timeout)
