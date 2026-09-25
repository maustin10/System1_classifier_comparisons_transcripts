#!/usr/bin/env python3
"""TypeSafe System One (Jev) scorer — hosted counterpart to the local NLI POC.

WHY THIS EXISTS
`zero_shot_decision_poc.py` runs a local NLI cross-encoder (ModernBERT) and
derives probabilities itself. TypeSafe's System One exposes the same *shape* of
capability as a hosted API: typed questions over a shared state, returning
calibrated probabilities rather than generated text. Running the identical twelve
questions through both isolates what changes when only the scorer changes.

CONTRACT (POST https://api.typesafe.ai/v1/systemone)
    Authorization: Bearer <key>
    {
      "model": "jev-latest",
      "state": <string | object | array>,        # the facts to judge
      "questions": {                              # 1..64 questions
        "<id>": {
          "type": "choice" | "score" | "noul",
          "instructions": "<what to decide>",
          "criteria": {...} | [...]              # required for choice/score
        }
      }
    }

A `choice` answer returns:
    {"type":"choice","choice":"<option>","probabilities":{...},"confidence":0..1}

Note the deliberate parallel to the local POC: `probabilities` maps option ->
float summing to 1, exactly like the local softmax output, so the two are
directly comparable. `confidence` is TypeSafe's own concentration measure and is
NOT the same quantity as the local POC's top-probability, so this module keeps
both and never conflates them.

USAGE
    # Show the exact payloads without calling the API (works offline):
    python3 poc/typesafe_decision_poc.py --dry-run

    # Score all twelve questions against the live API:
    python3 poc/typesafe_decision_poc.py --json > results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# Documented server-side validation limits. Checked locally so an oversized
# request fails immediately with a precise message rather than costing a
# round-trip to be rejected with a 422.
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 578
MAX_STATE_CHARS = 100_000
MAX_INSTRUCTION_CHARS = 20_000
MAX_ID_CHARS = 200

# The local POC's gate, reused verbatim so "would this have been answered?" means
# the same thing for both scorers.
MIN_PROBABILITY = 0.65
MIN_MARGIN = 0.15


class TypeSafeError(RuntimeError):
    """Raised for configuration, transport, and API-reported failures."""


def _load_default_cases() -> list[dict[str, Any]]:
    """Load the original POC cases only when the standalone CLI needs them.

    Transport helpers in this module are reused by other benchmarks, which
    should not fail merely because the original twelve-case fixture is absent.
    """
    try:
        from obvious_answers_benchmark import CASES
    except ModuleNotFoundError as error:
        raise TypeSafeError(
            "The original obvious_answers_benchmark fixture is not installed. "
            "Use a benchmark-specific runner or restore that module next to this script."
        ) from error
    return list(CASES)


def _resolve_api_key(explicit: str | None = None) -> str:
    """Find the API key, tolerating both plausible variable names.

    This repo's .env uses TYPESAFE_API; TypeSafe's own SDKs document
    TYPESAFE_API_KEY. Accepting both avoids a silent "unauthorized" that is
    really just a naming mismatch.
    """
    if explicit:
        return explicit.strip()
    for name in ("TYPESAFE_API", "TYPESAFE_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    raise TypeSafeError(
        "No TypeSafe API key found. Set TYPESAFE_API (or TYPESAFE_API_KEY) in the "
        "environment or the repo-root .env, or pass --api-key."
    )


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so this script has no hard dependency on python-dotenv.

    Existing environment variables win, matching python-dotenv's default and
    keeping an explicit shell override authoritative.
    """
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def build_question(case: dict[str, Any]) -> dict[str, Any]:
    """Map one benchmark case onto a System One `choice` question.

    The option DESCRIPTIONS are sent as criteria, not just the ids. This mirrors
    the local POC's hypothesis construction: an id like "positive" carries almost
    no signal on its own, and keeping both scorers on identical wording is what
    makes the comparison meaningful.
    """
    return {
        "type": "choice",
        "instructions": case["question"],
        "criteria": dict(case["options"]),
    }


def build_payload(cases: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Build one request per case.

    System One accepts up to 64 questions in a single call, but each request has
    ONE shared `state`. These twelve cases each have their own state, so they
    cannot be batched into a single request without leaking one case's facts into
    another's premise -- which would invalidate the comparison.
    """
    payloads = []
    for case in cases:
        state = case["state"]
        state_size = len(json.dumps(state))
        if state_size > MAX_STATE_CHARS:
            raise TypeSafeError(
                f"case {case['id']!r}: state is {state_size} chars, over the "
                f"{MAX_STATE_CHARS} limit"
            )
        if len(case["id"]) > MAX_ID_CHARS:
            raise TypeSafeError(f"case id {case['id']!r} exceeds {MAX_ID_CHARS} chars")
        if len(case["options"]) > MAX_CHOICE_OPTIONS:
            raise TypeSafeError(f"case {case['id']!r}: too many options")
        if len(case["question"]) > MAX_INSTRUCTION_CHARS:
            raise TypeSafeError(f"case {case['id']!r}: instructions too long")

        payloads.append(
            {
                "model": model,
                "state": state,
                "questions": {case["id"]: build_question(case)},
            }
        )
    return payloads


def call_api(
    payload: dict[str, Any],
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 90.0,
    max_retries: int = 4,
) -> dict[str, Any]:
    """POST one request and return the parsed response.

    urllib is used rather than requests/httpx so the script runs in any of this
    repo's virtualenvs without extra installs. It also honours the standard
    HTTPS_PROXY / SSL_CERT_FILE environment variables, which is mandatory on this
    network.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                content_type = response.headers.get("Content-Type", "")
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:600]
            if error.code == 401:
                raise TypeSafeError(
                    "401 Unauthorized -- the API key was rejected. Check TYPESAFE_API "
                    f"is the full key and has not expired.\n{detail}"
                ) from error
            if error.code == 422:
                raise TypeSafeError(
                    f"422 Validation error -- the request shape was rejected:\n{detail}"
                ) from error
            if error.code not in (429, 529) or attempt >= max_retries:
                raise TypeSafeError(f"HTTP {error.code} from TypeSafe:\n{detail}") from error
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2**attempt, 16)
            except ValueError:
                delay = min(2**attempt, 16)
            time.sleep(max(0.0, delay))
        except urllib.error.URLError as error:
            raise TypeSafeError(
                f"Could not reach {endpoint}: {error.reason}\n\n"
                "On the AT&T corporate network, api.typesafe.ai is blocked by the CSO "
                "proxy (requests are 302-redirected to a block page). Verify with:\n"
                "    curl -sI https://api.typesafe.ai/\n"
                "If the Location header points at blockpage.cgi, the domain needs to be "
                "allow-listed before this can run on-network."
            ) from error

    # A filtering proxy answers with 200 + HTML rather than failing outright, so a
    # non-JSON body must be reported as interception, not as malformed API output.
    if "json" not in content_type.lower() or raw.lstrip()[:1] not in "{[":
        raise TypeSafeError(
            f"Expected JSON from {endpoint} but received {content_type or 'unknown'}. "
            "This is the signature of a proxy block/interstitial page rather than an "
            f"API response.\nFirst 200 bytes: {raw[:200]!r}"
        )
    return json.loads(raw)


def summarise_answer(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Normalise one API response into the same record shape the local POC emits."""
    answers = response.get("answers") or {}
    answer = answers.get(case["id"])
    if answer is None:
        raise TypeSafeError(
            f"response contained no answer for question id {case['id']!r}; "
            f"got keys {list(answers)}"
        )

    probabilities: dict[str, float] = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    top_probability = ranked[0][1] if ranked else 0.0
    margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
    picked = answer.get("choice") or (ranked[0][0] if ranked else None)

    return {
        "id": case["id"],
        "reasoning": case["reasoning"],
        "state": case["state"],
        "question": case["question"],
        "options": case["options"],
        "expected": case["expected"],
        "picked": picked,
        "correct": picked == case["expected"],
        "probabilities": probabilities,
        "top_probability": top_probability,
        "margin": margin,
        # TypeSafe's own concentration metric -- reported alongside, never
        # substituted for top_probability, because they measure different things.
        "api_confidence": answer.get("confidence"),
        # Same gate as the local POC, so "would this be answered?" is comparable.
        "review_required": top_probability < MIN_PROBABILITY or margin < MIN_MARGIN,
        "model": response.get("model"),
        "usage": response.get("usage"),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    correct = sum(1 for row in rows if row["correct"])
    flagged = sum(1 for row in rows if row["review_required"])
    print()
    print(f"cases: {len(rows)}   correct: {correct}/{len(rows)}   flagged: {flagged}")
    print()
    header = (
        f"{'':2} {'case':20} {'reasoning':22} {'picked':12} "
        f"{'top_p':>7} {'margin':>7} {'api_conf':>9}  review"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mark = "OK" if row["correct"] else "XX"
        confidence = row["api_confidence"]
        confidence_text = f"{confidence:9.4f}" if isinstance(confidence, (int, float)) else f"{'n/a':>9}"
        print(
            f"{mark:2} {row['id']:20} {row['reasoning']:22} {str(row['picked']):12} "
            f"{row['top_probability']:7.4f} {row['margin']:7.4f} {confidence_text}  "
            f"{'yes' if row['review_required'] else '-'}"
        )

    print()
    print("full probability distributions:")
    for row in rows:
        ordered = sorted(row["probabilities"].items(), key=lambda item: -item[1])
        rendered = "  ".join(f"{name}={value:.4f}" for name, value in ordered)
        print(f"  {row['id']:20} {rendered}")

    tokens = sum((row.get("usage") or {}).get("input_tokens", 0) for row in rows)
    if tokens:
        print()
        print(f"total input tokens billed: {tokens}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help="Overrides TYPESAFE_API.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact request payloads and exit without calling the API.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this path (implies --json).",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help=(
            "Write the exact request/response JSON for each case to this path. "
            "The API key is never included."
        ),
    )
    parser.add_argument("--only", default=None, help="Run a single case id.")
    arguments = parser.parse_args()

    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    cases = [c for c in _load_default_cases() if arguments.only in (None, c["id"])]
    if not cases:
        raise TypeSafeError(f"no case matches --only {arguments.only!r}")
    if len(cases) > MAX_QUESTIONS:
        raise TypeSafeError(f"{len(cases)} cases exceeds the {MAX_QUESTIONS}-question limit")

    payloads = build_payload(cases, arguments.model)

    if arguments.dry_run:
        # Deliberately offline: proves the request shape is correct even while the
        # domain is blocked, and lets the payload be reviewed before spending money.
        print(json.dumps({"endpoint": arguments.endpoint, "requests": payloads}, indent=2))
        return 0

    api_key = _resolve_api_key(arguments.api_key)
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for case, payload in zip(cases, payloads, strict=True):
        response = call_api(
            payload,
            api_key,
            arguments.endpoint,
            arguments.timeout,
            arguments.max_retries,
        )
        if arguments.raw_output:
            raw_records.append(
                {
                    "id": case["id"],
                    "request": payload,
                    "response": response,
                }
            )
        rows.append(summarise_answer(case, response))

    if arguments.raw_output:
        raw_rendered = json.dumps(
            {
                "endpoint": arguments.endpoint,
                "requested_model": arguments.model,
                "cases": raw_records,
            },
            indent=2,
        ) + "\n"
        arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.raw_output.write_text(raw_rendered)
        print(f"wrote {len(raw_records)} raw TypeSafe responses to {arguments.raw_output}")

    if arguments.json or arguments.output:
        rendered = json.dumps({"cases": rows}, indent=2) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered)
            print(f"wrote {len(rows)} TypeSafe results to {arguments.output}")
        else:
            print(rendered, end="")
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TypeSafeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
