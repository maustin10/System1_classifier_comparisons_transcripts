#!/usr/bin/env python3
"""Evaluate TypeSafe.ai JEV Noul questions on the locked synthetic benchmark."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "benchmark_1000"
TYPESAFE_ENV = ROOT / ".env"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_classifiers_shared import ATTRIBUTE_DESCRIPTIONS  # noqa: E402
from typesafe_decision_poc import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    _load_dotenv,
    _resolve_api_key,
    call_api,
)


BOUNDARIES: dict[str, tuple[str, str]] = {
    "reset_equipment_mentioned": (
        "The caller or agent explicitly mentions resetting, restarting, or rebooting equipment.",
        "No reset, restart, or reboot is mentioned; generic troubleshooting does not count.",
    ),
    "unable_to_work_due_to_service_issues": (
        "The caller explicitly says the service problem prevented or left them unable to work.",
        "The caller is inconvenienced or has poor service but does not say they are unable to work.",
    ),
    "lagging_issues": (
        "The caller explicitly reports lag, latency, delayed sessions, or video calls that lag.",
        "The caller reports a different problem such as low throughput, disconnection, or an error without lag.",
    ),
    "poor_performance_complaint": (
        "The caller directly uses the concept of performance and explicitly says performance is poor, bad, inadequate, awful, or unacceptable.",
        "Do not infer this from poor service quality, unreliability, lag, low throughput, glitching, errors, or other symptoms when the caller does not directly characterize performance itself.",
    ),
    "glitching_issue": (
        "The caller explicitly reports glitching, screen jumping, or erratic application behavior.",
        "The caller reports lag, an error, disconnection, or another failure without glitching behavior.",
    ),
    "technician_visits_mentioned": (
        "A past, current, or planned technician visit is explicitly mentioned.",
        "No technician visit is mentioned.",
    ),
    "poor_service_quality_complaint": (
        "The caller directly uses the concept of service quality or reliability and explicitly says it is poor, bad, unreliable, or unacceptable.",
        "Do not infer this from poor performance, lag, low throughput, inability to connect, errors, or other symptoms when the caller does not directly criticize service quality or reliability.",
    ),
    "throughput_issues_mentioned": (
        "The caller explicitly reports low throughput, slow download speed, or a measured Mbps problem.",
        "The caller reports lag or generic poor performance without throughput or download-speed evidence.",
    ),
    "receiving_error_messages": (
        "The caller explicitly says they keep receiving an error or failure message such as Connection failed, Service unavailable, Profile unavailable, or Device not found.",
        "A SIM memory full notification belongs only to the separate SIM-memory attribute for this taxonomy. Other failures without an explicitly received error or failure message are false.",
    ),
    "remove_phone_lines_requested": (
        "The caller explicitly asks to remove, cancel, or disconnect a phone line.",
        "A phone line is discussed without a request to remove it.",
    ),
    "phone_purchase_upgrade_requested": (
        "The caller explicitly requests or expresses interest in purchasing or upgrading a phone.",
        "The agent offers a phone or plan but the caller does not request or express interest in a phone purchase or upgrade.",
    ),
    "issue_inside_building_mentioned": (
        "The conversation explicitly says the issue occurs inside a building, office, or indoor location.",
        "No inside or indoor location is stated.",
    ),
    "issue_outside_building_mentioned": (
        "The conversation explicitly says the issue occurs outside a building or involves an outdoor line or location.",
        "No outside or outdoor location is stated.",
    ),
    "sim_memory_issues_mentioned": (
        "A SIM memory problem or a SIM memory full message is explicitly mentioned.",
        "A phone or SIM problem is mentioned without a SIM memory issue.",
    ),
    "app_issue_mentioned": (
        "The caller explicitly says an application or app is malfunctioning, unavailable, glitching, or not working.",
        "A device, service, or Smart Home system issue is reported without an explicit application problem.",
    ),
    "unknown_charges_on_bill": (
        "The caller explicitly reports an unknown, unfamiliar, unexplained, or unrecognized bill charge.",
        "Billing is discussed without an unknown or unrecognized charge.",
    ),
    "credit_requested": (
        "The caller explicitly asks for a billing or service credit, refund, or account adjustment.",
        "A credit is discussed or applied without the caller requesting it, or no credit request occurs.",
    ),
    "credit_applied_by_agent": (
        "The agent explicitly confirms that a credit was applied, posted, or completed on the account.",
        "A credit is only requested, offered, documented, or submitted for review and has not been applied.",
    ),
    "phone_service_issue": (
        "The caller explicitly reports malfunctioning mobile or phone service, calling problems, or a SIM-related phone-service problem.",
        "A phone purchase, line removal, or account request occurs without a phone-service malfunction.",
    ),
    "smart_home_manager_issue": (
        "The caller explicitly reports a problem with Smart Home Manager or its connected-device controls.",
        "A generic internet, app, or connected-device issue occurs without Smart Home Manager being identified.",
    ),
    "sales_attempt_by_agent": (
        "The agent proactively offers, promotes, or attempts to add a paid product, service, plan, or protection package.",
        "The agent only handles the caller's existing request and makes no optional sales or upsell offer.",
    ),
    "agent_attempted_resolution": (
        "The agent takes or initiates a concrete troubleshooting, account, billing, provisioning, investigation, or resolution action.",
        "The agent only records or discusses the issue and explicitly makes no troubleshooting or account change.",
    ),
    "multiple_contacts_same_issue": (
        "The caller explicitly says this is a repeated, second, third, or later contact about the same issue.",
        "This is the first contact or no repeated contact is stated.",
    ),
    "fiber_line_issue_mentioned": (
        "A damaged, faulty, or suspected-problem fiber line is explicitly mentioned.",
        "Fiber service or internet is discussed without a fiber-line problem.",
    ),
    "unable_to_activate_internet": (
        "The caller explicitly says they cannot or have been unable to activate internet service.",
        "Internet service is active but has performance, connection, or quality problems.",
    ),
    "unable_to_connect_multiple_devices": (
        "The caller explicitly says multiple devices cannot connect or cannot remain connected at the same time.",
        "Only one device is affected, or devices are mentioned without a multiple-device connection failure.",
    ),
    "resolution_time_concern": (
        "The caller explicitly asks how long resolution will take or expresses concern about the resolution timeline.",
        "A timeline is volunteered by the agent without the caller asking or expressing concern.",
    ),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def structured_state(conversation: str) -> dict[str, Any]:
    messages = []
    for raw in conversation.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[Agent]:"):
            messages.append({"speaker": "agent", "text": line[len("[Agent]:"):].strip()})
        elif line.startswith("[Caller]:"):
            messages.append({"speaker": "caller", "text": line[len("[Caller]:"):].strip()})
        else:
            messages.append({"speaker": "unknown", "text": line})
    return {"conversation_type": "customer_care_call", "messages": messages}


def payload(row: dict[str, Any]) -> dict[str, Any]:
    questions = {}
    for attribute, description in ATTRIBUTE_DESCRIPTIONS.items():
        true_rule, false_rule = BOUNDARIES[attribute]
        questions[attribute] = {
            "type": "noul",
            "instructions": f"Is this attribute explicitly present in the conversation: {description}?",
            "criteria": {"true": true_rule, "false": false_rule},
        }
    return {"model": DEFAULT_MODEL, "state": structured_state(row["conversation"]), "questions": questions}


def run(part: str) -> None:
    split = read(WORK / "split.json")
    rows = [row for row in split["rows"] if row["split"] == part]
    output_path = WORK / f"jev_noul_{part}_results.json"
    output = read(output_path) if output_path.exists() else {"metadata": {}, "rows": {}}
    _load_dotenv(TYPESAFE_ENV)
    api_key = _resolve_api_key()
    started = time.perf_counter()
    initial = len(output["rows"])
    for index, row in enumerate(rows, start=1):
        key = str(row["id"])
        if key in output["rows"]:
            continue
        response = call_api(payload(row), api_key, timeout=120.0, max_retries=5)
        answers = response.get("answers") or {}
        probabilities = {}
        for attribute in ATTRIBUTE_DESCRIPTIONS:
            answer = answers.get(attribute)
            if not answer or "noul" not in answer:
                raise RuntimeError(f"Missing Noul answer for {row['id']} / {attribute}")
            probabilities[attribute] = round(float(answer["noul"]), 8)
        output["rows"][key] = {"probabilities": probabilities, "model": response.get("model")}
        write(output_path, output)
        print(f"JEV Noul {part} {index}/{len(rows)} row {row['id']}", flush=True)
    elapsed = time.perf_counter() - started
    output["metadata"] = {
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "part": part,
        "questions": len(ATTRIBUTE_DESCRIPTIONS),
        "rows": len(rows),
        "new_calls_current_run": len(output["rows"]) - initial,
        "elapsed_seconds_current_run": elapsed,
        "prompt": "Structured speaker state plus strict attribute-specific true/false criteria",
    }
    write(output_path, output)
    print(json.dumps(output["metadata"], indent=2))


def refine_validation() -> None:
    """Re-score only the three ambiguous labels after validation prompt review."""
    split = read(WORK / "split.json")
    rows = [row for row in split["rows"] if row["split"] == "validation"]
    source = read(WORK / "jev_noul_validation_results.json")
    output_path = WORK / "jev_noul_validation_v2_results.json"
    output = read(output_path) if output_path.exists() else json.loads(json.dumps(source))
    target_attributes = (
        "poor_performance_complaint",
        "poor_service_quality_complaint",
        "receiving_error_messages",
    )
    _load_dotenv(TYPESAFE_ENV)
    api_key = _resolve_api_key()
    started = time.perf_counter()
    completed = set(output.get("metadata", {}).get("refined_row_ids", []))
    for index, row in enumerate(rows, start=1):
        key = str(row["id"])
        if row["id"] in completed:
            continue
        questions = {}
        for attribute in target_attributes:
            description = ATTRIBUTE_DESCRIPTIONS[attribute]
            true_rule, false_rule = BOUNDARIES[attribute]
            questions[attribute] = {
                "type": "noul",
                "instructions": f"Is this attribute explicitly present in the conversation: {description}?",
                "criteria": {"true": true_rule, "false": false_rule},
            }
        response = call_api(
            {"model": DEFAULT_MODEL, "state": structured_state(row["conversation"]), "questions": questions},
            api_key,
            timeout=120.0,
            max_retries=5,
        )
        for attribute in target_attributes:
            output["rows"][key]["probabilities"][attribute] = round(
                float(response["answers"][attribute]["noul"]), 8
            )
        completed.add(row["id"])
        output["metadata"]["refined_row_ids"] = sorted(completed)
        write(output_path, output)
        print(f"JEV Noul validation-v2 {index}/{len(rows)} row {row['id']}", flush=True)
    output["metadata"].update({
        "part": "validation",
        "variant": "v2",
        "refined_attributes": list(target_attributes),
        "elapsed_seconds_current_run": time.perf_counter() - started,
    })
    write(output_path, output)
    print(json.dumps(output["metadata"], indent=2))


def metrics(truth: list[list[int]], prediction: list[list[int]]) -> dict[str, float | int]:
    pairs = [(y, p) for yr, pr in zip(truth, prediction) for y, p in zip(yr, pr)]
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(yr == pr for yr, pr in zip(truth, prediction)) / len(truth)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(pairs), "precision": precision,
        "recall": recall, "f1": f1, "exact_match_rate": exact,
    }


def best_threshold(y: list[int], probabilities: list[float]) -> float:
    best: tuple[float, float, float, float] | None = None
    selected = 0.5
    for step in range(5, 96):
        threshold = step / 100
        pred = [int(value >= threshold) for value in probabilities]
        current = metrics([y], [pred])
        candidate = (
            float(current["f1"]),
            float(current["precision"]),
            -abs(threshold - 0.5),
            threshold,
        )
        if best is None or candidate > best:
            best = candidate
            selected = threshold
    return selected


def evaluate() -> None:
    split = read(WORK / "split.json")
    attributes = split["metadata"]["attributes"]
    validation_rows = [row for row in split["rows"] if row["split"] == "validation"]
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    validation_path = WORK / "jev_noul_validation_v2_results.json"
    validation = read(validation_path if validation_path.exists() else WORK / "jev_noul_validation_results.json")
    test = read(WORK / "jev_noul_test_results.json")
    y_validation = [[int(row["truth"][a]) for a in attributes] for row in validation_rows]
    y_test = [[int(row["truth"][a]) for a in attributes] for row in test_rows]
    p_validation = [[validation["rows"][str(row["id"])]["probabilities"][a] for a in attributes] for row in validation_rows]
    p_test = [[test["rows"][str(row["id"])]["probabilities"][a] for a in attributes] for row in test_rows]
    thresholds = []
    for col in range(len(attributes)):
        thresholds.append(best_threshold(
            [row[col] for row in y_validation],
            [row[col] for row in p_validation],
        ))
    default_validation = [[int(value >= 0.5) for value in row] for row in p_validation]
    tuned_validation = [[int(value >= thresholds[col]) for col, value in enumerate(row)] for row in p_validation]
    default_test = [[int(value >= 0.5) for value in row] for row in p_test]
    tuned_test = [[int(value >= thresholds[col]) for col, value in enumerate(row)] for row in p_test]
    trained = read(WORK / "comparison_metrics.json")["models"]["modernbert_trained"]
    result = {
        "method": "JEV Noul with structured speaker state, strict criteria, and per-attribute F1 thresholds chosen on validation only",
        "thresholds": dict(zip(attributes, thresholds)),
        "threshold_summary": {
            "min": min(thresholds),
            "median": sorted(thresholds)[len(thresholds) // 2],
            "mean": sum(thresholds) / len(thresholds),
            "max": max(thresholds),
        },
        "jev_noul_default_0_5": {
            "validation": metrics(y_validation, default_validation),
            "test": metrics(y_test, default_test),
        },
        "jev_noul_validation_tuned": {
            "validation": metrics(y_validation, tuned_validation),
            "test": metrics(y_test, tuned_test),
        },
        "modernbert_trained_test": {
            **trained["micro"],
            "exact_match_rate": trained["exact_match_rate"],
        },
    }
    write(WORK / "jev_noul_comparison.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "validation-v2", "test", "evaluate"))
    args = parser.parse_args()
    if args.stage in {"validation", "test"}:
        run(args.stage)
    elif args.stage == "validation-v2":
        refine_validation()
    else:
        evaluate()


if __name__ == "__main__":
    main()
