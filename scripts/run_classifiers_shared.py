"""Shared attribute wording for synthetic transcript classifiers."""
from __future__ import annotations

from typing import Any

ATTRIBUTE_DESCRIPTIONS: dict[str, str] = {
    "reset_equipment_mentioned": "the caller mentions resetting or rebooting equipment",
    "unable_to_work_due_to_service_issues": "the caller says service problems made them unable to work",
    "lagging_issues": "the caller reports lagging, latency, or sessions that lag",
    "poor_performance_complaint": "the caller complains about poor or inadequate performance",
    "glitching_issue": "the caller reports a screen, app, or service glitching",
    "technician_visits_mentioned": "a technician visit is mentioned",
    "poor_service_quality_complaint": "the caller complains about poor service quality",
    "throughput_issues_mentioned": "the caller reports low throughput or slow measured download speed",
    "receiving_error_messages": "the caller reports receiving an error message",
    "remove_phone_lines_requested": "the caller asks to remove a phone line",
    "phone_purchase_upgrade_requested": "the caller asks to purchase or upgrade a phone",
    "issue_inside_building_mentioned": "the caller says the issue occurs inside a building",
    "issue_outside_building_mentioned": "the caller says the issue occurs outside a building",
    "sim_memory_issues_mentioned": "a SIM memory problem or SIM memory full message is mentioned",
    "app_issue_mentioned": "the caller reports an issue with an app",
    "unknown_charges_on_bill": "the caller reports an unknown or unfamiliar charge on the bill",
    "credit_requested": "the caller requests a bill credit or service credit",
    "credit_applied_by_agent": "the agent says that a credit was applied to the account",
    "phone_service_issue": "the caller reports that phone service is not working correctly",
    "smart_home_manager_issue": "the caller reports a Smart Home Manager problem",
    "sales_attempt_by_agent": "the agent attempts to sell, offer, or upsell a product or service",
    "agent_attempted_resolution": "the agent takes troubleshooting or resolution steps during the conversation",
    "multiple_contacts_same_issue": "the caller says they contacted support multiple times about the same issue",
    "fiber_line_issue_mentioned": "a problem or possible damage involving the fiber line is mentioned",
    "unable_to_activate_internet": "the caller says they are unable to activate internet service",
    "unable_to_connect_multiple_devices": "the caller says multiple devices cannot connect at the same time",
    "resolution_time_concern": "the caller asks about or expresses concern about how long resolution will take",
}


def build_jev_payload(row: dict[str, Any], model: str = "jev-latest") -> dict[str, Any]:
    questions = {
        attribute: {
            "type": "choice",
            "instructions": f"Is this attribute present in the conversation: {description}?",
            "criteria": {
                "present": f"Yes. The conversation explicitly states or clearly indicates that {description}.",
                "absent": f"No. The conversation does not state or indicate that {description}.",
            },
        }
        for attribute, description in ATTRIBUTE_DESCRIPTIONS.items()
    }
    return {"model": model, "state": row["conversation"], "questions": questions}
