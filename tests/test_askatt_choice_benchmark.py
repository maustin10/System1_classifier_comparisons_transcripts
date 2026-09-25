from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "askatt_choice_benchmark.json"


def test_choice_benchmark_truth_matches_declared_options() -> None:
    benchmark = json.loads(BENCHMARK.read_text())
    questions = benchmark["questions"]
    cases = benchmark["cases"]
    assert len(questions) == 4
    assert len(cases) == 24
    assert {case["difficulty"] for case in cases} == {
        "direct",
        "paraphrase",
        "distractor",
        "multiple_issues",
    }
    for case in cases:
        assert set(case["expected"]) == set(questions)
        for question_id, expected in case["expected"].items():
            assert expected in questions[question_id]["criteria"]


def test_choice_benchmark_ids_are_unique() -> None:
    cases = json.loads(BENCHMARK.read_text())["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
