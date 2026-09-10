"""Run the deterministic eight-case local evaluation without spending provider quota."""

import json
import os
from pathlib import Path

from main import _stub_judgement


ROOT = Path(__file__).parent
CASES = json.loads((ROOT / "eval_cases.json").read_text(encoding="utf-8"))


def expected_stub_category(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("charged", "invoice")):
        return "billing"
    if any(word in lowered for word in ("crash", "error")):
        return "bug"
    if any(word in lowered for word in ("add", "filter")):
        return "feature"
    return "other"


results = []
for case in CASES:
    predicted = _stub_judgement(expected_stub_category(case["text"]))
    results.append(predicted.category == case["expected_category"])

score = sum(results) / len(results)
print(json.dumps({"cases": len(results), "correct": sum(results), "score": score, "mode": "stub"}, indent=2))
