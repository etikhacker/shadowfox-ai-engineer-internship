import json
import os

from main import judge_text


if not os.getenv("OPENROUTER_API_KEY"):
    print(json.dumps({"status": "skipped", "reason": "OPENROUTER_API_KEY is not configured"}))
else:
    result = judge_text("Synthetic test: I was charged twice for a made-up subscription.")
    print(json.dumps({"status": "ok", "result": result.model_dump()}, indent=2))
