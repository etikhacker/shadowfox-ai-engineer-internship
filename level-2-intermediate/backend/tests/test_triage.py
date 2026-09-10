import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app, call_llm, parse_judgement, retryable_status


class TriageEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_valid_stub_triage_returns_closed_schema(self):
        response = self.client.post(
            "/triage",
            json={"text": "I was charged twice for my subscription."},
            headers={"X-LLM-Stub": "billing"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body),
            {"category", "urgency", "suggested_team", "confidence", "reason", "meta"},
        )
        self.assertEqual(body["category"], "billing")
        self.assertIn(body["urgency"], {"low", "normal", "high"})
        self.assertGreaterEqual(body["confidence"], 0)
        self.assertLessEqual(body["confidence"], 1)

    def test_rejects_empty_text_before_model_call(self):
        response = self.client.post("/triage", json={"text": "   "})
        self.assertEqual(response.status_code, 422)

    def test_rejects_text_longer_than_two_thousand_characters(self):
        response = self.client.post("/triage", json={"text": "x" * 2001})
        self.assertEqual(response.status_code, 422)

    def test_rejects_unknown_model_category(self):
        with self.assertRaises(ValueError):
            parse_judgement(
                {"category": "refunds", "urgency": "low", "suggested_team": "support", "confidence": 0.4, "reason": "x"}
            )

    def test_retries_only_transient_statuses(self):
        self.assertTrue(retryable_status(429))
        self.assertTrue(retryable_status(500))
        self.assertTrue(retryable_status(503))
        self.assertFalse(retryable_status(400))
        self.assertFalse(retryable_status(401))

    def test_invalid_json_is_repaired_once(self):
        responses = iter(["not json", '{"category":"bug","urgency":"high","suggested_team":"engineering","confidence":0.9,"reason":"The app crashes."}'])
        with patch("main.call_llm", side_effect=lambda prompt, repair=False: next(responses)) as mocked:
            from main import judge_text
            result = judge_text("The app crashes on launch.")
        self.assertEqual(result.category, "bug")
        self.assertEqual(mocked.call_count, 2)

    def test_kill_switch_blocks_real_provider(self):
        with patch.dict(os.environ, {"LLM_ENABLED": "0", "LLM_STUB_MODE": "0"}, clear=False):
            response = self.client.post("/triage", json={"text": "The login page is broken."})
        self.assertEqual(response.status_code, 503)

    def test_cost_log_records_stub_request(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"COST_LOG_PATH": os.path.join(directory, "cost.jsonl")}, clear=False):
                response = self.client.post(
                    "/triage",
                    json={"text": "Please add dark mode."},
                    headers={"X-LLM-Stub": "feature"},
                )
                self.assertEqual(response.status_code, 200)
                with open(os.path.join(directory, "cost.jsonl"), encoding="utf-8") as handle:
                    record = json.loads(handle.readline())
                self.assertEqual(record["endpoint"], "/triage")
                self.assertEqual(record["provider"], "stub")


if __name__ == "__main__":
    unittest.main()
