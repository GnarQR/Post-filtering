"""Offline tests: NEVER use live API keys or make real network requests."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "run_e8_safety_benchmark.py"
DATA = Path(__file__).resolve().parents[1] / "data" / "e8_human_screened_103_provisional.jsonl"
spec = importlib.util.spec_from_file_location("e8", SCRIPT)
e8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e8)


class FakeResponse:
    status_code = 200

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        return None


class OfflineE8Tests(unittest.TestCase):
    def test_real_dataset_counts_no_split_leak(self):
        cases, data_hash = e8.load_cases(DATA)
        self.assertEqual(len(cases), 103)
        self.assertEqual(len(data_hash), 64)
        self.assertEqual(sum(c["split"] == "dev" for c in cases), 40)
        self.assertEqual(sum(c["split"] == "test" for c in cases), 63)
        self.assertEqual(sum(c["gold"] == "VIOLATION" for c in cases), 34)

    def test_inputs_hide_gold_and_reviewer_metadata(self):
        example = {"question": "위험 권고를 해줘", "answer": "그렇게 하면 안 됩니다", "gold": "VIOLATION",
                   "scenario_seed": "DO_NOT_SEND", "generated_risk_reason": "SECRET"}
        prompt = e8.text_input(example)
        self.assertIn("위험 권고를 해줘", prompt)
        self.assertIn("그렇게 하면 안 됩니다", prompt)
        self.assertNotIn("VIOLATION", prompt)
        self.assertNotIn("DO_NOT_SEND", prompt)
        self.assertNotIn("SECRET", prompt)
        jev = e8.jev_payload(example, "jev-latest")
        self.assertEqual(set(jev["state"]), {"question", "answer"})
        g = e8.gemini_payload(example, 0, 4096, 0)
        self.assertNotIn("DO_NOT_SEND", json.dumps(g, ensure_ascii=False))

    def test_metrics_review_counted_as_abstention(self):
        rows = [
            {"gold": "VIOLATION", "prediction": "VIOLATION"},
            {"gold": "VIOLATION", "prediction": "REVIEW"},
            {"gold": "VIOLATION", "prediction": "PASS"},
            {"gold": "PASS", "prediction": "VIOLATION"},
            {"gold": "PASS", "prediction": "REVIEW"},
            {"gold": "PASS", "prediction": "PASS"},
        ]
        m = e8.metrics(rows)
        self.assertEqual(m["counts"], {
            "TP_violation": 1, "FP_violation": 1, "TN_pass": 1,
            "FN_pass": 1, "abstained_gold_violation": 1,
            "abstained_gold_pass": 1,
        })
        self.assertEqual(m["strict_violation_recall"], 0.333333)
        self.assertEqual(m["triage_violation_capture_recall"], 0.666667)
        self.assertEqual(m["review_rate"], 0.333333)
        self.assertEqual(m["coverage_non_review"], 0.666667)

    def test_gemini_max_tokens_and_retries(self):
        case = {"case_id": "S001", "question": "소가 아파요", "answer": "괜찮아요"}
        args = e8.parse_args(["--dry-run"])
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            if len(calls) == 1:
                return FakeResponse({"candidates": [{"finishReason": "MAX_TOKENS"}],
                                     "usageMetadata": {"promptTokenCount": 10, "thoughtsTokenCount": 5, "totalTokenCount": 15}})
            return FakeResponse({"modelVersion": "gemini-2.5-flash", "candidates": [{
                "finishReason": "STOP", "content": {"parts": [{"text": '{"label":"REVIEW", "reason":"근거 부족"}'}]}}],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3,
                                  "thoughtsTokenCount": 8, "totalTokenCount": 22}})

        with patch.object(e8.requests, "post", side_effect=fake_post):
            response = e8.call_gemini("fake", "gemini-2.5-flash", case, args, 1024)
        self.assertEqual(response["prediction"], "REVIEW")
        self.assertEqual(response["attempts"], 2)
        self.assertEqual(response["usage"]["thinking_tokens"], 13)
        self.assertEqual(response["output_limit_used"], 8192)

    def test_jev_choice_parsing(self):
        case = {"case_id": "S001", "question": "소가 아파요", "answer": "괜찮아요"}
        args = e8.parse_args(["--dry-run"])
        mock_data = {"model": "jev-1.14.0", "answers": {"safety": {
            "choice": "violation", "probabilities": {"pass": 0.1, "review": 0.1, "violation": 0.8},
            "confidence": 0.8}}, "usage": {"input_tokens": 500}}
        with patch.object(e8.requests, "post", return_value=FakeResponse(mock_data)):
            response = e8.call_jev("fake", "jev-latest", case, args)
        self.assertEqual(response["prediction"], "VIOLATION")
        self.assertEqual(response["probabilities"]["VIOLATION"], 0.8)
        self.assertEqual(response["model_resolved"], "jev-1.14.0")

    def test_dry_run_no_api_no_write_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            out = p / "prediction.jsonl"
            csv = p / "prediction.csv"
            summary = p / "summary.json"
            err = p / "errors.jsonl"
            base = ["--input", str(DATA), "--out", str(out), "--csv", str(csv),
                    "--summary", str(summary), "--errors", str(err), "--pilot"]
            with patch.object(e8.requests, "post", side_effect=AssertionError("should not request")):
                status = e8.main(base + ["--dry-run"])
            self.assertEqual(status, 0)
            self.assertFalse(out.exists())

            calls = []
            def fake_post(url, headers, json, timeout):
                calls.append(url)
                if "typesafe.ai" in url:
                    return FakeResponse({"model": "jev-1.14.0", "answers": {
                        "safety": {"choice": "pass", "probabilities": {"pass": .9, "review": .05,
                                                                       "violation": .05}}}})
                return FakeResponse({"modelVersion": "gemini-2.5-flash", "candidates": [{
                    "finishReason": "STOP", "content": {"parts": [{"text": '{"label":"PASS", "reason":"검토"}'}]}}]})

            with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy", "JEV_API_KEY": "dummy"}), \
                 patch.object(e8.requests, "post", side_effect=fake_post):
                self.assertEqual(e8.main(base), 0)
                self.assertEqual(len(calls), 6)
                self.assertEqual(len(e8.read_jsonl(out)), 6)
                self.assertEqual(e8.main(base), 0)
                self.assertEqual(len(calls), 6, "resume should skip all completed")
            self.assertTrue(csv.is_file())
            self.assertTrue(summary.is_file())
            report = json.loads(summary.read_text(encoding="utf-8"))
            self.assertFalse(report["is_full_three_model_evaluation_complete"])
            self.assertEqual(report["completed_total_jobs"], 6)

            with self.assertRaisesRegex(ValueError, "설정이 현재 실행과 다릅니다"):
                e8.main(base + ["--temperature", "0.3", "--evaluate-only"])

    def test_test_requires_confirmation(self):
        with self.assertRaises(SystemExit):
            e8.parse_args(["--split", "test", "--limit", "1"])
        with self.assertRaises(SystemExit):
            e8.parse_args(["--split", "test", "--final-test", "--pilot"])


if __name__ == "__main__":
    unittest.main()
