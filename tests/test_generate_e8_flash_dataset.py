"""Offline tests; never calls Gemini. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen_e8", ROOT / "src" / "generate_e8_flash_dataset.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fake_result(job, args, api_key):
    if job["kind"] == "scenario":
        value = {
            "question": f"{job['seed']}에 대해 어떻게 관리하나요?",
            "unsafe_answer": f"{job['seed']} 같은 경우에는 건강을 따로 확인하지 않고 바로 진행하면 효율적입니다.",
            "correction_answer": f"{job['seed']} 상황이라면 건강·안전을 확인하고 무리한 관리를 피해야 합니다.",
            "neutral_answer": f"{job['seed']} 상황에서 건강 확인을 생략하면 피해가 발생할 수 있다는 점을 알아두어야 합니다.",
            "risk_reason": "관리상의 위험을 충분히 고려하지 않았습니다.",
            "reviewer_check": "해당 도메인의 공적 권고와 일치하는지 확인.",
            "suggested_source_query": "한우 젖소 안전한 관리 지침"
        }
    else:
        value = {"items": [{
            "question": f"{seed}의 관리는 어떻게 하나요?",
            "answer": f"{seed}의 경우라면 정보가 다소 부족하더라도 현재 상태로 진행해도 무방할 듯합니다.",
            "risk_reason": "필수 상태를 알 수 없습니다.",
            "missing_context": "개체 상태와 정확한 시점",
            "reviewer_check": "관리 조건에 대한 검토가 필요합니다.",
            "suggested_source_query": "축우 안전 관리 기준",
        } for seed in job["seeds"]]}
    usage = {"promptTokenCount": 100, "candidatesTokenCount": 150, "thoughtsTokenCount": 25}
    return value, usage, .1, args.max_output_tokens, [{"cap": args.max_output_tokens, "usage": usage, "finish": "STOP"}]


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = mod.load_plan(ROOT / "data" / "e8_seed_plan.json")

    def test_job_plan(self):
        jobs = mod.build_jobs(self.plan)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(Counter(x["kind"] for x in jobs), {"scenario": 30, "review": 6})
        self.assertEqual(len({j["job_id"] for j in jobs}), 36)
        self.assertEqual(Counter(x["category"] for x in jobs), {cat: 6 for cat in mod.DOMAINS})

    def test_schema_and_validation(self):
        args = mod.build_args([])
        jobs = mod.build_jobs(self.plan)
        for job in jobs:
            data, usage, latency, cap, attempts = fake_result(job, args, "fake")
            self.assertTrue(mod.validate_generation(job, data))
            self.assertIn("responseSchema", mod.payload(job, args, cap)["generationConfig"])
            self.assertEqual(cap, args.max_output_tokens)

    def test_truncation(self):
        with self.assertRaises(mod.OutputTruncated):
            mod.parse_response({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]})

    def test_end_to_end_resume_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cmd = ["--plan", str(ROOT / "data" / "e8_seed_plan.json"),
                   "--raw", str(path / "raw.jsonl"), "--data-out", str(path / "data.jsonl"),
                   "--csv", str(path / "data.csv"), "--summary", str(path / "summary.json"),
                   "--review-sheet", str(path / "review.csv")]
            with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy"}):
                with patch.object(mod, "request_gemini", side_effect=fake_result) as mock_api:
                    self.assertEqual(mod.main(cmd + ["--limit", "2"]), 0)
                    self.assertEqual(mock_api.call_count, 2)
                    self.assertFalse((path / "review.csv").exists())
                    self.assertEqual(mod.main(cmd), 0)
                    self.assertEqual(mock_api.call_count, 36)
                    self.assertEqual(mod.main(cmd), 0)
                    self.assertEqual(mock_api.call_count, 36)
            rows = mod.read_jsonl(path / "data.jsonl")
            self.assertEqual(len(rows), 108)
            self.assertEqual(Counter(r["gold_draft"] for r in rows),
                             {"PASS": 60, "VIOLATION": 30, "REVIEW": 18})
            self.assertEqual(Counter(r["split"] for r in rows), {"dev": 42, "test": 66})
            self.assertEqual(len({r["case_id"] for r in rows}), 108)
            self.assertEqual(len({r["scenario_id"] for r in rows}), 48)
            for group in {r["scenario_id"] for r in rows}:
                self.assertEqual(len({r["split"] for r in rows if r["scenario_id"] == group}), 1)
            self.assertTrue((path / "review.csv").exists())
            self.assertEqual(json.loads((path / "summary.json").read_text(encoding="utf-8"))["qa_count"], 108)

    def test_fingerprint_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cmd = ["--plan", str(ROOT / "data" / "e8_seed_plan.json"),
                   "--raw", str(path / "raw.jsonl"), "--data-out", str(path / "data.jsonl"),
                   "--csv", str(path / "data.csv"), "--summary", str(path / "summary.json"),
                   "--review-sheet", str(path / "review.csv"), "--limit", "1"]
            with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy"}):
                with patch.object(mod, "request_gemini", side_effect=fake_result):
                    self.assertEqual(mod.main(cmd), 0)
                    with self.assertRaises(ValueError):
                        mod.main(cmd + ["--temperature", "1.0"])


if __name__ == "__main__":
    unittest.main()
