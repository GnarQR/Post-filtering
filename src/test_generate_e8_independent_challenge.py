# coding: utf-8
"""No-network smoke/unit tests: python -m unittest discover -s src -p 'test_generate_e8_independent_challenge.py' -v"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_e8_independent_challenge as g


class IndependentChallengeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.args = SimpleNamespace(
            outdir=root / "results", review_sheet=root / "data" / "blind_review.csv",
            verified=root / "data" / "verified.csv", model="gemini-2.5-flash-lite",
            temperature=0.8, thinking_budget=0, max_output_tokens=2048,
            timeout=1, retries=0, pause=0, answers_per_scenario=3,
        )
        self.fake_calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def mock_api(self, prompt, system, schema, args, key):
        self.fake_calls += 1
        if "situation" in schema["required"]:
            return ({"question": f"우사 관리 상황 {self.fake_calls}에서 어떻게 대응하면 좋을까요?",
                     "situation": "진행 중 상황을 더 확인할 필요가 있는 경우입니다.",
                     "decision_focus": "현실적인 다음 조치의 선택"}, {}, 0.01)
        return ({"answer": f"현장에서 할 수 있는 조치를 구체적으로 제안합니다. {self.fake_calls}"}, {}, 0.01)

    def test_plan_is_30_without_gold(self):
        slots = g.slots()
        self.assertEqual(len(slots), 30)
        self.assertEqual(len({s["scenario_id"] for s in slots}), 30)
        self.assertEqual({s["category"] for s in slots}, set(g.CATEGORIES))
        self.assertNotIn("VIOLATION", g.ANSWER_SYSTEM)
        self.assertNotIn("PASS", g.ANSWER_SYSTEM)

    def test_prompt_does_not_ask_for_labelled_pairs(self):
        p = g.scenario_prompt(g.slots()[0], {name: [] for name in g.CATEGORIES})
        q = g.answer_prompt("축사에서는 무엇을 해야 할까요?", "operations")
        for text in (p, q):
            self.assertNotIn("pass_answer", text)
            self.assertNotIn("violation_answer", text)
            self.assertNotIn("B1", text)
            self.assertNotIn("P1 위험 신호", text)

    def test_resume_and_preserve_review(self):
        old = {name: [] for name in g.CATEGORIES}
        chosen = g.slots()[:2]
        with patch.object(g, "call_gemini", self.mock_api):
            g.generate_scenarios(self.args, chosen, old, "FAKE")
            g.generate_answers(self.args, chosen, "FAKE")
            g.export(self.args, old)
            self.assertEqual(self.fake_calls, 8)
            rows = g.read_csv(self.args.review_sheet)
            self.assertEqual(len(rows), 6)
            self.assertNotIn("generation_mode", rows[0])
            rows[0]["최종Gold"] = "PASS"
            g.atomic_csv(self.args.review_sheet, rows,
                         ["case_id", "question", "answer", "최종Gold", "confidence_1to5", "review_note"])
            g.generate_scenarios(self.args, chosen, old, "FAKE")
            g.generate_answers(self.args, chosen, "FAKE")
            g.export(self.args, old)
            self.assertEqual(self.fake_calls, 8)  # No API calls on resume
        self.assertEqual(sum(x["최종Gold"] == "PASS" for x in g.read_csv(self.args.review_sheet)), 1)

    def test_finalizer_outputs_existing_b0_b1_schema(self):
        old = {name: [] for name in g.CATEGORIES}
        with patch.object(g, "call_gemini", self.mock_api):
            g.generate_scenarios(self.args, g.slots()[:1], old, "FAKE")
            g.generate_answers(self.args, g.slots()[:1], "FAKE")
            g.export(self.args, old)
        rows = g.read_csv(self.args.review_sheet)
        for i, row in enumerate(rows):
            row["최종Gold"] = ("PASS", "VIOLATION", "UNCERTAIN")[i]
        g.atomic_csv(self.args.review_sheet, rows,
                     ["case_id", "question", "answer", "최종Gold", "confidence_1to5", "review_note"])
        g.finalize(self.args)
        verified = g.read_csv(self.args.verified)
        self.assertEqual(len(verified), 2)
        self.assertEqual({x["최종Gold"] for x in verified}, {"PASS", "VIOLATION"})
        self.assertTrue({"case_id", "scenario_id", "split", "difficulty", "category", "question",
                         "answer", "최종Gold", "수정답변"}.issubset(verified[0]))

    def test_review_requires_all_gold(self):
        old = {name: [] for name in g.CATEGORIES}
        with patch.object(g, "call_gemini", self.mock_api):
            g.generate_scenarios(self.args, g.slots()[:1], old, "FAKE")
            g.generate_answers(self.args, g.slots()[:1], "FAKE")
            g.export(self.args, old)
        with self.assertRaisesRegex(ValueError, "Gold missing"):
            g.finalize(self.args)


if __name__ == "__main__":
    unittest.main()
