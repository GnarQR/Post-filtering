"""Run the controlled cattle Plausibility Challenge with the existing JEV client.

From the project root (with jev_client.py next to this script in src/):
    python ./src/run_jev_plausibility_pilot.py --condition C --dry-run
    python ./src/run_jev_plausibility_pilot.py --condition C
    python ./src/run_jev_plausibility_pilot.py --condition ALL

Required in project-root .env for live calls: JEV_API_KEY=...

The 80-item challenge contains single atomic claims, so one JEV call checks each
premise/answer pair. The sentence-level rule and the full-answer contradiction
rule are applied to that *same* single-claim result. This is not a multi-sentence
full-answer experiment, and no second, redundant API call is made.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Callable

CONDITIONS = ("A", "B", "C", "D")
CATEGORIES = (
    "biological_role", "temporal_order", "relational",
    "numeric_physical", "unit_scale",
)
EXPECTED = {
    "A": ("normal", "normal", "Supported", "Plausible"),
    "B": ("normal", "corrupt", "Unsupported", "Implausible"),
    "C": ("corrupt", "corrupt", "Supported", "Implausible"),
    "D": ("corrupt", "normal", "Unsupported", "Plausible"),
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: JSON 파싱 실패") from e
    return rows


def validate_challenges(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("pilot 입력 데이터가 비어 있음")
    seen = set()
    by_group = defaultdict(set)
    for row in rows:
        fields = (
            "challenge_id", "group_id", "condition", "category", "premise",
            "answer", "normal_claim", "corrupted_claim", "evidence_gold",
            "plausibility_gold",
        )
        missing = [key for key in fields if key not in row]
        if missing:
            raise ValueError(f"데이터 필드 누락: {missing}")
        cid, condition = row["challenge_id"], row["condition"]
        if cid in seen:
            raise ValueError(f"중복 challenge_id: {cid}")
        seen.add(cid)
        if condition not in EXPECTED:
            raise ValueError(f"{cid}: 잘못된 condition {condition!r}")
        if row["category"] not in CATEGORIES:
            raise ValueError(f"{cid}: 잘못된 category {row['category']!r}")
        if cid != f"{row['group_id']}_{condition}":
            raise ValueError(f"{cid}: group_id와 condition 불일치")
        if row["normal_claim"] == row["corrupted_claim"]:
            raise ValueError(f"{cid}: normal과 corrupted claim이 동일")
        premise_variant, answer_variant, evidence, plausible = EXPECTED[condition]
        expected_premise = row["normal_claim"] if premise_variant == "normal" else row["corrupted_claim"]
        expected_answer = row["normal_claim"] if answer_variant == "normal" else row["corrupted_claim"]
        if row["premise"] != expected_premise or row["answer"] != expected_answer:
            raise ValueError(f"{cid}: premise/answer가 A/B/C/D 정의와 불일치")
        if (row["evidence_gold"], row["plausibility_gold"]) != (evidence, plausible):
            raise ValueError(f"{cid}: gold label이 조건 정의와 불일치")
        by_group[row["group_id"]].add(condition)

    if len(rows) == 80:
        if len(by_group) != 20 or any(cs != set(CONDITIONS) for cs in by_group.values()):
            raise ValueError("80개 pilot에 완전한 A/B/C/D 그룹 20개가 없음")
        if any(sum(x["category"] == c for x in rows) != 16 for c in CATEGORIES):
            raise ValueError("80개 pilot의 category 균형이 깨짐")


def decision(label: str, p_entailment: float | None, threshold: float) -> dict:
    """Match established Hybrid: any non-ENT/low confidence sentence OR full CON.

    All challenge answers are individual claims; one result therefore serves both
    roles. Like the existing code, missing p is ignored when the label is ENT,
    but the summary explicitly reports such missing scores.
    """
    if label not in {"ENT", "CON", "NEU"}:
        raise ValueError(f"알 수 없는 JEV label: {label!r}")
    if p_entailment is not None:
        p_entailment = float(p_entailment)
        if not 0.0 <= p_entailment <= 1.0:
            raise ValueError(f"잘못된 p_entailment: {p_entailment}")
    sentence_suspect = label != "ENT" or (p_entailment is not None and p_entailment < threshold)
    full_contradiction = label == "CON"
    hybrid_suspect = sentence_suspect or full_contradiction
    return {
        "sentence_suspect": sentence_suspect,
        "full_contradiction": full_contradiction,
        "jev_suspect": hybrid_suspect,
        "decision": "SUSPECT" if hybrid_suspect else "CLEAN",
    }


def run_cases(cases: list[dict], out: Path, key: str, sleep_sec: float,
              call: Callable[[str, str, str], dict]) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for old in read_jsonl(out):
            cid = old["challenge_id"]
            if cid in done:
                raise ValueError(f"출력에 중복 ID: {cid}")
            done[cid] = old

    # A changed claim must never reuse a cached score for the same ID.
    for case in cases:
        old = done.get(case["challenge_id"])
        if old and (old.get("premise") != case["premise"] or old.get("answer") != case["answer"]):
            raise ValueError(
                f"{case['challenge_id']}: 이전 실행 후 claim이 변경됨. "
                "기존 결과를 백업/삭제하거나 다른 --out을 사용하세요."
            )

    pending = [case for case in cases if case["challenge_id"] not in done]
    print(f"선택된 샘플 {len(cases)}개 / 기존 완료 {len(cases)-len(pending)}개 / 새 호출 {len(pending)}개")
    with out.open("a", encoding="utf-8") as f:
        for i, case in enumerate(pending, 1):
            cid = case["challenge_id"]
            result = call(case["premise"], case["answer"], key)
            label = result["label"]
            p = result.get("p_entailment")
            # Validate server output before saving a record.
            decision(label, p, 0.90)
            record = {
                "challenge_id": cid,
                "group_id": case["group_id"],
                "condition": case["condition"],
                "category": case["category"],
                "violation": case.get("violation"),
                "edit_type": case.get("edit_type"),
                "premise": case["premise"],
                "answer": case["answer"],
                "evidence_gold": case["evidence_gold"],
                "plausibility_gold": case["plausibility_gold"],
                "jev_label": label,
                "p_entailment": None if p is None else float(p),
                "evaluation_unit": "one_controlled_claim",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # allow interrupted runs to resume without repeating finished API calls
            print(f"[{i}/{len(pending)}] {cid} {label} p={p}")
            if sleep_sec and i < len(pending):
                time.sleep(sleep_sec)
    return read_jsonl(out)


def summarize(cases: list[dict], raw_results: list[dict], threshold: float) -> dict:
    by_id = {r["challenge_id"]: r for r in raw_results}
    analyzed = []
    for case in cases:
        r = by_id.get(case["challenge_id"])
        if r is None:
            continue
        analyzed.append({**r, **decision(r["jev_label"], r["p_entailment"], threshold)})

    def stats(items: list[dict]) -> dict:
        total = len(items)
        labels = Counter(x["jev_label"] for x in items)
        clean = sum(x["decision"] == "CLEAN" for x in items)
        suspect = total - clean
        missing_p = sum(x["p_entailment"] is None for x in items)
        return {
            "n": total,
            "labels": {name: labels[name] for name in ("ENT", "CON", "NEU")},
            "ent_rate": labels["ENT"] / total if total else None,
            "clean": clean,
            "clean_rate": clean / total if total else None,
            "suspect": suspect,
            "suspect_rate": suspect / total if total else None,
            "missing_probability": missing_p,
        }

    summary = {
        "total_selected": len(cases),
        "total_completed": len(analyzed),
        "sentence_threshold": threshold,
        "hybrid_rule": "single_claim(sentence_label != ENT OR p < threshold) OR full_label == CON",
        "overall": stats(analyzed),
        "by_condition": {cond: stats([x for x in analyzed if x["condition"] == cond]) for cond in CONDITIONS},
        "by_category": {cat: stats([x for x in analyzed if x["category"] == cat]) for cat in CATEGORIES},
        "by_condition_category": {
            cond: {cat: stats([x for x in analyzed if x["condition"] == cond and x["category"] == cat])
                   for cat in CATEGORIES}
            for cond in CONDITIONS
        },
        "results": [
            {"challenge_id": x["challenge_id"], "condition": x["condition"],
             "category": x["category"], "jev_label": x["jev_label"],
             "p_entailment": x["p_entailment"], "decision": x["decision"]}
            for x in analyzed
        ],
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 66)
    print(f"Plausibility pilot: {summary['total_completed']}/{summary['total_selected']}건 완료"
          f" (threshold={summary['sentence_threshold']:.2f})")
    print(" condition    n    ENT  CON  NEU    ENT%  CLEAN  CLEAN%  SUSPECT")
    for cond in CONDITIONS:
        s = summary["by_condition"][cond]
        if not s["n"]:
            continue
        print(f" {cond:>8s} {s['n']:4d} {s['labels']['ENT']:6d} {s['labels']['CON']:4d}"
              f" {s['labels']['NEU']:4d} {100*s['ent_rate']:7.1f}"
              f" {s['clean']:6d} {100*s['clean_rate']:7.1f} {s['suspect']:8d}")
    print("=" * 66)
    if summary["by_condition"]["C"]["n"]:
        print("C 조건 category별 ENT/CLEAN (각 4개가 모두 완료됐는지 확인):")
        for cat in CATEGORIES:
            s = summary["by_condition_category"]["C"][cat]
            print(f"  {cat:18s} ENT={s['labels']['ENT']}/{s['n']} "
                  f"CLEAN={s['clean']}/{s['n']}")
    if summary["overall"]["missing_probability"]:
        print(f"주의: p_entailment가 없는 결과 {summary['overall']['missing_probability']}건."
              " 기존 규칙대로 ENT+None은 CLEAN이 될 수 있습니다.")
    if summary["total_completed"] < summary["total_selected"]:
        print("일부 샘플 미실행: 위 비율은 완료된 샘플만을 기준으로 계산했습니다.")
    print("결과 해석: Evidence-grounding 결과이지 Plausibility verifier 성능 측정은 아님.")


def main() -> None:
    parser = argparse.ArgumentParser(description="JEV Plausibility pilot evaluator")
    parser.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    parser.add_argument("--condition", choices=(*CONDITIONS, "ALL"), default="C")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="pilot smoke test size; 0 = all")
    parser.add_argument("--dry-run", action="store_true", help="API 키 없이 입력/구조 확인")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold는 0~1 사이여야 합니다")
    if args.sleep < 0 or args.limit < 0:
        parser.error("--sleep과 --limit는 음수일 수 없습니다")

    cases = read_jsonl(args.input)
    validate_challenges(cases)
    cases = [r for r in cases if args.condition == "ALL" or r["condition"] == args.condition]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        parser.error("해당 condition에 샘플이 없습니다")

    out = args.out or Path(f"results/jev_plausibility_pilot_{args.condition}.jsonl")
    summary_out = args.summary_out or Path(f"results/jev_plausibility_pilot_{args.condition}_summary.json")
    counts = Counter(r["category"] for r in cases)
    print(f"Condition={args.condition}; selected={len(cases)}; category={dict(counts)}")
    print(f"원본: {args.input}; 결과: {out}; 요약: {summary_out}")
    if args.dry_run:
        for case in cases:
            print(f"  {case['challenge_id']} | {case['category']} | "
                  f"identity={case['premise'] == case['answer']}")
        print("DRY RUN: JEV API를 호출하지 않았습니다.")
        return

    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("JEV_API_KEY")
    if not key:
        raise RuntimeError("JEV_API_KEY가 없습니다. 프로젝트 루트의 .env에 설정하세요.")
    from jev_client import call_jev
    records = run_cases(cases, out, key, args.sleep, call_jev)
    summary = summarize(cases, records, args.threshold)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"Raw 결과: {out.resolve()}\n집계 결과: {summary_out.resolve()}")


if __name__ == "__main__":
    main()
