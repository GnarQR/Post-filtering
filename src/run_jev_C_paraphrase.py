"""JEV entailment test of manually approved C paraphrases.

Project root:
  python .\\src\\run_jev_C_paraphrase.py --dry-run
  python .\\src\\run_jev_C_paraphrase.py --limit 2
  python .\\src\\run_jev_C_paraphrase.py

REQUIRES data/plausibility_C_paraphrases_review.csv approved=Y for each case.
Original corrupted C premise is preserved; only hypothesis is paraphrased.
Uses original jev_client.call_jev and the frozen threshold=0.90.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from jev_client import call_jev

load_dotenv()

VERSION = "C_paraphrase_manually_approved_v1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_answer(row: dict) -> str:
    """Edited answer overrides generator text only when explicitly approved."""
    return (row.get("edited_answer") or row.get("generated_answer") or "").strip()


def read_cases(pilot_path: Path, review_path: Path) -> tuple[list[dict], int, int]:
    baseline = {r["challenge_id"]: r for r in load_jsonl(pilot_path) if r["condition"] == "C"}
    if len(baseline) != 20:
        raise ValueError("기존 C 조건 20개가 필요합니다")
    with review_path.open("r", encoding="utf-8-sig", newline="") as file:
        review = list(csv.DictReader(file))
    if len({r["case_id"] for r in review}) != len(review):
        raise ValueError("수동 검수 CSV case_id 중복")
    cases = []
    pending = rejected = 0
    for item in review:
        cid = item["case_id"]
        src = baseline.get(cid)
        if src is None:
            raise ValueError(f"기본 C 조건에 없는 ID: {cid}")
        if item["premise"] != src["premise"]:
            raise ValueError(f"{cid}: 검수 CSV 원문과 C premise가 다릅니다")
        approved = (item.get("approved") or "").strip().upper()
        if approved not in {"", "Y", "N"}:
            raise ValueError(f"{cid}: approved는 Y, N 또는 빈 칸만 가능합니다")
        if approved == "N":
            rejected += 1
            continue
        if approved != "Y":
            pending += 1
            continue
        hypothesis = clean_answer(item)
        if not hypothesis:
            raise ValueError(f"{cid}: 승인된 답변이 비어 있습니다")
        if "".join(hypothesis.split()) == "".join(src["premise"].split()):
            raise ValueError(f"{cid}: 동일 문장 복사여서 paraphrase 효과를 측정할 수 없음")
        cases.append({
            "case_id": cid,
            "group_id": src["group_id"],
            "category": src["category"],
            "violation": src["violation"],
            "premise": src["premise"],
            "paraphrase": hypothesis,
            "gold_evidence": "Supported",
            "gold_plausibility": "Implausible",
            "input_fingerprint": hashlib.sha256((src["premise"] + "\n" + hypothesis).encode("utf-8")).hexdigest(),
            "experiment_version": VERSION,
        })
    return cases, pending, rejected


def decide(label: str, p: float | None, threshold: float) -> str:
    if label not in ("ENT", "CON", "NEU"):
        raise ValueError(f"잘못된 JEV label: {label}")
    if p is not None and not (0 <= float(p) <= 1):
        raise ValueError(f"p_entailment 범위 오류: {p}")
    suspect = label != "ENT" or (p is not None and float(p) < threshold)
    # A one-claim answer gives the same label for both sentence and full-answer checks.
    suspect = suspect or label == "CON"
    return "SUSPECT" if suspect else "CLEAN"


def summarize(cases: list[dict], done: dict[str, dict], threshold: float,
              total_pending: int, rejected: int, out: Path) -> dict:
    completed = [done[c["case_id"]] for c in cases if c["case_id"] in done]
    hist = Counter(x["jev_label"] for x in completed)
    clean = sum(decide(r["jev_label"], r.get("p_entailment"), threshold) == "CLEAN" for r in completed)
    cat = {}
    for category in sorted({x["category"] for x in cases}):
        sub = [r for r in completed if r["category"] == category]
        cat[category] = {"n": len(sub), "ENT": sum(r["jev_label"] == "ENT" for r in sub),
                         "CLEAN": sum(decide(r["jev_label"], r.get("p_entailment"), threshold) == "CLEAN" for r in sub)}
    result = {
        "version": VERSION,
        "note": "Single-claim JEV evidence-faithfulness test. Not a plausibility test.",
        "review_approved": len(cases), "review_pending": total_pending, "review_rejected": rejected,
        "completed": len(completed), "threshold": threshold,
        "labels": {k: hist[k] for k in ("ENT", "CON", "NEU")},
        "CLEAN": clean, "SUSPECT": len(completed) - clean,
        "ENT_rate": hist["ENT"] / len(completed) if completed else None,
        "CLEAN_rate": clean / len(completed) if completed else None,
        "by_category": cat,
        "raw_results": str(out),
    }
    print(f"\n완료 {len(completed)}/{len(cases)} approved | ENT={hist['ENT']} CON={hist['CON']} NEU={hist['NEU']}")
    print(f"CLEAN={clean} SUSPECT={len(completed)-clean} @ {threshold:.2f}")
    for k, v in cat.items():
        print(f"  {k:18s} ENT={v['ENT']}/{v['n']} CLEAN={v['CLEAN']}/{v['n']}")
    return result


def compare_identity(identity_path: Path, completed: list[dict], threshold: float):
    if not identity_path.exists():
        print("원래 C identity 결과 파일이 없어 비교를 생략했습니다")
        return
    old = {r["challenge_id"]: r for r in load_jsonl(identity_path)}
    matches = [(old[r["case_id"]], r) for r in completed if r["case_id"] in old]
    print(f"\n[기존 C 동일문장 ↔ paraphrase 비교, {len(matches)}쌍]")
    for a, b in matches:
        prev = decide(a["jev_label"], a.get("p_entailment"), threshold)
        curr = decide(b["jev_label"], b.get("p_entailment"), threshold)
        if prev != curr or a["jev_label"] != b["jev_label"]:
            print(f"  {b['case_id']} : {a['jev_label']}/{prev} → {b['jev_label']}/{curr}")
    print("이 결과는 expression robustness 검사이며, Plausibility Guardrail의 탐지 성능이 아닙니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    ap.add_argument("--review", type=Path, default=Path("data/plausibility_C_paraphrases_review.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/jev_plausibility_C_paraphrase.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("results/jev_plausibility_C_paraphrase_summary.json"))
    ap.add_argument("--baseline", type=Path, default=Path("results/jev_plausibility_pilot_C.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1 or args.limit < 0 or args.sleep < 0:
        ap.error("threshold는 [0,1], limit와 sleep은 0 이상이어야 합니다")
    cases, pending, rejected = read_cases(args.input, args.review)
    selected = cases[:args.limit] if args.limit else cases
    print(f"승인={len(cases)}/20, 미검수={pending}, 거절={rejected}, 이번 선택={len(selected)}")
    if args.dry_run:
        for c in selected:
            print(f"  {c['case_id']} | {c['category']} | original: {c['premise'][:65]}\n         paraphrase: {c['paraphrase'][:65]}")
        return
    if not selected:
        print("승인된 claim이 없어 종료합니다. 검수 CSV의 approved 열을 Y로 수정하세요.")
        return
    key = os.getenv("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없습니다")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.out) if args.out.exists() else []
    done = {r["case_id"]: r for r in existing}
    if len(done) != len(existing):
        raise ValueError("기존 결과 case_id 중복")
    for c in cases:
        old = done.get(c["case_id"])
        if old and (old.get("input_fingerprint") != c["input_fingerprint"] or old.get("experiment_version") != VERSION):
            raise ValueError(f"{c['case_id']}: 검수 후 데이터가 변경됨. 새로운 --out 경로를 사용하세요")
    with args.out.open("a", encoding="utf-8") as f:
        for i, c in enumerate(selected, 1):
            if c["case_id"] in done:
                print(f"[{i}/{len(selected)}] {c['case_id']} 캐시 사용")
                continue
            result = call_jev(c["premise"], c["paraphrase"], key)
            decide(result["label"], result.get("p_entailment"), args.threshold)
            row = {**c, "jev_label": result["label"], "p_entailment": result.get("p_entailment")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[c["case_id"]] = row
            print(f"[{i}/{len(selected)}] {c['case_id']} {result['label']} p={result['p_entailment']}")
            if i < len(selected) and args.sleep:
                time.sleep(args.sleep)
    summary = summarize(cases, done, args.threshold, pending, rejected, args.out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_identity(args.baseline, [done[c["case_id"]] for c in selected if c["case_id"] in done], args.threshold)
    print(f"저장: {args.out} / {args.summary_out}")


if __name__ == "__main__":
    main()
