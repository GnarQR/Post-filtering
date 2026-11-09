"""E4: Full-claim Axiom-NLI (no atomic rewriting).

Keep this file beside `probe_jev_biological_relations.py` and `jev_client.py`
in the project's src/ directory; run from the project root.

  python .\\src\\probe_jev_full_axiom_nli.py --dry-run
  python .\\src\\probe_jev_full_axiom_nli.py --limit 2
  python .\\src\\probe_jev_full_axiom_nli.py

E3 and E4 use the exact same DOMAIN_AXIOMS and `jev_client.call_jev`.
Only the hypothesis changes: E3=manually compressed atomic claim;
E4=original full answer claim. This is a *development probe*, not a test estimate.

E4 label interpretation: CON = contradiction with the provided axiom;
ENT = supported by axiom; NEU = not determined by axiom. Neither ENT
nor NEU proves general-world truth.
"""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from jev_client import call_jev
from probe_jev_biological_relations import DOMAIN_AXIOMS, load_examples

MODE = "full-axiom-nli"
DEFAULT_OUT = Path("results/jev_bio_probe_full_axiom_nli.jsonl")
DEFAULT_E3 = Path("results/jev_bio_probe_axiom_nli.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise RuntimeError(f"{path}:{i}: JSONL 오류: {err}") from err
    return result


def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def counts(rows: list[dict]) -> None:
    if not rows:
        print("완료된 결과가 없습니다.")
        return
    totals = Counter()
    for row in rows:
        totals[(row["gold"], row["predicted"])] += 1
    print("\n[Gold × NLI 판정: E4]")
    print(f"{'gold':<9} {'ENT':>4} {'CON':>4} {'NEU':>4}")
    for gold in ("VALID", "INVALID"):
        print(f"{gold:<9} " + " ".join(f"{totals[(gold, label)]:>4}" for label in ("ENT", "CON", "NEU")))
    invalid = sum(r["gold"] == "INVALID" for r in rows)
    valid = sum(r["gold"] == "VALID" for r in rows)
    tp = totals[("INVALID", "CON")]
    fp = totals[("VALID", "CON")]
    print(f"CON-as-violation: INVALID detected {tp}/{invalid}; VALID incorrectly flagged {fp}/{valid}")
    print("주의: NEU는 VALID의 증거가 아니며, 비-CON은 '이 원칙과 명시적 모순 미검출'만 의미합니다.")


def compare(e3_path: Path, e4_rows: list[dict]) -> None:
    if not e3_path.exists():
        print(f"\n[E3 비교 생략] 파일 없음: {e3_path}")
        return
    e3 = {r["probe_id"]: r for r in read_jsonl(e3_path)}
    common = [(row, e3[row["probe_id"]]) for row in e4_rows if row["probe_id"] in e3]
    if not common:
        print("\n[E3 비교 생략] 공통 probe_id 없음")
        return
    print(f"\n[E3 atomic-axiom ↔ E4 full-axiom: 공통 {len(common)}개]")
    changes = 0
    for e4, e3_row in common:
        if e4["gold"] != e3_row["gold"]:
            raise RuntimeError(f"gold 불일치: {e4['probe_id']}")
        if e4["predicted"] != e3_row["predicted"]:
            changes += 1
            print(f"  {e4['probe_id']}: gold={e4['gold']}, E3={e3_row['predicted']} -> E4={e4['predicted']}")
    print(f"label 변경: {changes}/{len(common)}")
    for name, subset in (("E3", [e3_row for _, e3_row in common]), ("E4", [e4 for e4, _ in common])):
        tp = sum(r["gold"] == "INVALID" and r["predicted"] == "CON" for r in subset)
        fp = sum(r["gold"] == "VALID" and r["predicted"] == "CON" for r in subset)
        print(f"  {name}: INVALID→CON={tp}, VALID→CON={fp}")
    print("비교 주의: 문장 축약과 원문 선택의 효과를 비교한 개발용 11개 대응 사례입니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="E4: 검증된 생물학적 원칙 + 원래 긴 claim으로 JEV NLI 실험")
    ap.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--compare-to", type=Path, default=DEFAULT_E3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.limit < 0 or args.sleep < 0:
        ap.error("--limit과 --sleep은 0 이상이어야 합니다.")
    if args.out.resolve() == args.compare_to.resolve():
        ap.error("E4 --out과 E3 --compare-to를 같은 파일로 지정하면 안 됩니다.")

    selected = load_examples(args.input, include_controls=not args.no_controls)
    if args.limit:
        selected = selected[: args.limit]
    print(f"E4 Full-Claim Axiom-NLI | {len(selected)} cases | output={args.out}")
    print("Premise: E3와 동일한 DOMAIN_AXIOMS / Hypothesis: 원본 전체 answer")
    print(f"Axiom SHA256: {digest(DOMAIN_AXIOMS)[:16]}")

    if args.dry_run:
        print(f"[공통 도메인 원칙] {DOMAIN_AXIOMS}\n")
        for r in selected:
            print(f"  {r['probe_id']} gold={r['gold']} | {r['original_claim']}")
        return

    load_dotenv()
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없습니다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.out) if args.out.exists() else []
    done = {}
    for row in existing:
        pid = row["probe_id"]
        if pid in done:
            raise RuntimeError(f"결과 파일에 중복된 probe_id: {pid}")
        done[pid] = row

    with args.out.open("a", encoding="utf-8") as f:
        for i, sample in enumerate(selected, 1):
            pid = sample["probe_id"]
            statement = sample["original_claim"].strip()
            if not statement:
                raise RuntimeError(f"{pid}: 입력 claim이 비었습니다.")
            fingerprint = digest(DOMAIN_AXIOMS + "\n\n" + statement)
            if pid in done:
                old = done[pid]
                if old.get("input_sha256") != fingerprint or old.get("mode") != MODE:
                    raise RuntimeError(f"{pid}: 이전 결과가 현재 입력/원칙과 달라요. 파일을 확인하세요: {args.out}")
                print(f"[{i}/{len(selected)}] {pid} already completed")
                continue

            result = call_jev(DOMAIN_AXIOMS, statement, key)
            label = result["label"]
            if label not in {"ENT", "CON", "NEU"}:
                raise RuntimeError(f"{pid}: 알 수 없는 NLI label {label}")
            record = {
                "probe_id": pid,
                "type": sample["type"],
                "gold": sample["gold"],
                "mode": MODE,
                "mode_note": "full_original_answer_vs_fixed_domain_axioms",
                "input_statement": statement,
                "original_claim": statement,
                "axioms": DOMAIN_AXIOMS,
                "input_sha256": fingerprint,
                "predicted": label,
                "p_entailment": result.get("p_entailment"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[pid] = record
            print(f"[{i}/{len(selected)}] {pid} gold={sample['gold']} -> {label}")
            time.sleep(args.sleep)

    completed = [done[r["probe_id"]] for r in selected if r["probe_id"] in done]
    print(f"\nCompleted {len(completed)}/{len(selected)}")
    counts(completed)
    compare(args.compare_to, completed)
    print(f"\n결과 저장: {args.out.resolve()}")


if __name__ == "__main__":
    main()
