"""E5: Attribution x claim complexity on JEV Axiom-NLI.

Run from the repository root with this file beside:
  src/probe_jev_biological_relations.py
  src/jev_client.py

E5 uses exactly the E3/E4 DOMAIN_AXIOMS and the same call_jev API.
It holds the axioms constant and crosses two factors:
  complexity: 'atomic' (E3 manually curated claim) vs 'full' (E4 source claim)
  attribution: no '문서에 따르면' vs prefixed '문서에 따르면'

Defaults to BIO_001_A and BIO_001_B (8 JEV calls), the pair where
E3 yielded CON and E4 yielded NEU for the impossible claim. Optionally
use --all-bio (32 calls) as a small exploratory robustness check.

Important: 'atomic' and 'full' also differ in wording, not only length.
This is a mechanism probe, NOT an independent test-set performance estimate.

Usage (PowerShell):
  python .\\src\\probe_jev_e5_attribution_complexity.py --dry-run
  python .\\src\\probe_jev_e5_attribution_complexity.py --limit 2
  python .\\src\\probe_jev_e5_attribution_complexity.py
  python .\\src\\probe_jev_e5_attribution_complexity.py --all-bio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from jev_client import call_jev
from probe_jev_biological_relations import DOMAIN_AXIOMS, load_examples

EXPERIMENT_ID = "E5_attribution_x_complexity_v1"
JEV_MODEL = "jev-latest"  # The value configured in jev_client.py
OUT_DEFAULT = Path("results/jev_bio_probe_e5_attribution_complexity.jsonl")
SUMMARY_DEFAULT = Path("results/jev_bio_probe_e5_attribution_complexity_summary.json")
ATTRIBUTION_PREFIX = "문서에 따르면 "
ATTRIBUTION_PATTERN = re.compile(r"^\s*문서에\s+따르면\s*[,，]?\s*")


def remove_leading_attribution(statement: str) -> str:
    """Remove ONLY an initial '문서에 따르면'; preserve other content."""
    return ATTRIBUTION_PATTERN.sub("", statement.strip(), count=1).strip()


def make_hypothesis(base: str, attributed: bool) -> str:
    text = remove_leading_attribution(base)
    if not text:
        raise ValueError("출처 표현을 제거한 뒤 hypothesis가 비어 있습니다.")
    return ATTRIBUTION_PREFIX + text if attributed else text


def jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{line_no}: JSONL 오류: {e}") from e
    return rows


def digest_json(obj: dict) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cases(input_path: Path, pair: str, all_bio: bool) -> list[dict]:
    samples = load_examples(input_path, include_controls=False)
    selected = (
        samples if all_bio else [s for s in samples if s["probe_id"] in {f"{pair}_A", f"{pair}_B"}]
    )
    if len(selected) != (8 if all_bio else 2):
        raise RuntimeError("A/B sample 구성 오류: BIO case 개수를 확인하세요.")

    # Invalid cases appear first so --limit 2 examines the relevant failure.
    selected.sort(key=lambda s: (s["probe_id"].rsplit("_", 1)[0], s["gold"] != "INVALID"))
    cases = []
    for s in selected:
        if not s["atomic_claim"].strip() or not s["original_claim"].strip():
            raise RuntimeError(f"{s['probe_id']}: 비어 있는 atomic/full claim")
        for complexity, base in (("atomic", s["atomic_claim"]), ("full", s["original_claim"])):
            for attributed in (False, True):
                hypothesis = make_hypothesis(base, attributed)
                cell_id = f"{s['probe_id']}::{complexity}::{'attributed' if attributed else 'plain'}"
                input_contract = {
                    "experiment_id": EXPERIMENT_ID,
                    "jev_model": JEV_MODEL,
                    "axioms": DOMAIN_AXIOMS,
                    "hypothesis": hypothesis,
                    "gold": s["gold"],
                    "complexity": complexity,
                    "attributed": attributed,
                }
                cases.append({
                    "case_id": cell_id,
                    "probe_id": s["probe_id"],
                    "gold": s["gold"],
                    "complexity": complexity,
                    "attributed": attributed,
                    "hypothesis": hypothesis,
                    "axioms": DOMAIN_AXIOMS,
                    "experiment_id": EXPERIMENT_ID,
                    "jev_model": JEV_MODEL,
                    "input_sha256": digest_json(input_contract),
                })

    ids = [r["case_id"] for r in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("중복 case_id 발견")
    return cases


def outcome_counts(rows: list[dict]) -> dict:
    counts = Counter((r["gold"], r["complexity"], r["attributed"], r["predicted"]) for r in rows)
    by_cell = []
    for gold in ("INVALID", "VALID"):
        for complexity in ("atomic", "full"):
            for attributed in (False, True):
                n = sum(counts[(gold, complexity, attributed, lbl)] for lbl in ("ENT", "CON", "NEU"))
                by_cell.append({
                    "gold": gold,
                    "complexity": complexity,
                    "attributed": attributed,
                    "n": n,
                    **{lbl: counts[(gold, complexity, attributed, lbl)] for lbl in ("ENT", "CON", "NEU")},
                })
    tp = sum(r["gold"] == "INVALID" and r["predicted"] == "CON" for r in rows)
    fp = sum(r["gold"] == "VALID" and r["predicted"] == "CON" for r in rows)
    invalid = sum(r["gold"] == "INVALID" for r in rows)
    valid = sum(r["gold"] == "VALID" for r in rows)
    return {"completed": len(rows), "invalid_total": invalid, "valid_total": valid,
            "invalid_con": tp, "valid_con": fp, "by_cell": by_cell}


def print_report(selected: list[dict], done: dict[str, dict]) -> dict:
    completed = [done[r["case_id"]] for r in selected if r["case_id"] in done]
    summary = outcome_counts(completed)
    print(f"\n[E5 completed {len(completed)}/{len(selected)}]")
    print(f"{'Gold':<8} {'Length':<7} {'Attr':<4} {'n':>2} {'ENT':>3} {'CON':>3} {'NEU':>3}")
    for c in summary["by_cell"]:
        print(f"{c['gold']:<8} {c['complexity']:<7} {('Y' if c['attributed'] else 'N'):<4} "
              f"{c['n']:>2} {c['ENT']:>3} {c['CON']:>3} {c['NEU']:>3}")
    print(f"INVALID -> CON {summary['invalid_con']}/{summary['invalid_total']}; "
          f"VALID -> CON {summary['valid_con']}/{summary['valid_total']}")
    print("주의: NEU는 반증되지 않았다는 의미일 뿐, VALID라는 증거가 아닙니다.")

    for pid in sorted({r["probe_id"] for r in selected}):
        group = { (r["complexity"], r["attributed"]): done.get(r["case_id"])
                 for r in selected if r["probe_id"] == pid }
        if not any(group.values()):
            continue
        print(f"\n[{pid}]  (row=atomic/full; col=plain/attributed)")
        print(f"{'':<8} {'No attribution':<17} {'With attribution':<17}")
        for complexity in ("atomic", "full"):
            labels = []
            for attributed in (False, True):
                result = group.get((complexity, attributed))
                labels.append(result["predicted"] if result else "PENDING")
            print(f"{complexity:<8} {labels[0]:<17} {labels[1]:<17}")

        # Contrasts hold the other factor fixed, but this 2x2 alone is not causal proof.
        if all(group.get((cx, at)) is not None for cx in ("atomic", "full") for at in (False, True)):
            changed_attribution = sum(
                group[(cx, False)]["predicted"] != group[(cx, True)]["predicted"]
                for cx in ("atomic", "full")
            )
            changed_complexity = sum(
                group[("atomic", at)]["predicted"] != group[("full", at)]["predicted"]
                for at in (False, True)
            )
            print(f"Attribution contrast label changes: {changed_attribution}/2; "
                  f"Complexity contrast changes: {changed_complexity}/2")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E5: 출처 귀속 표현 x atomic/full claim의 2x2 JEV-NLI 진단 실험")
    ap.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    ap.add_argument("--pair", choices=[f"BIO_{i:03d}" for i in range(1, 5)], default="BIO_001")
    ap.add_argument("--all-bio", action="store_true", help="BIO_001~004 A/B 총 32개 조건으로 확장")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-saved", action="store_true", help="저장된 JSONL의 3개 확률을 API 재호출 없이 표시")
    args = ap.parse_args()
    if args.limit < 0 or args.sleep < 0:
        ap.error("--limit과 --sleep은 0 이상이어야 합니다.")
    if args.out.resolve() == args.summary.resolve():
        ap.error("--out과 --summary가 같으면 안 됩니다.")
    if args.show_saved:
        if not args.out.exists():
            raise FileNotFoundError(f"기존 결과 파일이 없습니다: {args.out}")
        rows = jsonl_rows(args.out)
        print(f"기존 결과 {len(rows)}개: {args.out} (API 미호출)")
        for r in rows:
            print(
                f"{r.get('case_id')} | {r.get('predicted')} | "
                f"P(ENT)={r.get('p_entailment')} | "
                f"P(CON)={r.get('p_contradiction')} | "
                f"P(NEU)={r.get('p_neutral')}"
            )
        return
    cases = make_cases(args.input, args.pair, args.all_bio)
    if args.limit:
        cases = cases[:args.limit]
    print(f"E5 Attribution x Complexity: {len(cases)} cases; "
          f"scope={'ALL BIO A/B' if args.all_bio else args.pair + ' A/B'}")
    print(f"Axiom SHA256={hashlib.sha256(DOMAIN_AXIOMS.encode('utf-8')).hexdigest()[:16]}")
    print(f"JSONL output={args.out}; summary={args.summary}")

    if args.dry_run:
        print(f"[DOMAIN_AXIOMS] {DOMAIN_AXIOMS}")
        for c in cases:
            print(f"  {c['case_id']} gold={c['gold']}\n    {c['hypothesis']}")
        return

    load_dotenv()
    api_key = os.getenv("JEV_API_KEY")
    if not api_key:
        raise RuntimeError("프로젝트 루트의 .env에 JEV_API_KEY를 설정해 주세요.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = jsonl_rows(args.out) if args.out.exists() else []
    done: dict[str, dict] = {}
    for row in existing:
        pid = row.get("case_id")
        if not pid or pid in done:
            raise RuntimeError(f"결과 파일의 중복/비정상 case_id: {pid}")
        done[pid] = row

    with args.out.open("a", encoding="utf-8") as f:
        for i, c in enumerate(cases, 1):
            cid = c["case_id"]
            if cid in done:
                old = done[cid]
                if old.get("input_sha256") != c["input_sha256"] or old.get("experiment_id") != EXPERIMENT_ID:
                    raise RuntimeError(f"기존 결과와 현재 설정이 달라 중단합니다: {cid}. 출력을 별도 파일로 지정하세요.")
                missing = [k for k in ("p_entailment", "p_contradiction", "p_neutral") if old.get(k) is None]
                extra = f"; 저장된 확률 누락: {', '.join(missing)}" if missing else ""
                print(f"[{i}/{len(cases)}] {cid} SKIP (existing {old['predicted']}{extra})")
                continue
            res = call_jev(DOMAIN_AXIOMS, c["hypothesis"], api_key)
            label = res["label"]
            if label not in ("ENT", "CON", "NEU"):
                raise RuntimeError(f"{cid}: 알 수 없는 JEV label: {label}")
            saved = {
                **c,
                "predicted": label,
                "p_entailment": res.get("p_entailment"),
                "p_contradiction": res.get("p_contradiction"),
                "p_neutral": res.get("p_neutral"),
            }
            if saved["p_contradiction"] is None or saved["p_neutral"] is None:
                raise RuntimeError(
                    f"{cid}: JEV 응답에 P(CON) 또는 P(NEU)가 없습니다. "
                    "src/jev_client.py에서 두 확률을 반환하도록 수정했는지 확인하세요. "
                    "이 결과는 저장되지 않았습니다."
                )
            f.write(json.dumps(saved, ensure_ascii=False) + "\n")
            f.flush()
            done[cid] = saved
            print(
                f"[{i}/{len(cases)}] {cid} gold={c['gold']} -> {label} "
                f"P(ENT)={saved['p_entailment']} "
                f"P(CON)={saved['p_contradiction']} "
                f"P(NEU)={saved['p_neutral']}"
            )
            time.sleep(args.sleep)

    summary = print_report(cases, done)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump({"experiment_id": EXPERIMENT_ID,
                   "scope": "all_bio" if args.all_bio else args.pair,
                   "selected_cases": len(cases),
                   "axioms_sha256": hashlib.sha256(DOMAIN_AXIOMS.encode("utf-8")).hexdigest(),
                   **summary}, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {args.out.resolve()}")


if __name__ == "__main__":
    main()
