"""Small diagnostic probe for cattle biological-role contradictions.

Run from the project root (where data/, results/, src/ and .env are located):
  python src/probe_jev_biological_relations.py --mode null-nli --dry-run
  python src/probe_jev_biological_relations.py --mode null-nli
  python src/probe_jev_biological_relations.py --mode atomic-choice
  python src/probe_jev_biological_relations.py --mode axiom-nli

Three experimental modes are MECHANISM PROBES, not independent test-set estimates.
A and B examples are taken from 4 biological_role base groups. The manually
extracted atomic statements are an ORACLE / feasibility upper-bound; do not treat
them as an automated extractor's performance.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

from jev_client import call_jev

API_URL = "https://api.typesafe.ai/v1/systemone"

# Manually curated short propositions. The contrast targets whether sentence
# length / distracting modifiers obscure the same biological contradiction.
ATOMS = {
    "BIO_001_A": "임신한 암소는 구리 요구량을 가진다.",
    "BIO_001_B": "임신한 수소는 구리 요구량을 가진다.",
    "BIO_002_A": "임신한 암소의 칼륨 요구량은 0.5~0.7%이다.",
    "BIO_002_B": "임신한 수소의 칼륨 요구량은 0.5~0.7%이다.",
    "BIO_003_A": "번식용 암소는 임신과 분만 관리의 대상이 된다.",
    "BIO_003_B": "번식용 수소는 임신과 분만 관리의 대상이 된다.",
    "BIO_004_A": "암소에게 수정란을 이식하여 임신시킬 수 있다.",
    "BIO_004_B": "수소에게 수정란을 이식하여 임신시킬 수 있다.",
}

# These prevent superficial '수소 + 임신' co-occurrence matching.
CONTROL_ATOMS = [
    ("CTRL_001", "수소는 임신하지 않는다.", "VALID"),
    ("CTRL_002", "수소는 임신한 암소와 함께 사육될 수 있다.", "VALID"),
    ("CTRL_003", "수소의 정액은 암소의 임신에 사용될 수 있다.", "VALID"),
]

# Explicit background axioms. This mode is NOT 'context-free': it tests
# whether a small, vetted ontology can make an NLI classifier useful.
DOMAIN_AXIOMS = (
    "일반적인 소의 생식에서 암소는 암컷이고 수소는 수컷이다. "
    "암소는 임신할 수 있지만 수소는 임신할 수 없다. "
    "수정란을 이식받아 임신하는 개체는 암소이며 수소는 수정란을 이식받아 임신할 수 없다."
)


def load_examples(path: Path, include_controls: bool):
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cid = item["challenge_id"]
        if item["category"] == "biological_role" and item["condition"] in {"A", "B"}:
            cases.append({
                "probe_id": cid,
                "original_claim": item["answer"],
                "atomic_claim": ATOMS[cid],
                "gold": "VALID" if item["condition"] == "A" else "INVALID",
                "type": "pilot_pair",
            })
    if len(cases) != 8:
        raise RuntimeError(f"Expected 8 BIO A/B samples; got {len(cases)}")
    cases.sort(key=lambda x: x["probe_id"])
    if include_controls:
        cases.extend({"probe_id": pid, "original_claim": txt,
                      "atomic_claim": txt, "gold": gold, "type": "hard_negative_control"}
                     for pid, txt, gold in CONTROL_ATOMS)
    return cases


def call_choice(statement: str, key: str) -> dict:
    payload = {
        "model": "jev-latest",
        "state": statement,
        "questions": {
            "plausibility": {
                "type": "choice",
                "instructions": (
                    "검색 문서의 근거 여부가 아니라 소의 실제 생물학적 가능성을 판단하라. "
                    "주체와 생리 상태의 관계를 확인하고 명백히 불가능할 때만 INVALID를 선택하라. "
                    "단순히 '수소'와 '임신'이라는 단어가 함께 등장한다는 이유로 INVALID로 판정하지 마라."
                ),
                "criteria": {
                    "VALID": "서술한 생물학적 관계가 현실적으로 성립할 수 있음",
                    "INVALID": "동일 개체의 성별과 임신·분만 역할이 명백히 양립 불가능함",
                    "UNKNOWN": "검증할 정보가 충분하지 않거나 판단이 불확실함",
                },
            }
        },
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(5):
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            if response.status_code in {429, 500, 502, 503, 529}:
                last_error = f"HTTP {response.status_code}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            result = response.json()["answers"]["plausibility"]
            label = result["choice"].upper()
            if label not in {"VALID", "INVALID", "UNKNOWN"}:
                raise RuntimeError(f"Unexpected JEV label: {label}")
            probs = {k.upper(): v for k, v in result.get("probabilities", {}).items()}
            return {"label": label, "p_invalid": probs.get("INVALID")}
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"JEV choice API 5회 실패: {last_error}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["null-nli", "atomic-choice", "axiom-nli"], required=True)
    ap.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    selected = load_examples(args.input, include_controls=not args.no_controls)
    if args.limit:
        selected = selected[:args.limit]
    out = args.out or Path(f"results/jev_bio_probe_{args.mode.replace('-', '_')}.jsonl")
    print(f"Mode={args.mode} cases={len(selected)} output={out}")

    def statement_for(row):
        return row["original_claim"] if args.mode == "null-nli" else row["atomic_claim"]

    if args.dry_run:
        for r in selected:
            print(f"  {r['probe_id']} gold={r['gold']} | {statement_for(r)}")
        return

    key = os.getenv("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY를 지정하세요")
    out.parent.mkdir(parents=True, exist_ok=True)
    finished = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                finished[r["probe_id"]] = r
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(selected, 1):
            pid = row["probe_id"]
            if pid in finished:
                print(f"[{i}/{len(selected)}] {pid} already completed")
                continue
            stmt = statement_for(row)
            if args.mode == "null-nli":
                # Intentional empty document: classic NLI has no evidence here.
                result = call_jev("", stmt, key)
                mode_note = "empty_premise_nli"
            elif args.mode == "axiom-nli":
                result = call_jev(DOMAIN_AXIOMS, stmt, key)
                mode_note = "explicit_axioms_nli_not_context_free"
            else:
                result = call_choice(stmt, key)
                mode_note = "atomic_proposition_direct_plausibility_choice"
            saved = {"probe_id": pid, "type": row["type"], "gold": row["gold"],
                     "mode": args.mode, "mode_note": mode_note,
                     "input_statement": stmt, "original_claim": row["original_claim"],
                     "predicted": result["label"], "p_entailment": result.get("p_entailment"),
                     "p_invalid": result.get("p_invalid")}
            f.write(json.dumps(saved, ensure_ascii=False) + "\n")
            f.flush()
            finished[pid] = saved
            print(f"[{i}/{len(selected)}] {pid} gold={row['gold']} -> {result['label']}")
            time.sleep(args.sleep)

    summary = defaultdict(Counter)
    for row in selected:
        res = finished.get(row["probe_id"])
        if res:
            summary[res["gold"]][res["predicted"]] += 1
    print("\nCounts by gold (not an independent test-set performance estimate):")
    for gold in ["VALID", "INVALID"]:
        print(f"  {gold}: {dict(summary[gold])}")
    if args.mode in {"null-nli", "axiom-nli"}:
        print("NOTE: NLI's NEU is not a factual FALSE; do NOT equate it to INVALID.")
    else:
        print("NOTE: Atomic extraction was manually curated and is NOT a deployed extractor.")


if __name__ == "__main__":
    main()
