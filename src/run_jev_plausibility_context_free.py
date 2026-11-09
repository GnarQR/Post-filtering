"""Evaluate context-free plausibility with JEV on 40 unique A/B claims.

Project root:
  python .\\src\\run_jev_plausibility_context_free.py --dry-run
  python .\\src\\run_jev_plausibility_context_free.py --limit 2
  python .\\src\\run_jev_plausibility_context_free.py

Does NOT alter the existing jev_client.py or Semantic Guardrail.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from jev_client import JEV_ENDPOINT

load_dotenv()

PROMPT_VERSION = "jev_context_free_plausibility_v1"
# Intentional: no evidence or question context is given to this verifier.
PLAUSIBILITY_QUESTION = {
    "type": "choice",
    "instructions": (
        "다음 축우(소) 관련 진술의 현실적 성립 가능성을 판단하세요. "
        "생물학적 성별·역할, 출생·분만의 시간 순서, 혈연관계, "
        "물리량의 부호·비율, 단위와 크기를 검토하세요. "
        "문장 속 '문서에 따르면' 같은 표현은 진술의 출처 표시일 뿐, "
        "그 내용이 사실이라는 근거로 사용하지 마세요. "
        "확실히 불가능한 주장 하나라도 있으면 invalid로 판정하고, "
        "드물거나 생소하다는 이유만으로 invalid를 선택하지 마세요. "
        "가능 여부를 판단할 정보가 부족하면 unknown을 선택하세요. "
        "인용된 가정이나 진술도 현실에서 성립할 수 있는지를 평가하세요."
    ),
    "criteria": {
        "valid": "현실의 생물학적·시간적·관계적·물리적 제약과 모순되지 않는 진술",
        "invalid": "명확한 생물학적·시간적·관계적·물리적 불가능성을 포함하는 진술",
        "unknown": "정보가 부족하거나 불가능성을 신뢰성 있게 판단할 수 없는 진술",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON 오류: {path}:{i}: {exc}") from exc
    return rows


def build_cases(pilot: list[dict]) -> list[dict]:
    """A normal and B corrupt claim per group; C/D are identical texts."""
    by_group = defaultdict(dict)
    for row in pilot:
        gid, condition = row["group_id"], row["condition"]
        if condition in by_group[gid]:
            raise ValueError(f"중복 challenge: {gid}_{condition}")
        by_group[gid][condition] = row

    cases = []
    for gid, conditions in sorted(by_group.items()):
        if set(conditions) != {"A", "B", "C", "D"}:
            raise ValueError(f"{gid}: A/B/C/D 불완전: {sorted(conditions)}")
        a, b, c, d = (conditions[k] for k in "ABCD")
        if a["answer"] != d["answer"] or b["answer"] != c["answer"]:
            raise ValueError(f"{gid}: A/D 또는 B/C 답변이 다름")
        for r in (a, b):
            cases.append({
                "case_id": r["challenge_id"],
                "group_id": gid,
                "source_condition": r["condition"],
                "category": r["category"],
                "violation": r["violation"],
                "claim": r["answer"],
                "gold": "VALID" if r["condition"] == "A" else "INVALID",
            })
    if len(cases) != 40 or Counter(c["gold"] for c in cases) != {"VALID": 20, "INVALID": 20}:
        raise ValueError("예상한 40개 균형 명제 데이터가 아님")
    return cases


def call_jev_plausibility(claim: str, key: str, model: str, max_retries: int = 5) -> dict:
    payload = {
        "model": model,
        "state": f"[검증할 진술]\n{claim}",
        "questions": {"plausibility": PLAUSIBILITY_QUESTION},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(JEV_ENDPOINT, headers=headers, json=payload, timeout=40)
            if response.status_code in (429, 500, 502, 503, 529):
                last_error = f"HTTP {response.status_code}: {response.text[:150]}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            ans = data["answers"]["plausibility"]
            choice = ans["choice"]
            if choice not in PLAUSIBILITY_QUESTION["criteria"]:
                raise ValueError(f"Unexpected JEV choice: {choice!r}")
            probs = ans.get("probabilities") or {}
            return {
                "jev_label": choice.upper(),
                "p_valid": probs.get("valid"),
                "p_invalid": probs.get("invalid"),
                "p_unknown": probs.get("unknown"),
                "jev_model": data.get("model", model),
                "usage": data.get("usage"),
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"JEV {max_retries}회 호출 실패: {last_error}")


def print_summary(cases: list[dict], results: list[dict]) -> None:
    expected = {x["case_id"] for x in cases}
    done = {x["case_id"]: x for x in results if x.get("case_id") in expected}
    evaluated = list(done.values())
    print(f"\n평가 완료 {len(evaluated)}/{len(cases)} (미완료는 결과에 포함하지 않음)")
    for gold in ("VALID", "INVALID"):
        matched = [r for r in evaluated if r["gold"] == gold]
        pred = Counter(r["jev_label"] for r in matched)
        n = len(matched)
        print(f"{gold:7s} n={n:2d} | VALID={pred['VALID']:2d} INVALID={pred['INVALID']:2d} UNKNOWN={pred['UNKNOWN']:2d}")
    for category in sorted({x["category"] for x in cases}):
        vals = [r for r in evaluated if r["category"] == category]
        hits = sum(r["jev_label"] == r["gold"] for r in vals)
        print(f"  {category:18s} exact={hits}/{len(vals)}")
    if len(evaluated) == len(cases):
        tp = sum(r["jev_label"] == "INVALID" and r["gold"] == "INVALID" for r in evaluated)
        fn = sum(r["jev_label"] != "INVALID" and r["gold"] == "INVALID" for r in evaluated)
        fp = sum(r["jev_label"] == "INVALID" and r["gold"] == "VALID" for r in evaluated)
        tn = sum(r["jev_label"] != "INVALID" and r["gold"] == "VALID" for r in evaluated)
        print(f"INVALID detection (UNKNOWN counts as not INVALID): TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"Recall={tp/(tp+fn):.3f} Specificity={tn/(tn+fp):.3f}")
        print("NOTE: UNKNOWN also requires separate REVIEW handling; don't treat it as confidently VALID.")


def main() -> None:
    parser = argparse.ArgumentParser(description="JEV context-free plausibility pilot (40 unique claims)")
    parser.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/jev_plausibility_context_free40.jsonl"))
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = build_cases(load_jsonl(args.input))
    to_run = cases[:args.limit] if args.limit > 0 else cases
    print(f"고유 claim={len(cases)}, 이번 선택={len(to_run)}, 범주={dict(Counter(c['category'] for c in cases))}")
    print(f"JEV model={args.model}; prompt={PROMPT_VERSION}; 출력={args.out}")
    if args.dry_run:
        for c in to_run:
            print(f"{c['case_id']}: gold={c['gold']} | {c['claim'][:90]}")
        return

    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없습니다")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.out) if args.out.exists() else []
    done = {r["case_id"]: r for r in existing}
    if len(done) != len(existing):
        raise RuntimeError("결과에 중복 case_id가 있음. 파일을 확인하세요")
    for c in cases:
        if c["case_id"] in done:
            old = done[c["case_id"]]
            if old.get("claim") != c["claim"] or old.get("prompt_version") != PROMPT_VERSION or old.get("requested_model") != args.model:
                raise RuntimeError(f"기존 결과와 조건 충돌: {c['case_id']}. 새 --out 경로 사용 권장")

    with args.out.open("a", encoding="utf-8") as file:
        for idx, c in enumerate(to_run, 1):
            if c["case_id"] in done:
                print(f"[{idx}/{len(to_run)}] {c['case_id']}: 저장된 결과 사용")
                continue
            result = call_jev_plausibility(c["claim"], key, args.model)
            row = {**c, "prompt_version": PROMPT_VERSION, "requested_model": args.model, **result}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            done[c["case_id"]] = row
            print(f"[{idx}/{len(to_run)}] {c['case_id']} gold={c['gold']} predicted={result['jev_label']} p_invalid={result['p_invalid']}")
            time.sleep(args.sleep)
    print_summary(cases, list(done.values()))


if __name__ == "__main__":
    main()
