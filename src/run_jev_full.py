"""
run_jev_full.py

각 Q&A 샘플의 전체 answer를 한 번에 JEV로 판정해 raw 결과를 저장한다.
문장별 JEV 결과(run_jev.py)와 비교하기 위한 실험용 코드다.

- 여기서는 threshold를 적용하지 않는다.
- raw label / p_entailment만 저장한다.
- threshold 선택은 compare_jev_modes.py에서 수행한다.
- 기존 결과가 있으면 sample_id 기준으로 이어서 진행한다.

사용:
  python .\src\run_jev_full.py --limit 5
  python .\src\run_jev_full.py
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from jev_client import call_jev

load_dotenv()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path} line {line_no}: JSON decode error: {e}"
                ) from e
    return rows


def main(input_path: Path, out_path: Path, limit: int, sleep_sec: float):
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없음")

    samples = load_jsonl(input_path)
    if limit:
        samples = samples[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        for row in load_jsonl(out_path):
            done_ids.add(row["sample_id"])
        if done_ids:
            print(f"기존 full-answer 결과 {len(done_ids)}개 발견 -> 이어서 진행")

    with out_path.open("a", encoding="utf-8") as f:
        for i, s in enumerate(samples, start=1):
            sid = s["sample_id"]
            if sid in done_ids:
                continue

            premise_full = " ".join(s.get("premise_chunks", []))
            answer_full = s.get("answer", "").strip()
            if not answer_full:
                raise RuntimeError(f"{sid}: answer가 비어 있음")

            v = call_jev(premise_full, answer_full, key)

            row = {
                "sample_id": sid,
                "domain": s.get("domain"),
                "question": s.get("question"),
                "answer": answer_full,
                "jev_full_label": v["label"],
                "p_entailment": v["p_entailment"],
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            p = v["p_entailment"]
            p_text = "None" if p is None else f"{p:.3f}"
            print(
                f"[{i}/{len(samples)}] {sid} ({s.get('domain')}) "
                f"-> {v['label']} / p={p_text}"
            )

            time.sleep(sleep_sec)

    rows = load_jsonl(out_path)
    label_counts = {}
    for r in rows:
        label = r["jev_full_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n완료: {out_path.resolve()}")
    print(f"전체 full-answer 결과: {len(rows)}개")
    print(f"label 분포: {label_counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated_qna.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/jev_full_results.jsonl"),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="JEV API 호출 간 대기 시간(초)",
    )
    a = ap.parse_args()
    main(a.input, a.out, a.limit, a.sleep)
