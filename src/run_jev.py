"""
run_jev.py

generated_qna.jsonl(아직 사람 라벨 없음)의 각 답변 문장에 Jev 판정을 미리 매겨서
jev_scouted_qna.jsonl로 저장. gold 라벨이 없으니 정확도/재현율은 계산 못 함 -
"Jev가 어떻게 봤는지" 참고 자료를 미리 만들어, 나중에 라벨링할 때 의심스러운
샘플(Jev가 CON을 주거나 확신도가 낮은 문장이 있는 샘플)부터 먼저 볼 수 있게 하는 용도.

주의: 이 결과를 Jev 성능 검증에 쓰면 안 됨(gold가 없음). 어디까지나 라벨링
우선순위를 정하기 위한 사전 스캔. 라벨링할 때는 먼저 직접 판단하고 나서
Jev 결과와 비교하는 순서를 권장 (Jev를 먼저 보고 거기 맞춰 판단하면 편향 위험).

필요: .env에 JEV_API_KEY

사용법:
  python run_jev.py                      # 전체
  python run_jev.py --limit 5             # 먼저 5개만
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

JEV_THRESHOLD = 0.90  # Parameter sweep 실험 완료


def main(input_path: Path, out_path: Path, limit: int):
    from jev_client import call_jev
    import os
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없음")

    samples = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        samples = samples[:limit]

    done_ids = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done_ids.add(json.loads(l)["sample_id"])
        if done_ids:
            print(f"기존 결과 {len(done_ids)}개 발견 -> 이어서 진행")

    with out_path.open("a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if s["sample_id"] in done_ids:
                continue
            premise_full = " ".join(s["premise_chunks"])
            jev_sentences = []
            for sent in s["sentences"]:
                v = call_jev(premise_full, sent, key)
                jev_sentences.append({"sent": sent, "jev_label": v["label"], "p_entailment": v["p_entailment"]})
                time.sleep(0.3)

            row = dict(s)
            row["jev_sentences"] = jev_sentences
            row["jev_suspect"] = any(  # Parameter sweep 실험 완료
                js["jev_label"] != "ENT" or (js["p_entailment"] is not None and js["p_entailment"] < JEV_THRESHOLD) 
                for js in jev_sentences
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            flag = "  <- 의심" if row["jev_suspect"] else ""
            print(f"[{i+1}/{len(samples)}] {s['sample_id']} ({s['domain']}){flag}")
            for js in jev_sentences:
                mark = "⚠" if js["jev_label"] != "ENT" else " "
                p = js["p_entailment"]
                print(f"   {mark} [{js['jev_label']} {p if p is None else round(p,2)}] {js['sent'][:70]}")

    print(f"\n완료: {out_path.resolve()}")

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    suspects = [r for r in rows if r.get("jev_suspect")]
    print(
        f"\n전체 {len(rows)}개 샘플 중 {len(suspects)}개가 "
        f"'의심' 표시됨 "
        f"(JEV가 ENT가 아니거나 p_entailment<{JEV_THRESHOLD:.2f}인 문장 포함)"
    )
    print("라벨링할 때 이 샘플들부터 먼저 보면 의미 있는 케이스를 더 빨리 만날 수 있음:")
    for r in suspects:
        print(f"  - {r['sample_id']} ({r['domain']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/generated_qna.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/jev_results.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.input, a.out, a.limit)
