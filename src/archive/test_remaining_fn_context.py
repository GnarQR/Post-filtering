"""
test_remaining_fn_context.py

현재 Hybrid 규칙에서 남아 있는 False Negative(FN)를 찾고,
각 FN의 인접 답변 문장에 대해 B/C 실험을 수행한다.

현재 Hybrid 규칙:
  1) Sentence-level:
     문장 하나라도 label != ENT 또는 p_entailment < 0.90 이면 SUSPECT
  2) Full-answer:
     전체 답변 JEV label == CON 이면 SUSPECT
  3) 둘 중 하나라도 SUSPECT이면 최종 SUSPECT

B:
  이전 답변 문장 -> 현재 답변 문장

C:
  원문 + 이전 답변 문장 -> 현재 답변 문장

사용:
  python ./src/test_remaining_fn_context.py

결과:
  results/remaining_fn_context_test.jsonl
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


def sentence_suspect(sample, threshold: float) -> bool:
    for js in sample.get("jev_sentences", []):
        label = js.get("jev_label")
        p = js.get("p_entailment")

        if label != "ENT":
            return True

        if p is not None and p < threshold:
            return True

    return False


def hybrid_suspect(sentence_row, full_row, sentence_threshold: float) -> bool:
    sentence_flag = sentence_suspect(sentence_row, sentence_threshold)

    # 현재 실험에서 full-answer는 "명시적 contradiction"만 보조 신호로 사용
    full_contradiction = full_row.get("jev_full_label") == "CON"

    return sentence_flag or full_contradiction


def fmt_result(result):
    p = result.get("p_entailment")
    p_text = "None" if p is None else f"{p:.3f}"
    return f"{result.get('label')} / p={p_text}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--generated",
        type=Path,
        default=Path("data/generated_qna.jsonl"),
    )
    ap.add_argument(
        "--sentence",
        type=Path,
        default=Path("data/jev_results.jsonl"),
    )
    ap.add_argument(
        "--full",
        type=Path,
        default=Path("data/jev_full_results.jsonl"),
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=Path("data/reference_all_236_labeled.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/remaining_fn_context_test.jsonl"),
    )
    ap.add_argument(
        "--sentence-threshold",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.3,
    )

    args = ap.parse_args()

    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError(".env에 JEV_API_KEY가 없음")

    generated_rows = load_jsonl(args.generated)
    sentence_rows = load_jsonl(args.sentence)
    full_rows = load_jsonl(args.full)
    gold_rows = load_jsonl(args.gold)

    generated_map = {r["sample_id"]: r for r in generated_rows}
    sentence_map = {r["sample_id"]: r for r in sentence_rows}
    full_map = {r["sample_id"]: r for r in full_rows}
    gold_map = {r["sample_id"]: r for r in gold_rows}

    common_ids = (
        set(generated_map)
        & set(sentence_map)
        & set(full_map)
        & set(gold_map)
    )

    if not common_ids:
        raise RuntimeError("공통 sample_id가 없음")

    # 현재 Hybrid가 놓친 실제 Unsupported 샘플만 찾기
    fn_ids = []

    for sid in sorted(common_ids):
        gold_unsupported = (
            gold_map[sid].get("sample_reference_label") == "Unsupported"
        )

        if not gold_unsupported:
            continue

        pred_suspect = hybrid_suspect(
            sentence_map[sid],
            full_map[sid],
            args.sentence_threshold,
        )

        if not pred_suspect:
            fn_ids.append(sid)

    print(f"\n현재 Hybrid의 남은 FN: {len(fn_ids)}개")
    for sid in fn_ids:
        print(f"- {sid}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as fout:
        for sid in fn_ids:
            g = generated_map[sid]
            srow = sentence_map[sid]
            frow = full_map[sid]

            premise_full = " ".join(g.get("premise_chunks", []))
            sentences = g.get("sentences", [])

            print("\n" + "=" * 100)
            print(f"[FN] {sid}")
            print(f"Question: {g.get('question')}")
            print(f"Answer: {g.get('answer')}")

            print("\n[기존 Sentence-level JEV]")
            for i, js in enumerate(srow.get("jev_sentences", []), start=1):
                print(
                    f"  S{i}: {js.get('jev_label')} "
                    f"/ p={js.get('p_entailment')}"
                )
                print(f"      {js.get('sent')}")

            print("\n[기존 Full-answer JEV]")
            print(
                f"  {frow.get('jev_full_label')} "
                f"/ p={frow.get('p_entailment')}"
            )

            sample_result = {
                "sample_id": sid,
                "question": g.get("question"),
                "answer": g.get("answer"),
                "sentence_level": srow.get("jev_sentences", []),
                "full_answer": {
                    "label": frow.get("jev_full_label"),
                    "p_entailment": frow.get("p_entailment"),
                },
                "adjacent_tests": [],
            }

            if len(sentences) < 2:
                print("\n답변 문장이 1개라 B/C 인접 문장 실험을 할 수 없음.")
                fout.write(
                    json.dumps(sample_result, ensure_ascii=False) + "\n"
                )
                continue

            print("\n[B/C 인접 문장 실험]")

            # S1->S2, S2->S3, ...
            for i in range(1, len(sentences)):
                prev_sent = sentences[i - 1]
                curr_sent = sentences[i]

                # B: 직전 답변 문장만 premise
                b = call_jev(prev_sent, curr_sent, key)
                time.sleep(args.sleep)

                # C: 원문 + 직전 답변 문장을 premise
                c_premise = (
                    premise_full
                    + "\n\n[직전 답변 문장]\n"
                    + prev_sent
                )
                c = call_jev(c_premise, curr_sent, key)
                time.sleep(args.sleep)

                print(f"\n  Pair S{i} -> S{i+1}")
                print(f"    Prev: {prev_sent}")
                print(f"    Curr: {curr_sent}")
                print(f"    B = {fmt_result(b)}")
                print(f"    C = {fmt_result(c)}")

                sample_result["adjacent_tests"].append({
                    "from_sentence_index": i,
                    "to_sentence_index": i + 1,
                    "previous_sentence": prev_sent,
                    "current_sentence": curr_sent,
                    "B_previous_to_current": {
                        "label": b.get("label"),
                        "p_entailment": b.get("p_entailment"),
                    },
                    "C_source_plus_previous_to_current": {
                        "label": c.get("label"),
                        "p_entailment": c.get("p_entailment"),
                    },
                })

            fout.write(
                json.dumps(sample_result, ensure_ascii=False) + "\n"
            )
            fout.flush()

    print(f"\n저장 완료: {args.out.resolve()}")


if __name__ == "__main__":
    main()
