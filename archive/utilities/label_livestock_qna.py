"""
label_livestock_qna.py

generate_livestock_qna.py가 만든 generated_qna.jsonl을 한 샘플씩 터미널에
띄워서, 문서(premise)를 보고 생성된 답변의 각 문장이 지지(s)/환각(h)/보류(n)인지
직접 라벨링. 결과는 기존 파이프라인(collect_features.py, add_jev_to_ragtruth.py
등)과 동일한 스키마로 저장 -> 바로 Jev/NLI 채점에 재사용 가능.

조작:
  s = 지지(supported)       h = 환각(hallucinated)
  n = 판단 보류(건너뜀, 결과에 저장 안 함)
  p = 문서(premise) 다시 보기
  b = 이 샘플 처음부터 다시
  q = 저장하고 종료 (다음 실행 시 이어서 진행)

이어하기: 이미 완료한 sample_id는 건너뜀. 한 샘플 중간에 멈춰도, 그 샘플은
처음부터 다시 라벨링(부분 저장 안 함 -> 샘플 하나 단위로만 진행 상황 기록).

사용법:
  python label_livestock_qna.py --input generated_qna.jsonl --out fixed_sample_livestock_pilot.jsonl
"""

import argparse
import json
import textwrap
from pathlib import Path

SUBSET_NAME = "livestock_pilot"


def wrap(text, width=90, indent="  "):
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


def show_premise(sample):
    print("\n" + "=" * 78)
    print(f"[문서] {sample['title']}  (도메인: {sample['domain']}, 축종: {sample['scope']})  p.{sample['pages']}")
    print("=" * 78)
    for i, chunk in enumerate(sample["premise_chunks"]):
        tag = " <- 중심 페이지" if len(sample["premise_chunks"]) > 1 and i == len(sample["premise_chunks"]) // 2 else ""
        print(f"\n--- 청크 {i+1}{tag} ---")
        print(wrap(chunk))
    print("=" * 78 + "\n")


def label_one_sample(sample):
    while True:
        show_premise(sample)
        print(f"[질문] {sample['question']}\n")
        print(f"[생성된 전체 답변]\n{wrap(sample['answer'])}\n")
        print(f"총 {len(sample['sentences'])}개 문장을 하나씩 판정합니다. (s=지지 h=환각 n=보류 p=문서다시보기 b=처음부터 q=저장후종료)\n")

        results = []
        restart = False
        i = 0
        while i < len(sample["sentences"]):
            sent = sample["sentences"][i]
            print(f"\n[{i+1}/{len(sample['sentences'])}] {sent}")
            ans = input("  > ").strip().lower()
            if ans == "s":
                results.append({"sent": sent, "gold": True})
                i += 1
            elif ans == "h":
                results.append({"sent": sent, "gold": False})
                i += 1
            elif ans == "n":
                i += 1  # 저장 안 하고 건너뜀
            elif ans == "p":
                show_premise(sample)
                continue
            elif ans == "b":
                restart = True
                break
            elif ans == "q":
                return results, "quit"
            else:
                print("  !! s/h/n/p/b/q 중 하나를 입력해줘")
                continue
        if restart:
            continue
        return results, "done"


def main(input_path: Path, out_path: Path):
    samples = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    done_ids = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                row = json.loads(l)
                done_ids.add(row.get("_sample_id"))
    todo = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"전체 {len(samples)}개 중 {len(done_ids)}개 완료, {len(todo)}개 남음\n")

    with out_path.open("a", encoding="utf-8") as f:
        for idx, sample in enumerate(todo):
            print(f"\n########## 샘플 {idx+1}/{len(todo)} ##########")
            results, status = label_one_sample(sample)
            if status == "quit":
                # 진행 중이던 샘플은 일부만 라벨링됐어도 저장하지 않음(=미완료로 남김).
                # sample_id가 출력 파일에 전혀 안 남아야, 다음 실행 때 done_ids 체크에서
                # "완료"로 잘못 판정돼 영구히 건너뛰어지는 일이 없음 -> 다음 실행 시 처음부터 다시.
                print(f"\n현재 샘플은 저장하지 않고 종료 (다음 실행 시 이 샘플부터 다시 시작). "
                      f"지금까지 완료한 샘플은 저장됨: {out_path.resolve()}")
                return
            for r in results:
                f.write(json.dumps({
                    "premise_chunks": sample["premise_chunks"],
                    "sent": r["sent"], "gold": r["gold"], "subset": SUBSET_NAME,
                    "question": sample["question"], "doc": sample["doc"],
                    "domain": sample["domain"], "pages": sample["pages"],
                    "_sample_id": sample["sample_id"],
                }, ensure_ascii=False) + "\n")
            f.flush()
    print(f"\n전체 완료! 저장 위치: {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("generated_qna.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("fixed_sample_livestock_pilot.jsonl"))
    a = ap.parse_args()
    main(a.input, a.out)
