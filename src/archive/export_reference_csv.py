import csv
import json
from pathlib import Path


INPUT_PATH = Path("data/reference_all_236_labeled.jsonl")
OUTPUT_PATH = Path("results/reference_all_236_labeled.csv")


def normalize_null(value):
    """
    None 또는 문자열 'null'을 CSV에서는 빈칸으로 처리.
    """
    if value is None:
        return ""

    if isinstance(value, str) and value.strip().lower() == "null":
        return ""

    return value


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    sample_count = 0
    sentence_count = 0
    inconsistent_samples = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[JSON 오류] line {line_no}: {e}")
                continue

            sample_count += 1

            sample_id = sample.get("sample_id", "")
            domain = sample.get("domain", "")
            sample_label = sample.get("sample_reference_label", "")
            unsupported_count = sample.get("unsupported_sentence_count", 0)
            judge = sample.get("reference_judge", "")
            blind_to_jev = sample.get("blind_to_jev", "")

            sentence_labels = sample.get("sentence_labels", [])

            # 실제 sentence label 기준 Unsupported 개수 재계산
            actual_unsupported_count = sum(
                1
                for s in sentence_labels
                if s.get("reference_label") == "Unsupported"
            )

            # sample-level label과 sentence-level label이 일치하는지 확인
            expected_sample_label = (
                "Unsupported"
                if actual_unsupported_count > 0
                else "Supported"
            )

            is_consistent = (
                sample_label == expected_sample_label
                and unsupported_count == actual_unsupported_count
            )

            if not is_consistent:
                inconsistent_samples.append(sample_id)

            for sentence_info in sentence_labels:
                sentence_count += 1

                csv_rows.append({
                    "샘플_ID": sample_id,
                    "도메인": domain,
                    "샘플_최종판정": sample_label,
                    "미지원_문장수": unsupported_count,

                    # 사람이 볼 때 0번보다는 1번부터가 편해서 +1
                    "문장번호": sentence_info.get(
                        "sentence_index", 0
                    ) + 1,

                    "문장판정": sentence_info.get(
                        "reference_label", ""
                    ),

                    "문장": sentence_info.get(
                        "sent", ""
                    ),

                    "판정근거": sentence_info.get(
                        "reason", ""
                    ),

                    "오류유형": normalize_null(
                        sentence_info.get("error_type")
                    ),

                    "판정자": judge,
                    "JEV_비공개_평가": blind_to_jev,

                    # gold 데이터 검수용
                    "라벨_일관성": (
                        "OK"
                        if is_consistent
                        else "CHECK"
                    ),
                })

    fieldnames = [
        "샘플_ID",
        "도메인",
        "샘플_최종판정",
        "미지원_문장수",
        "문장번호",
        "문장판정",
        "문장",
        "판정근거",
        "오류유형",
        "판정자",
        "JEV_비공개_평가",
        "라벨_일관성",
    ]

    # utf-8-sig:
    # Windows Excel에서 한글 CSV를 바로 열어도 안 깨지게 함
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 70)
    print("Reference CSV export 완료")
    print("=" * 70)

    print(f"샘플 수   : {sample_count}")
    print(f"문장 수   : {sentence_count}")
    print(f"저장 위치 : {OUTPUT_PATH.resolve()}")

    print()

    if inconsistent_samples:
        print(
            f"[주의] sample/sentence label 불일치 "
            f"{len(inconsistent_samples)}개 발견"
        )

        for sid in inconsistent_samples:
            print(f"  - {sid}")
    else:
        print("모든 sample/sentence label 일관성 OK")


if __name__ == "__main__":
    main()