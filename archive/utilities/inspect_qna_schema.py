"""
inspect_qna_schema.py

qna_ground_truth.json(40문항 축우 QnA)의 실제 구조를 확인. 이 파일이 리스트인지
딕셔너리인지, 키 이름이 "번호"/"질문"/"정답"인지 다른 이름인지 등을 먼저 봐야
매칭 스크립트의 필드명을 정확히 맞출 수 있음.

사용법:
  python inspect_qna_schema.py
  python inspect_qna_schema.py --path 다른_경로.json
"""

import argparse
import json
from pathlib import Path


def main(path: Path):
    if not path.exists():
        print(f"!! {path} 없음. 같은 폴더에 있는 .json 파일 목록:")
        for p in Path(".").glob("*.json"):
            print(" -", p)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"최상위 타입: {type(data).__name__}")

    if isinstance(data, list):
        print(f"리스트 길이: {len(data)}")
        print("\n=== 첫 번째 항목 ===")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1500])
    elif isinstance(data, dict):
        keys = list(data.keys())
        print(f"최상위 키 개수: {len(keys)}")
        print(f"키 예시: {keys[:5]}")
        first_key = keys[0]
        print(f"\n=== data['{first_key}'] ===")
        print(json.dumps(data[first_key], ensure_ascii=False, indent=2)[:1500])
    else:
        print("예상 못한 구조:", str(data)[:500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("qna_ground_truth.json"))
    a = ap.parse_args()
    main(a.path)
