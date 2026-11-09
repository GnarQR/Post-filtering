"""
inspect_ragtruth_schema.py

RAGTruth(사람이 직접 span 단위로 라벨링한 환각 탐지 데이터셋)를
wandb/RAGTruth-processed(HuggingFace 가공본)로 불러와서 실제 컬럼 구조와
예시 1개를 출력. inspect_ragbench_schema.py와 동일한 목적 -- 이걸 먼저
돌려서 스키마를 확인한 뒤, 특징 수집 스크립트의 필드명을 맞춰야 함.

RAGBench(rungalileo/ragbench)와는 컬럼 구조가 다를 가능성이 높음 -- 특히
"환각 여부"를 어떤 필드로 표시하는지(예: 문장 단위 라벨 vs span 단위
character offset)가 핵심 확인 포인트.

사용법:
  python inspect_ragtruth_schema.py
"""

from datasets import load_dataset

print("RAGTruth(wandb/RAGTruth-processed) 로딩 중...")
ds = load_dataset("wandb/RAGTruth-processed")

print(f"\n스플릿: {list(ds.keys())}")
for split_name, split_ds in ds.items():
    print(f"  {split_name}: {len(split_ds)}개")

first_split = list(ds.keys())[0]
print(f"\n컬럼 목록({first_split}): {ds[first_split].column_names}")

print(f"\n=== 예시 1개 (전체 구조, {first_split} 스플릿) ===")
example = ds[first_split][0]
for key, value in example.items():
    value_str = str(value)
    if len(value_str) > 300:
        value_str = value_str[:300] + f"... (총 {len(value_str)}자)"
    print(f"\n[{key}] (타입: {type(value).__name__})")
    print(f"  {value_str}")

print("\n\n이 출력 결과를 그대로 복사해서 보내줘 -> 정확한 컬럼명으로 특징 수집 스크립트 완성할게.")
