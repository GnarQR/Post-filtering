"""
inspect_ragbench_schema.py

RAGBench(hotpotqa 서브셋) 데이터셋을 불러와서 실제 컬럼 구조와 예시 1개를
출력. 이걸 먼저 돌려서 스키마를 확인한 뒤, 본 실험 스크립트의 필드명을
맞춰야 함 (huggingface.co 접속이 막힌 환경에서 작성해서, 정확한 컬럼명을
직접 검증하지 못했음 -> 이 스크립트로 먼저 확인 필요).

사용법:
  pip install datasets --break-system-packages   (필요시)
  python inspect_ragbench_schema.py
"""

from datasets import load_dataset

print("RAGBench(hotpotqa) 로딩 중...")
ds = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")

print(f"\n전체 행 수: {len(ds)}")
print(f"\n컬럼 목록: {ds.column_names}")

print("\n=== 예시 1개 (전체 구조) ===")
example = ds[0]
for key, value in example.items():
    value_str = str(value)
    if len(value_str) > 300:
        value_str = value_str[:300] + f"... (총 {len(value_str)}자)"
    print(f"\n[{key}] (타입: {type(value).__name__})")
    print(f"  {value_str}")

print("\n\n이 출력 결과를 클로한테 그대로 복사해서 보내줘 -> 정확한 컬럼명으로 본 실험 스크립트 완성할게.")
