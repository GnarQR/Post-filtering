"""
inspect_ragtruth_qa_example.py

RAGTruth에서 (1) task_type == 'QA'이면서 (2) 실제로 환각 라벨이 있는
예시 하나를 찾아, context/output/hallucination_labels를 잘리지 않은
전체 텍스트로 출력. 문장 단위 변환 로직을 설계하기 위한 확인용.

사용법:
  python inspect_ragtruth_qa_example.py
"""

from datasets import load_dataset

print("RAGTruth 로딩 중...")
ds = load_dataset("wandb/RAGTruth-processed")["train"]

qa_ds = ds.filter(lambda ex: ex["task_type"] == "QA")
print(f"\nQA 태스크 예시 수: {len(qa_ds)}")

# 환각 라벨이 있는(hallucination_labels가 빈 리스트가 아닌) 예시 하나 찾기
target = None
for ex in qa_ds:
    if ex["hallucination_labels"] not in ("[]", "", None):
        target = ex
        break

if target is None:
    print("!! QA 태스크 중 환각 라벨 있는 예시를 못 찾음 -> 빈 예시라도 하나 출력")
    target = qa_ds[0]

print("\n=== context (전체) ===")
print(target["context"])

print("\n=== output (전체) ===")
print(target["output"])

print("\n=== hallucination_labels (원본 문자열) ===")
print(target["hallucination_labels"])

print("\n=== hallucination_labels_processed ===")
print(target["hallucination_labels_processed"])

print("\n=== model / quality ===")
print(target["model"], "/", target["quality"])
