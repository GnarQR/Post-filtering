from pathlib import Path
import json

path = Path("data/generated_qna.jsonl")
bad_id = "08_농업기술길잡이_육우_p290"
tmp = Path("data/generated_qna_tmp.jsonl")

total = 0
removed = 0

with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8", newline="\n") as fout:
    for line in fin:
        if not line.strip():
            continue

        obj = json.loads(line)
        total += 1

        if obj.get("sample_id") == bad_id:
            removed += 1
            print("제거:", bad_id)
            continue

        fout.write(line if line.endswith("\n") else line + "\n")

tmp.replace(path)

print(f"원본: {total}개")
print(f"제거: {removed}개")
print(f"남음: {total - removed}개")