"""
match_qna_to_pages.py

"RAG 자료/OCR/<문서폴더>/pages.jsonl" 20개를 전부 훑어서, 40문항 QnA 각각에
대해 어느 문서의 몇 페이지가 근거가 될 만한지 후보를 찾아줌.

방식: 질문+정답 텍스트에서 한글 키워드(2글자 이상)를 뽑고, 각 페이지 텍스트와
겹치는 단어 수로 점수를 매겨 문서별/전체 상위 후보를 출력. 완벽한 검색은 아니고
"사람이 확인할 후보를 좁혀주는" 용도 — 최종적으로는 사람이 실제 페이지 내용을
보고 premise로 쓸지 판단해야 함.

qna_ground_truth.json의 정확한 키 이름은 --q-key(질문)/--a-key(정답)/--id-key(번호)
로 지정 (기본값은 "질문"/"정답"/"번호" — inspect_qna_schema.py로 먼저 확인).

사용법:
  python match_qna_to_pages.py --ocr-root "RAG 자료/OCR" --top-k 5
  python match_qna_to_pages.py --ocr-root "RAG 자료/OCR" --qna-num 1   # 1번 문항만 자세히
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

STOPWORDS = set("""
그리고 그러나 또한 또는 등을 등의 등은 것은 것이 것을 있는 있습니다 합니다 하는 위해
대한 대해 경우 때문 이는 이를 통해 한다 하며 하고 있다 되는 됩니다 되어 관련 이상 이하
""".split())


def tokenize(text: str) -> set:
    words = re.findall(r"[가-힣]{2,}", text)
    return {w for w in words if w not in STOPWORDS}


def load_qna(path: Path, id_key: str, q_key: str, a_key: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else list(data.values())
    out = []
    for it in items:
        qid = it.get(id_key, it.get("id", "?"))
        q = it.get(q_key, "")
        a = it.get(a_key, "")
        out.append({"id": qid, "question": q, "answer": a, "tokens": tokenize(q + " " + a)})
    return out


def load_all_docs(ocr_root: Path):
    docs = {}
    for folder in sorted(ocr_root.iterdir()):
        pf = folder / "pages.jsonl"
        if not pf.exists():
            continue
        pages = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
        for p in pages:
            p["doc"] = folder.name
            p["tokens"] = tokenize(p["text"])
        docs[folder.name] = pages
    return docs


def score_pages(q_tokens: set, pages: list, top_k: int):
    scored = []
    for p in pages:
        overlap = q_tokens & p["tokens"]
        if overlap:
            scored.append((len(overlap), p["doc"], p["page"], overlap))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def main(ocr_root: Path, qna_path: Path, id_key, q_key, a_key, top_k, only_num, out_path):
    print("QnA 로딩...")
    qna = load_qna(qna_path, id_key, q_key, a_key)
    print(f"  {len(qna)}문항")

    print("문서 전체 로딩 중 (20개 폴더)...")
    docs = load_all_docs(ocr_root)
    all_pages = [p for pages in docs.values() for p in pages]
    print(f"  문서 {len(docs)}개, 전체 페이지 {len(all_pages)}개")

    report = []
    for item in qna:
        if only_num and str(item["id"]) != str(only_num):
            continue
        top = score_pages(item["tokens"], all_pages, top_k)
        report.append({"id": item["id"], "question": item["question"], "top": [
            {"doc": d, "page": p, "overlap": n, "keywords": sorted(ov)[:8]} for n, d, p, ov in top
        ]})
        print(f"\n[{item['id']}] {item['question'][:60]}")
        if not top:
            print("   !! 후보 없음 (키워드가 아예 안 겹침)")
        for n, d, p, ov in top:
            print(f"   {d} p.{p}  (겹침 단어 {n}개: {', '.join(sorted(ov)[:6])})")

    if out_path:
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장 완료: {out_path.resolve()}")

    zero = [r for r in report if not r["top"]]
    if zero and not only_num:
        print(f"\n=== 후보가 아예 없는 문항 {len(zero)}개 ===")
        for r in zero:
            print(f"  [{r['id']}] {r['question'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", type=Path, required=True, help='예: "RAG 자료/OCR"')
    ap.add_argument("--qna-path", type=Path, default=Path("qna_ground_truth.json"))
    ap.add_argument("--id-key", default="번호")
    ap.add_argument("--q-key", default="질문")
    ap.add_argument("--a-key", default="정답")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qna-num", default=None, help="이 번호 문항만 자세히 보기")
    ap.add_argument("--out", type=Path, default=Path("qna_page_candidates.json"))
    a = ap.parse_args()
    main(a.ocr_root, a.qna_path, a.id_key, a.q_key, a.a_key, a.top_k, a.qna_num, a.out)
