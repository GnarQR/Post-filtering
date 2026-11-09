"""
show_page_text.py

match_qna_to_pages.py가 저장한 qna_page_candidates.json을 보고, 특정 문항의
후보 페이지 원문(pages.jsonl의 text 필드)을 그대로 출력. 이 후보가 실제로
쓸 만한 근거인지 사람이 눈으로 확인하는 용도.

사용법:
  python show_page_text.py --qna-num 3                     # 3번 문항의 상위 후보 전부
  python show_page_text.py --qna-num 3 --rank 1             # 1위 후보만
  python show_page_text.py --doc 01_한국가축사양표준_한우_2022 --page 35   # 문서+페이지 직접 지정
"""

import argparse
import json
from pathlib import Path


def load_jsonl_robust(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text, strict=False)
    decoder = json.JSONDecoder(strict=False)
    objs, idx, n = [], 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def get_page_text(ocr_root: Path, doc: str, page: int):
    pf = ocr_root / doc / "pages.jsonl"
    if not pf.exists():
        return None
    for p in load_jsonl_robust(pf):
        if p["page"] == page:
            return p["text"]
    return None


def main(ocr_root: Path, candidates_path: Path, qna_num, rank, doc, page):
    if doc and page:
        text = get_page_text(ocr_root, doc, page)
        print(f"=== {doc} p.{page} ===\n")
        print(text if text else "!! 못 찾음")
        return

    report = json.loads(candidates_path.read_text(encoding="utf-8"))
    item = next((r for r in report if str(r["id"]) == str(qna_num)), None)
    if not item:
        print(f"!! {qna_num}번 문항을 candidates 파일에서 못 찾음")
        return

    print(f"[{item['id']}] {item['question']}\n")
    tops = item["top"]
    if rank:
        tops = [tops[rank - 1]]

    for i, cand in enumerate(tops, start=(rank or 1)):
        text = get_page_text(ocr_root, cand["doc"], cand["page"])
        print(f"\n{'='*70}\n{i}위 후보: {cand['doc']} p.{cand['page']} (겹침 키워드: {', '.join(cand['keywords'])})\n{'='*70}")
        print(text if text else "!! 못 찾음")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", type=Path, default=Path("RAG 자료/OCR"))
    ap.add_argument("--candidates", type=Path, default=Path("qna_page_candidates.json"))
    ap.add_argument("--qna-num", default=None)
    ap.add_argument("--rank", type=int, default=None, help="이 순위 후보 하나만 (예: 1)")
    ap.add_argument("--doc", default=None, help="문서 폴더명 직접 지정")
    ap.add_argument("--page", type=int, default=None, help="페이지 번호 직접 지정")
    a = ap.parse_args()
    main(a.ocr_root, a.candidates, a.qna_num, a.rank, a.doc, a.page)
