"""
sample_pages.py

축우_자료인벤토리.xlsx(수집목록 시트)로 "OCR 폴더명 -> 도메인/축종범위"를 매핑하고,
RAG 자료/OCR 아래 18개 문서에서 도메인별로 페이지를 층화 샘플링.

- front_matter/back_matter 페이지, 글자 수가 너무 적은 페이지(표지/목차 등 추정)는 제외
- --window로 앞뒤 인접 페이지를 묶어서 premise_chunks로 구성 가능
  (예: --window 1 -> 가운데 페이지 + 앞 1페이지 + 뒤 1페이지, 총 3청크)
- 결과를 sampled_pages.jsonl로 저장 (이어하기 불필요 - 매번 새로 샘플링되는 가벼운 작업,
  이미 있으면 덮어쓰지 않고 그대로 사용 -> 재현성 보장)

사용법:
  python sample_pages.py --inventory 축우_자료인벤토리.xlsx --ocr-root "rag_data/OCR" --n-per-domain 5
  python sample_pages.py --n-per-domain 3 --window 1   # 인접 페이지 포함
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


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


def build_folder_domain_map(inventory_path: Path, ocr_root: Path):
    """연번(숫자) -> (도메인, 축종범위, 자료명)을 읽고, 실제 OCR 폴더명과
    연번 접두사(예: '01_')로 매칭."""
    df = pd.read_excel(inventory_path, sheet_name="수집목록")
    by_num = {int(r["연번"]): (r["도메인"], r["축종범위"], r["자료명"]) for _, r in df.iterrows()}

    mapping = {}
    for folder in sorted(ocr_root.iterdir()):
        if not (folder / "pages.jsonl").exists():
            continue
        prefix = folder.name.split("_", 1)[0]
        try:
            num = int(prefix)
        except ValueError:
            print(f"  !! {folder.name}: 번호 접두사를 못 읽음, 건너뜀")
            continue
        if num not in by_num:
            print(f"  !! {folder.name}: 인벤토리에 연번 {num}이 없음, 건너뜀")
            continue
        domain, scope, title = by_num[num]
        mapping[folder.name] = {"domain": domain, "scope": scope, "title": title}
    return mapping


def collect_usable_pages(ocr_root: Path, doc_meta: dict, min_chars: int):
    """도메인별로 (doc, page_idx_in_list, page_num) 후보 리스트 생성."""
    by_domain = defaultdict(list)
    doc_pages = {}  # doc -> 전체 page 레코드 리스트 (window 조립용)
    for doc, meta in doc_meta.items():
        pages = load_jsonl_robust(ocr_root / doc / "pages.jsonl")
        pages.sort(key=lambda p: p["page"])
        doc_pages[doc] = pages
        for i, p in enumerate(pages):
            if p.get("page_class") != "content":
                continue
            if p.get("chars", 0) < min_chars:
                continue
            by_domain[meta["domain"]].append((doc, i))
    return by_domain, doc_pages


def make_premise(doc_pages: dict, doc: str, center_idx: int, window: int):
    pages = doc_pages[doc]
    lo = max(0, center_idx - window)
    hi = min(len(pages), center_idx + window + 1)
    chunk = pages[lo:hi]
    return [p["text"] for p in chunk], [p["page"] for p in chunk]


def main(inventory: Path, ocr_root: Path, n_per_domain: int, window: int, min_chars: int, seed: int, out_path: Path):
    print("인벤토리로 문서-도메인 매핑 구성 중...")
    doc_meta = build_folder_domain_map(inventory, ocr_root)
    print(f"  매핑된 문서 {len(doc_meta)}개")
    for d, m in doc_meta.items():
        print(f"    {d}: 도메인={m['domain']}, 축종범위={m['scope']}")

    print("\n사용 가능한 페이지 수집 중...")
    by_domain, doc_pages = collect_usable_pages(ocr_root, doc_meta, min_chars)
    for domain, cands in by_domain.items():
        print(f"  {domain}: 후보 {len(cands)}페이지")

    rng = random.Random(seed)
    samples = []
    for domain, cands in by_domain.items():
        n = min(n_per_domain, len(cands))
        if n < n_per_domain:
            print(f"  !! {domain}: 요청한 {n_per_domain}개보다 후보가 적어 {n}개로 줄임")
        picked = rng.sample(cands, n)
        for doc, idx in picked:
            chunks, page_nums = make_premise(doc_pages, doc, idx, window)
            samples.append({
                "sample_id": f"{doc}_p{page_nums[len(page_nums)//2] if len(page_nums)>1 else page_nums[0]}",
                "doc": doc, "domain": domain, "scope": doc_meta[doc]["scope"],
                "title": doc_meta[doc]["title"], "pages": page_nums,
                "premise_chunks": chunks,
            })
    rng.shuffle(samples)

    out_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples), encoding="utf-8")
    print(f"\n샘플 {len(samples)}개 저장: {out_path.resolve()}")
    from collections import Counter
    print("도메인별 분포:", dict(Counter(s["domain"] for s in samples)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=Path("축우_자료인벤토리.xlsx"))
    ap.add_argument("--ocr-root", type=Path, default=Path("rag_data/OCR"))
    ap.add_argument("--n-per-domain", type=int, default=5)
    ap.add_argument("--window", type=int, default=0, help="가운데 페이지 기준 앞뒤로 몇 페이지씩 더 포함할지")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=Path, default=Path("sampled_pages_raw.jsonl"))
    a = ap.parse_args()
    main(a.inventory, a.ocr_root, a.n_per_domain, a.window, a.min_chars, a.seed, a.out)
