"""
filter_pages.py

sample_livestock_pages.py가 뽑은 페이지 중 '숫자표 / 표 중심 / 깨진 OCR' 페이지를
텍스트 지표로 걸러내고, 같은 문서에서 페이지 창(window)이 겹치는 샘플을 제거한 뒤,
도메인별 목표 개수만큼만 남긴다. (tables_detected 필드에 의존하지 않음)

판정은 '중앙 페이지'(sample_id 끝의 _p<번호>)의 텍스트 기준:
  - 너무짧음   : 공백 제외 글자 < 150
  - OCR깨짐    : 한글/한자/ASCII가 아닌 문자 비율 > 3% 이거나 제어문자 > 0.5%
  - 숫자표     : 한글 비율 < 10%        (KPN 육종가 표, 숫자 위주 표)
  - 표중심     : markdown 표('|' 시작 줄) 비중 > 60% 이면서 한글 비율 < 30%
임계값은 60개 샘플(눈으로 확인한 불량 페이지 14개)로 맞춘 값. 스크린샷 잔해 같은
일부 깨진 OCR은 못 거를 수 있음(알고 있는 한계).

사용법:
  python filter_pages.py --input sampled_pages_raw.jsonl --out sampled_pages.jsonl \
      --target-per-domain 75
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

def load_jsonl_robust(path: Path):
    text = path.read_text(encoding="utf-8")

    stripped = text.lstrip()

    # JSON 배열 형식도 대응
    if stripped.startswith("["):
        return json.loads(text, strict=False)

    # JSON 객체가 연속으로 저장된 형식 대응
    decoder = json.JSONDecoder(strict=False)

    objs = []
    idx = 0
    n = len(text)

    while idx < n:
        # 객체 사이의 일반 공백만 건너뜀
        while idx < n and text[idx] in " \t\r\n":
            idx += 1

        if idx >= n:
            break

        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패 위치: {idx}")
            print(repr(text[max(0, idx-100):idx+300]))
            raise e

        objs.append(obj)
        idx = end

    return objs

SEP = "\n\n--- [Chunk 구분] ---\n\n"


def _is_hangul(c):
    o = ord(c)
    return 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def _is_han(c):
    return 0x4E00 <= ord(c) <= 0x9FFF


def page_reason(text, min_chars=150, min_hangul=0.10, pipe_ratio=0.6, pipe_hangul=0.30):
    """불량이면 사유 문자열, 정상이면 None."""
    ns = [c for c in text if not c.isspace()]
    n = max(len(ns), 1)
    if len(ns) < min_chars:
        return "너무짧음"
    hangul = sum(_is_hangul(c) for c in ns) / n
    foreign = sum(1 for c in ns if c.isalpha() and ord(c) > 127 and not _is_hangul(c) and not _is_han(c)) / n
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r") / n
    lines = [l for l in text.split("\n") if l.strip()]
    total = max(sum(len(l) for l in lines), 1)
    pipe = sum(len(l) for l in lines if l.lstrip().startswith("|")) / total

    if foreign > 0.03 or ctrl > 0.005:
        return "OCR깨짐"
    if hangul < min_hangul:
        return "숫자표(한글 낮음)"
    if pipe > pipe_ratio and hangul < pipe_hangul:
        return "표중심(markdown)"
    return None


def center_text(row):
    chunks = row["premise_chunks"]
    if isinstance(chunks, str):
        chunks = chunks.split(SEP)
    pages = row.get("pages") or []
    try:
        cp = int(row["sample_id"].rsplit("_p", 1)[-1])
        idx = pages.index(cp) if cp in pages else len(chunks) // 2
    except (ValueError, KeyError):
        idx = len(chunks) // 2
    return chunks[min(idx, len(chunks) - 1)]


def doc_key(row):
    return row.get("doc") or row["sample_id"].rsplit("_p", 1)[0]


def main(input_path, out_path, target, seed, **thr):
    rows = load_jsonl_robust(input_path)
    
    random.Random(seed).shuffle(rows)  # 겹침 제거 시 앞쪽 페이지로 쏠리지 않게

    kept = []
    taken_pages = defaultdict(set)          # 문서별로 이미 쓴 페이지
    n_raw = Counter(r["domain"] for r in rows)
    n_kept = Counter()
    dropped = defaultdict(Counter)          # domain -> reason -> n

    for r in rows:
        dom = r["domain"]
        why = page_reason(center_text(r), **thr)
        if why is None:
            pages = set(r.get("pages") or [])
            if pages & taken_pages[doc_key(r)]:
                why = "창 겹침"
        if why is None and n_kept[dom] >= target:
            why = "목표 초과(정상)"
        if why:
            dropped[dom][why] += 1
            continue
        kept.append(r)
        n_kept[dom] += 1
        taken_pages[doc_key(r)] |= set(r.get("pages") or [])

    kept.sort(key=lambda r: (r["domain"], r["sample_id"]))
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")

    print(f"{'도메인':<8}{'원본':>6}{'유지':>6}  제거 사유")
    for dom in sorted(n_raw):
        why = ", ".join(f"{k} {v}" for k, v in dropped[dom].most_common())
        print(f"{dom:<8}{n_raw[dom]:>6}{n_kept[dom]:>6}  {why}")
    print(f"\n합계: 원본 {len(rows)} -> 유지 {len(kept)}  ({out_path.resolve()})")
    short = [d for d in sorted(n_raw) if n_kept[d] < target]
    for d in short:
        print(f"⚠ {d}: 목표 {target}개 중 {n_kept[d]}개만 확보 (부족 {target - n_kept[d]}개)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("sampled_pages_raw.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("sampled_pages.jsonl"))
    ap.add_argument("--target-per-domain", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--min-hangul", type=float, default=0.10)
    ap.add_argument("--pipe-ratio", type=float, default=0.6)
    ap.add_argument("--pipe-hangul", type=float, default=0.30)
    a = ap.parse_args()
    main(a.input, a.out, a.target_per_domain, a.seed,
         min_chars=a.min_chars, min_hangul=a.min_hangul,
         pipe_ratio=a.pipe_ratio, pipe_hangul=a.pipe_hangul)
