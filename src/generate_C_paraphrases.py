"""Gemini paraphrase generator for 20 corrupted C premises (manual review REQUIRED).

Project root usage:
  python .\\src\\generate_C_paraphrases.py --dry-run
  python .\\src\\generate_C_paraphrases.py --limit 2
  python .\\src\\generate_C_paraphrases.py

Output JSONL contains unapproved drafts, CSV must be manually approved before
running run_jev_C_paraphrase.py. Requires .env GEMINI_API_KEY for live calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VERSION = "c_paraphrase_preserve_corruption_v1"
PROMPT = """다음 소(축우) 관련 문장을 의미가 같지만 표면 표현이 다른 한국어 문장으로 1개 재서술하세요.
이 작업은 오류를 포함한 문장의 보존 실험입니다. 원문의 생물학적·시간적·관계적·수치적 오류를
절대 고치거나 완화하지 마세요. 상식적으로 틀렸더라도 동일한 틀린 명제를 유지해야 합니다.

반드시 지키세요:
1. 주체의 종·성별·개체 역할, 생리적 상태(임신·분만 등), 친자 관계의 방향을 바꾸지 마세요.
2. 사건의 전후·시점·부정 및 모든 숫자·부호·범위·단위를 그대로 유지하세요.
3. 정보 추가·삭제·추론·정당화 없이, 어순과 표현만 자연스럽게 바꾸세요.
4. '수소'를 '수컷 소'처럼 동의어로 바꾸는 것은 가능하나, '암소'로 수정하면 안 됩니다.
5. 원문의 문장을 그대로 복사하지 말고, 다른 표현을 사용하세요.
6. 결과 문장만 'paraphrase' 필드에 넣어 JSON 객체 하나로 출력하세요.

원문:
{claim}
"""

# Heuristic warnings are a triage aid, NEVER a substitute for manual review.
NUMBER = re.compile(r"(?<![A-Za-z0-9가-힣])[-+]?\d+(?:[.,]\d+)?")
UNIT = re.compile(r"(?<![A-Za-z가-힣])(?:kg|mg|g|톤|%|℃|°C|mL|ml|L)(?![A-Za-z])", re.I)
ROLE_TOKENS = (
    "수소", "암소", "송아지", "어미소", "아비", "외조부", "외조모", "수정란",
    "분만", "임신", "초유", "출생", "생후", "도체율", "급여", "섭취",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def selected_cases(pilot_path: Path) -> list[dict]:
    cases = [row for row in load_jsonl(pilot_path) if row["condition"] == "C"]
    if len(cases) != 20 or len({r["challenge_id"] for r in cases}) != 20:
        raise ValueError("C 조건 20개 또는 ID 고유성 오류")
    if any(r["premise"] != r["answer"] or r["plausibility_gold"] != "Implausible" for r in cases):
        raise ValueError("기본 C 조건은 동일한 corrupted premise/answer여야 합니다")
    return cases


def check_draft(original: str, rewrite: str) -> list[str]:
    warnings = []
    if not rewrite.strip():
        warnings.append("EMPTY")
    if re.sub(r"\s+", "", original) == re.sub(r"\s+", "", rewrite):
        warnings.append("IDENTICAL_TEXT")
    if sorted(NUMBER.findall(original)) != sorted(NUMBER.findall(rewrite)):
        warnings.append("NUMERIC_CHANGE_REVIEW")
    if sorted(x.lower() for x in UNIT.findall(original)) != sorted(x.lower() for x in UNIT.findall(rewrite)):
        warnings.append("UNIT_CHANGE_REVIEW")
    for token in ROLE_TOKENS:
        if token in original and token not in rewrite:
            warnings.append(f"ROLE_WORD_CHANGED:{token}")
    return warnings


def call_gemini(claim: str, model_name: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai 라이브러리 필요: pip install google-generativeai") from exc
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("프로젝트 .env에 GEMINI_API_KEY를 설정하세요")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name, generation_config={
        "temperature": 0.2, "response_mime_type": "application/json",
    })
    last = None
    for attempt in range(4):
        try:
            response = model.generate_content(PROMPT.format(claim=claim))
            obj = json.loads(response.text.strip())
            rewrite = obj.get("paraphrase")
            if not isinstance(rewrite, str) or not rewrite.strip():
                raise ValueError(f"잘못된 paraphrase 응답: {obj!r}")
            return rewrite.strip()
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini 4회 요청 실패: {last}")


FIELDS = ["case_id", "group_id", "category", "violation", "premise", "generated_answer",
          "auto_warnings", "approved", "edited_answer", "reviewer_notes", "model", "prompt_version"]


def create_review_csv(jsonl_rows: list[dict], review_path: Path) -> None:
    """Sync new drafts into review CSV, preserving existing human decisions/edits."""
    review_path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if review_path.exists():
        with review_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["case_id"] in old:
                    raise ValueError(f"검수 CSV의 case_id가 중복됨: {row['case_id']}")
                old[row["case_id"]] = row
    merged = []
    for row in jsonl_rows:
        current = {
            "case_id": row["case_id"], "group_id": row["group_id"],
            "category": row["category"], "violation": row["violation"],
            "premise": row["premise"], "generated_answer": row["generated_answer"],
            "auto_warnings": " | ".join(row["auto_warnings"]),
            "approved": "", "edited_answer": "", "reviewer_notes": "",
            "model": row["model"], "prompt_version": VERSION,
        }
        previous = old.pop(row["case_id"], None)
        if previous:
            for key in ("premise", "generated_answer", "prompt_version", "model"):
                if previous.get(key) != current[key]:
                    raise ValueError(f"검수 원본이 변경됨: {row['case_id']} ({key})")
            for key in ("approved", "edited_answer", "reviewer_notes"):
                current[key] = previous.get(key, "")
        merged.append(current)
    if old:
        raise ValueError(f"검수 CSV에 새 JSONL에서 찾을 수 없는 항목: {list(old)}")
    # Write atomically so partial-generation reruns cannot destroy manual edits.
    temp = review_path.with_suffix(review_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    temp.replace(review_path)
    print(f"검수 CSV 동기화: {review_path} ({len(merged)}개; 기존 검수 내용 보존)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/plausibility_pilot80.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/plausibility_C_paraphrases_draft.jsonl"))
    ap.add_argument("--review", type=Path, default=Path("data/plausibility_C_paraphrases_review.csv"))
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.limit < 0 or args.sleep < 0:
        ap.error("--limit와 --sleep은 음수일 수 없습니다")
    cases = selected_cases(args.input)
    subset = cases[:args.limit] if args.limit else cases
    print(f"C 조건={len(cases)}개, 이번 선택={len(subset)}개; model={args.model}; prompt={VERSION}")
    if args.dry_run:
        for row in subset:
            print(f"  {row['challenge_id']} | {row['category']} | {row['premise'][:75]}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.out) if args.out.exists() else []
    done = {r["case_id"]: r for r in existing}
    if len(done) != len(existing):
        raise ValueError("초안 JSONL case_id 중복")
    for case in cases:
        old = done.get(case["challenge_id"])
        if old and (old.get("premise") != case["premise"] or old.get("model") != args.model
                    or old.get("prompt_version") != VERSION):
            raise ValueError(f"기존 결과와 입력/버전 충돌: {case['challenge_id']}")
    with args.out.open("a", encoding="utf-8") as file:
        for i, case in enumerate(subset, 1):
            cid = case["challenge_id"]
            if cid in done:
                print(f"[{i}/{len(subset)}] {cid}: 기존 초안 재사용")
                continue
            rewrite = call_gemini(case["premise"], args.model)
            warnings = check_draft(case["premise"], rewrite)
            row = {"case_id": cid, "group_id": case["group_id"],
                   "category": case["category"], "violation": case["violation"],
                   "premise": case["premise"], "generated_answer": rewrite,
                   "auto_warnings": warnings, "model": args.model, "prompt_version": VERSION}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            done[cid] = row
            print(f"[{i}/{len(subset)}] {cid}: warning={warnings or 'none'} | {rewrite[:80]}")
            if i < len(subset) and args.sleep:
                time.sleep(args.sleep)
    create_review_csv([done[r["challenge_id"]] for r in cases if r["challenge_id"] in done], args.review)
    print(f"초안 {len(done)}/20 생성. 검수 CSV를 확인하고, 모든 수치·속성·불가능성을 유지하는 행에만 approved=Y를 입력하세요.")
    if len(done) < 20:
        print("부분 생성 상태입니다. 같은 명령에서 --limit 없이 실행하면 나머지 초안을 생성하고 검수 CSV에 새 행을 추가합니다.")


if __name__ == "__main__":
    main()
