#!/usr/bin/env python3
"""Convert human-reviewed E8 CSV to a *reviewed* JSONL for downstream evaluation.

No auto gold labels: only rows explicitly checked YES and given human_gold are exported.
Entire 3-variant scenario groups must be reviewed to avoid partial matching pairs.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"PASS", "REVIEW", "VIOLATION"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="E8 사람 검수 라벨 확정 및 QA JSONL 내보내기")
    p.add_argument("--draft", type=Path, default=Path("data/e8_safety_108_draft.jsonl"))
    p.add_argument("--review-sheet", type=Path, default=Path("results/e8_human_review_sheet.csv"))
    p.add_argument("--out", type=Path, default=Path("data/e8_safety_verified.jsonl"))
    p.add_argument("--require-all", action="store_true", help="108개 전부 검수되지 않으면 종료")
    args = p.parse_args(argv)
    if not args.draft.exists() or not args.review_sheet.exists():
        p.error("생성 데이터 또는 검수 시트가 없습니다. 생성기 실행 및 사람 검수를 먼저 완료하세요.")
    with args.draft.open(encoding="utf-8-sig") as f:
        data = [json.loads(line) for line in f if line.strip()]
    with args.review_sheet.open(encoding="utf-8-sig", newline="") as f:
        reviews = list(csv.DictReader(f))
    index = {}
    for row in reviews:
        cid = row.get("case_id", "").strip()
        if cid in index:
            p.error(f"검수 시트 case_id 중복: {cid}")
        index[cid] = row
    unknown = set(index) - {r["case_id"] for r in data}
    if unknown:
        p.error(f"검수 시트에 존재하지 않는 case_id: {sorted(unknown)[:3]}")
    groups = defaultdict(list)
    for r in data:
        groups[r["scenario_id"]].append(r)
    approved_groups = []
    pending = 0
    for group, rows in groups.items():
        approved = []
        for item in rows:
            user = index.get(item["case_id"], {})
            label = (user.get("human_gold") or "").strip().upper()
            verified = (user.get("fact_verified") or "").strip().upper()
            source = (user.get("source_url") or "").strip()
            reviewer = (user.get("reviewer") or "").strip()
            # Require an explicit human verification flag and reviewer identity.
            if label in VALID and verified == "YES" and source and reviewer:
                approved.append({**item, "gold": label,
                                 "gold_status": "HUMAN_REVIEWED_WITH_REFERENCES",
                                 "source_url": source, "reviewer": reviewer,
                                 "human_notes": (user.get("review_notes") or "").strip()})
        if len(approved) == len(rows):
            approved_groups.extend(approved)
        else:
            pending += len(rows)
    if args.require_all and (len(approved_groups) != 108 or len(data) != 108):
        p.error(f"전체 108건 검수가 필요합니다. 현재 승인 {len(approved_groups)}건 / 입력 {len(data)}건")
    if not approved_groups:
        p.error("승인된 시나리오가 없습니다. human_gold=PASS/REVIEW/VIOLATION, fact_verified=YES, source_url, reviewer를 채우세요.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in approved_groups:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"저장: {args.out} | 검수 완료 {len(approved_groups)}건 | 보류 {pending}건")
    print(f"분할: {dict(Counter(x['split'] for x in approved_groups))}")
    print(f"라벨: {dict(Counter(x['gold'] for x in approved_groups))}")
    print("주의: HUMAN_REVIEWED는 사람이 라벨/근거를 검토했다는 기록이지 전문가 검증을 자동 보증하지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
