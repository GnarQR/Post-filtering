#!/usr/bin/env python3
# coding: utf-8
"""Finalize 60 E8-Hard draft cases after binary human labeling. No API calls.
All 60 rows must be reviewed. Exclusions must remove whole pairs to preserve pairing.
"""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
from collections import Counter,defaultdict


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def write_atomic(path:Path,text:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+".tmp")
    tmp.write_text(text,encoding="utf-8")
    os.replace(tmp,path)


def clean(v):return (v or "").strip()


def finalize(draft:list[dict],review:list[dict])->tuple[list[dict],dict]:
    src={r['case_id']:r for r in draft}
    rev={r['case_id']:r for r in review}
    if len(src)!=len(draft) or len(rev)!=len(review) or set(src)!=set(rev):
        raise ValueError("case_id 누락·중복 또는 원본과 검수표 항목 불일치")
    if len(draft)!=60:raise ValueError(f"원본이 60 QA가 아닙니다: {len(draft)}")
    pairs=defaultdict(list)
    decisions={}
    for cid,s in src.items():
        r=rev[cid]
        if clean(r.get('question'))!=s['question'] or clean(r.get('answer'))!=s['answer']:
            raise ValueError(f"{cid}: 검수표의 원본 질문·답변이 바뀌었습니다. 수정질문/수정답변 열에만 기입하세요")
        decision=clean(r.get('검수상태'))
        if decision not in ('승인','수정','제외'):
            raise ValueError(f"{cid}: 검수상태가 미기입/오입력입니다: {decision!r}")
        label=clean(r.get('최종Gold')).upper() or s['gold_draft']
        if decision!='제외' and label not in ('PASS','VIOLATION'):
            raise ValueError(f"{cid}: Gold는 PASS 또는 VIOLATION만 허용")
        q=clean(r.get('수정질문')) or s['question']
        a=clean(r.get('수정답변')) or s['answer']
        if decision=='수정' and not (clean(r.get('수정질문')) or clean(r.get('수정답변')) or clean(r.get('최종Gold'))):
            raise ValueError(f"{cid}: 수정 판정인데 실제 수정이 없습니다")
        if decision=='승인' and (clean(r.get('수정질문')) or clean(r.get('수정답변'))):
            raise ValueError(f"{cid}: 승인인데 수정문이 있습니다. 검수상태=수정으로 바꿔주세요")
        decisions[cid]={'decision':decision,'gold':label,'question':q,'answer':a,'review':r}
        pairs[s['scenario_id']].append(cid)
    if len(pairs)!=30 or any(len(v)!=2 for v in pairs.values()):
        raise ValueError("최종 데이터 30개 쌍 구조가 무너졌습니다")
    output=[]
    dropped=[]
    for pair_id,cids in pairs.items():
        items=[decisions[cid] for cid in cids]
        excluded=[item['decision']=='제외' for item in items]
        if any(excluded):
            if not all(excluded):raise ValueError(f"{pair_id}: 한쪽만 제외하면 안 됩니다. 같은 시나리오의 두 행 모두 제외하세요")
            dropped.extend(cids)
            continue
        if items[0]['question']!=items[1]['question']:
            raise ValueError(f"{pair_id}: 대조쌍의 최종 질문이 서로 다릅니다. 수정질문을 양쪽에 동일하게 입력하세요")
        if {item['gold'] for item in items}!={'PASS','VIOLATION'}:
            raise ValueError(f"{pair_id}: 대조쌍은 PASS/VIOLATION 한 개씩 있어야 합니다")
        if items[0]['answer']==items[1]['answer']:
            raise ValueError(f"{pair_id}: 최종 답변이 같습니다")
        for cid in cids:
            src_case=src[cid];v=decisions[cid];rr=v['review']
            output.append({**src_case,'question_original':src_case['question'],
                           'answer_original':src_case['answer'],'gold_original':src_case['gold_draft'],
                           'question':v['question'],'answer':v['answer'],'gold':v['gold'],
                           'review_decision':v['decision'], 'human_answer_edited':v['answer']!=src_case['answer'],
                           'human_gold_changed':v['gold']!=src_case['gold_draft'],
                           'reviewer_note':clean(rr.get('검수메모')),
                           'source_url_reviewer':clean(rr.get('근거URL')),
                           'gold_status':'HUMAN_REVIEWED_FIXED'})
    summary={'accepted_cases':len(output),'excluded_cases':len(dropped),
             'included_pairs':len(output)//2,'excluded_ids':dropped,
             'splits':dict(Counter(r['split'] for r in output)),
             'gold':dict(Counter(r['gold'] for r in output)),
             'difficulty':dict(Counter(r['difficulty'] for r in output)),
             'human_changes':sum(r['human_answer_edited'] or r['human_gold_changed'] for r in output)}
    return output,summary


def main(argv=None)->int:
    p=argparse.ArgumentParser(description='E8-Hard human-reviewed binary Gold finalizer')
    p.add_argument('--draft',type=Path,default=Path('data/e8_hard_60_draft.jsonl'))
    p.add_argument('--review',type=Path,default=Path('results/e8_hard_human_review.csv'))
    p.add_argument('--out',type=Path,default=Path('data/e8_hard_60_human_final.jsonl'))
    p.add_argument('--summary',type=Path,default=Path('results/e8_hard_final_summary.json'))
    args=p.parse_args(argv)
    raw=read_jsonl(args.draft)
    with args.review.open('r',encoding='utf-8-sig',newline='') as f:
        reviews=list(csv.DictReader(f))
    out,summary=finalize(raw,reviews)
    if not out:raise ValueError('전체가 제외됐으므로 평가셋을 만들 수 없습니다')
    write_atomic(args.out,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    write_atomic(args.summary,json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(f"최종 데이터 저장: {args.out} / {len(out)} QA / {summary['included_pairs']}쌍")
    print(f"분할={summary['splits']}, Gold={summary['gold']}, 제외={summary['excluded_cases']}")
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
        print('[ERROR]',e,file=sys.stderr);sys.exit(2)
