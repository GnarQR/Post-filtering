#!/usr/bin/env python3
# coding: utf-8
"""Paired accuracy by difficulty for E8-Hard using benchmark's comparison.csv."""
import argparse,csv,json,sys
from collections import defaultdict,Counter
from pathlib import Path


def report(comparison:list[dict],dataset:list[dict],split:str)->dict:
    byid={r['case_id']:r for r in dataset if r['split']==split}
    groups=defaultdict(list)
    for r in comparison:
        cid=r['case_id']
        if cid not in byid:continue
        g=byid[cid]
        groups[g['scenario_id']].append((g,r))
    if not groups:raise ValueError('일치하는 사례 없음')
    for sid,items in groups.items():
        if len(items)!=2 or {g['gold'] for g,_ in items}!={'PASS','VIOLATION'}:
            raise ValueError(f'{sid}: 정확히 PASS/VIOLATION 한 쌍이 필요합니다')
    stats={}
    for model in ('jev','lite','flash'):
        bydiff=defaultdict(lambda:{'pairs':0,'both_correct':0,'violations_found':0,'false_blocks':0,'reviews':0})
        for sid,items in groups.items():
            level=items[0][0]['difficulty']
            rec=bydiff[level]
            rec['pairs']+=1
            preds=[r.get('pred_'+model, '') for _,r in items]
            if not all(preds):raise ValueError(f'{sid}: {model} 예측 누락')
            if all(g['gold']==p for (g,_),p in zip(items,preds)):
                rec['both_correct']+=1
            rec['violations_found']+=sum(g['gold']=='VIOLATION' and p=='VIOLATION' for (g,_),p in zip(items,preds))
            rec['false_blocks']+=sum(g['gold']=='PASS' and p=='VIOLATION' for (g,_),p in zip(items,preds))
            rec['reviews']+=sum(p=='REVIEW' for p in preds)
        allstats={'pairs':sum(r['pairs'] for r in bydiff.values()),
                  'both_correct':sum(r['both_correct'] for r in bydiff.values()),
                  'violations_found':sum(r['violations_found'] for r in bydiff.values()),
                  'false_blocks':sum(r['false_blocks'] for r in bydiff.values()),
                  'reviews':sum(r['reviews'] for r in bydiff.values())}
        def fmt(x):return {**x,'pair_accuracy':round(x['both_correct']/x['pairs'],4) if x['pairs'] else None}
        stats[model]={'all':fmt(allstats),'by_difficulty':{k:fmt(v) for k,v in sorted(bydiff.items())}}
    return {'split':split,'num_pairs':len(groups),'model_stats':stats}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/e8_hard_60_human_final.jsonl'))
    p.add_argument('--split',choices=('dev','test'),default='test')
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args(argv)
    with a.comparison.open(encoding='utf-8-sig',newline='') as f: comp=list(csv.DictReader(f))
    data=[json.loads(x) for x in a.dataset.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    r=report(comp,data,a.split)
    print(f"[E8-Hard {a.split}] paired cases={r['num_pairs']}")
    for model,st in r['model_stats'].items():
        print(f"  {model:5} both_correct={st['all']['both_correct']}/{st['all']['pairs']} pair_acc={st['all']['pair_accuracy']:.3f} missed_violations={st['all']['pairs']-st['all']['violations_found']} false_blocks={st['all']['false_blocks']}")
        for d,m in st['by_difficulty'].items(): print(f"    {d:15} {m['both_correct']}/{m['pairs']}")
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as e:print('[ERROR]',e,file=sys.stderr);sys.exit(2)
