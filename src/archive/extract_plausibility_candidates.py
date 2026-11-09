# extract_plausibility_candidates_v4.py
import argparse, csv, json, re
from pathlib import Path

CATEGORIES = ["biological_role","relational","temporal_order","unit_scale","numeric_physical"]
DISPLAY_ORDER = ["biological_role","temporal_order","relational","numeric_physical","unit_scale"]
PREFIX = {"biological_role":"BIO","temporal_order":"TMP","relational":"REL","numeric_physical":"NUM","unit_scale":"UNT"}

EXCLUDED_SPECIES = ["돼지","비육돈","자돈","모돈","육성돈","돈사","양돈","닭","육계","산란계","병아리","계사","양계","오리","오리사","염소","산양","말","마사"]
CATTLE_MARKERS = ["한우","젖소","송아지","암소","수소","씨수소","보증씨수소","번식우","임신우","착유우","건유우","육성우","비육우","거세우","거세한우","어미 소","어미소","소가 ","소의 "]

BIO_PATTERNS = [
    re.compile(r"(암소|번식우|임신우).{0,50}(임신|분만|출산|수태|착유)", re.I),
    re.compile(r"(임신|분만|출산|수태|착유).{0,50}(암소|번식우|임신우)", re.I),
    re.compile(r"(수소|씨수소|보증씨수소).{0,50}(정액|정자)", re.I),
    re.compile(r"(정액|정자).{0,50}(수소|씨수소|보증씨수소)", re.I),
    re.compile(r"(암소|번식우).{0,50}(난자|배란|수정란)", re.I),
    re.compile(r"(난자|배란|수정란).{0,50}(암소|번식우)", re.I),
    re.compile(r"(수정란).{0,80}(암소).{0,50}(임신)", re.I),
]
REL_PATTERNS = [
    re.compile(r"(아비|어미|부모|외조부|외조모|조부|조모).{0,70}(자손|송아지|후대|딸|아들)", re.I),
    re.compile(r"(자손|송아지|후대|딸|아들).{0,70}(아비|어미|부모|외조부|외조모|조부|조모)", re.I),
    re.compile(r"[가-힣A-Za-z0-9]+\s*의\s*(아비|어미|외조부|외조모|조부|조모)", re.I),
    re.compile(r"[가-힣A-Za-z0-9]+\s*의\s*(자손|후대|송아지|딸|아들)", re.I),
]
REL_TERMS = ["아비","어미","부모","자손","후대","외조부","외조모","조부","조모","딸","아들"]

TEMP_STRONG_PATTERNS = [
    re.compile(r"(생후)\s*\d+(?:\.\d+)?\s*(일|주|개월|년|일령|개월령|월령)?", re.I),
    re.compile(r"(출생\s*(직후|후|전))", re.I),
    re.compile(r"(분만\s*(직후|후|전|직전))", re.I),
    re.compile(r"(수정\s*(후|전))", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*(일령|개월령|월령)", re.I),
]
TEMP_EVENT_TERMS = ["출생","태어","분만","생후","일령","개월령","월령","초유","이유시기","이유 후","이유 전"]

UNIT_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:[.,]\d+)?(?:\s*~\s*-?\d+(?:[.,]\d+)?)?\s*(kg|g|mg|톤|t|L|ℓ|ml|mL|m²|m2|cm|mm|km|m|%|℃|°C)(?![A-Za-z])", re.I)
ANIMAL_NUMERIC_TERMS = ["체중","생시체중","산유량","유량","섭취량","사료섭취량","급여량","증체량","일당증체량","농후사료소요량","칼슘"," 인","구리","마그네슘","유지방","유단백"]
UNIT_SCALE_TERMS = ["체중","생시체중","산유량","유량","섭취량","급여량","농후사료소요량","칼슘"," 인","구리","마그네슘"]
CROP_SCALE_BLOCKERS = ["kg/ha","kg/㏊","톤/ha","톤/㏊","건물 수량","TDN 수량","가소화양분 수량","암 이삭 수량","파종","월동률"]

def load_jsonl(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                rows.append(json.loads(line))
    return rows

def norm(x): return re.sub(r"\s+"," ",str(x or "")).strip()

def sample_label(row):
    for k in ("sample_reference_label","reference_label","gold_label"):
        if row.get(k) is not None: return str(row[k]).strip()
    return ""

def sent_label(row):
    for k in ("reference_label","label","gold_label"):
        if row.get(k) is not None: return str(row[k]).strip()
    return ""

def sent_text(row):
    for k in ("sent","sentence","text"):
        v=row.get(k)
        if isinstance(v,str) and v.strip(): return norm(v)
    return ""

def sentence_rows(gold):
    for k in ("sentence_labels","sentences","reference_sentences"):
        v=gold.get(k)
        if isinstance(v,list): return v
    return []

def premise_text(gen):
    ch=gen.get("premise_chunks")
    if isinstance(ch,list): return norm(" ".join(norm(x) for x in ch))
    return norm(gen.get("premise_full") or gen.get("premise") or "")

def split_premise(text):
    return [norm(p) for p in re.split(r"(?<=[.!?])\s+|\n+",text) if 18 <= len(norm(p)) <= 260]

def has_excluded_species(text): return any(x in text for x in EXCLUDED_SPECIES)
def has_cattle_context(text): return any(x in text for x in CATTLE_MARKERS)

def quality_ok(text):
    text=norm(text)
    return 18 <= len(text) <= 240 and bool(re.search(r"[가-힣]",text)) and text.count("|") < 2 and not text.endswith("?") and not has_excluded_species(text)

def score_bio(text, origin):
    hits=[p.pattern for p in BIO_PATTERNS if p.search(text)]
    if not hits: return None
    score=12+4*len(hits)+(5 if ("암소" in text and any(x in text for x in ["임신","분만","출산","수태"])) else 0)+(4 if origin=="answer" else 0)
    return score,hits,"성별/생식 역할을 뒤집는다. 예: 암소 임신→수소 임신."

def score_rel(text, origin):
    hits=[p.pattern for p in REL_PATTERNS if p.search(text)]
    terms=sorted({x for x in REL_TERMS if x in text})
    if not hits and len(terms)<2: return None
    score=10+3*len(hits)+len(terms)+(4 if origin=="answer" else 2)
    return score,hits or terms,"부모-자손 방향을 뒤집거나 자기참조 관계를 만든다. 예: A의 아비=B → A의 아비=A."

def score_temp(text, origin):
    hits=[p.pattern for p in TEMP_STRONG_PATTERNS if p.search(text)]
    if not hits or not any(x in text for x in TEMP_EVENT_TERMS): return None
    if re.search(r"\b(19|20)\d{2}년\s*(이후|이전)",text) and not any(x in text for x in ["출생","분만","생후","일령","월령"]): return None
    score=10+3*len(hits)+(5 if any(x in text for x in ["분만 후","분만 전","출생 후","출생 전","출생 직후","생후"]) else 0)+(4 if origin=="answer" else 0)
    return score,hits,"출생·분만 전후관계를 뒤집는다. 예: 분만 후 12시간→분만 12시간 전."

def score_numeric(text, origin):
    units=list(UNIT_RE.finditer(text))
    terms=[x for x in ANIMAL_NUMERIC_TERMS if x in text]
    if not units or not terms: return None
    score=8+2*len(units)+min(5,len(terms))+(5 if any(x in text for x in ["체중","생시체중","섭취량","급여량","농후사료소요량"]) else 0)+(4 if origin=="answer" else 0)
    return score,[m.group(0) for m in units],"값의 부호를 바꾸어 물리적으로 불가능한 양으로 변형. 예: 급여량 2.3kg→-2.3kg."

def score_unit(text, origin):
    if any(x.lower() in text.lower() for x in CROP_SCALE_BLOCKERS): return None
    units=list(UNIT_RE.finditer(text))
    if not units or not any(x in text for x in UNIT_SCALE_TERMS): return None
    scalable=[m for m in units if m.group(1).lower() not in {"%","℃","°c"}]
    if not scalable: return None
    score=10+3*len(scalable)+(5 if any(x in text for x in ["체중","산유량","유량","급여량","칼슘"," 인","구리"]) else 0)+(4 if origin=="answer" else 0)
    return score,[m.group(0) for m in scalable],"단위를 10^3 이상 확대한다. 예: 32g→32kg, 0.5mg→0.5kg."

SCORERS={"biological_role":score_bio,"relational":score_rel,"temporal_order":score_temp,"numeric_physical":score_numeric,"unit_scale":score_unit}

def add_candidates(pool,sid,gen,text,origin,sentence_index,sentence_reference_label):
    if not quality_ok(text): return
    context=norm(gen.get("question","")+" "+gen.get("answer","")+" "+text)
    if not has_cattle_context(context): return
    for category,scorer in SCORERS.items():
        out=scorer(text,origin)
        if out is None: continue
        score,hits,hint=out
        pool.append({
            "sample_id":sid,"domain":gen.get("domain",""),"category":category,"score":score,
            "source_origin":origin,"sentence_index":sentence_index,"source_claim":text,
            "rule_hits":" || ".join(map(str,hits)),"mutation_hint":hint,
            "question":norm(gen.get("question","")),"answer":norm(gen.get("answer","")),
            "premise_excerpt":premise_text(gen)[:650],"sample_reference_label":"Supported",
            "sentence_reference_label":sentence_reference_label,
            "manual_keep":"","manual_corrupted_claim":"","manual_notes":""
        })

def build_candidates(generated,gold_rows):
    gen_map={str(x.get("sample_id")):x for x in generated if x.get("sample_id") is not None}
    pool=[]
    for gold in gold_rows:
        sid=str(gold.get("sample_id"))
        if sid not in gen_map or sample_label(gold)!="Supported": continue
        gen=gen_map[sid]
        for idx,s in enumerate(sentence_rows(gold)):
            if sent_label(s)!="Supported": continue
            add_candidates(pool,sid,gen,sent_text(s),"answer",s.get("sentence_index",idx),"Supported")
        for p_idx,text in enumerate(split_premise(premise_text(gen))):
            if not quality_ok(text) or not has_cattle_context(text): continue
            temp=[]
            add_candidates(temp,sid,gen,text,"premise",f"P{p_idx}","SOURCE")
            for row in temp:
                if row["category"] in {"biological_role","relational","temporal_order"}:
                    pool.append(row)
    return pool

def deduplicate(pool):
    best={}
    for row in pool:
        key=(row["category"],norm(row["source_claim"]))
        if key not in best or row["score"]>best[key]["score"]:
            best[key]=row
    return list(best.values())

def select_top(pool,per_category=4):
    grouped={cat:sorted([r for r in pool if r["category"]==cat],key=lambda r:(-r["score"],0 if r["source_origin"]=="answer" else 1,r["sample_id"])) for cat in CATEGORIES}
    selected=[]; used_claims=set(); used_samples=set()
    for cat in CATEGORIES:
        chosen=0
        for row in grouped[cat]:
            claim=norm(row["source_claim"])
            if claim in used_claims or row["sample_id"] in used_samples: continue
            selected.append(dict(row)); used_claims.add(claim); used_samples.add(row["sample_id"]); chosen+=1
            if chosen>=per_category: break
        if chosen<per_category:
            for row in grouped[cat]:
                claim=norm(row["source_claim"])
                if claim in used_claims: continue
                selected.append(dict(row)); used_claims.add(claim); chosen+=1
                if chosen>=per_category: break
    rank={c:i for i,c in enumerate(DISPLAY_ORDER)}
    selected.sort(key=lambda r:(rank[r["category"]],-r["score"]))
    counters={c:0 for c in DISPLAY_ORDER}
    for row in selected:
        c=row["category"]; counters[c]+=1; row["candidate_id"]=f"{PREFIX[c]}_{counters[c]:03d}"
    return selected

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cols=["candidate_id","category","score","sample_id","domain","source_origin","sentence_index","source_claim","rule_hits","mutation_hint","question","answer","premise_excerpt","sample_reference_label","sentence_reference_label","manual_keep","manual_corrupted_claim","manual_notes"]
    cols=[c for c in cols if any(c in r for r in rows)]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--generated",type=Path,default=Path("data/generated_qna.jsonl"))
    ap.add_argument("--gold",type=Path,default=Path("data/reference_all_236_labeled.jsonl"))
    ap.add_argument("--all-out",type=Path,default=Path("results/plausibility_candidates_v4_all.csv"))
    ap.add_argument("--top-out",type=Path,default=Path("results/plausibility_candidates_v4_top20.csv"))
    ap.add_argument("--per-category",type=int,default=4)
    args=ap.parse_args()

    generated=load_jsonl(args.generated); gold=load_jsonl(args.gold)
    pool=deduplicate(build_candidates(generated,gold))
    rank={c:i for i,c in enumerate(DISPLAY_ORDER)}
    pool.sort(key=lambda r:(rank[r["category"]],-r["score"],0 if r["source_origin"]=="answer" else 1,r["sample_id"]))
    selected=select_top(pool,args.per_category)
    write_csv(args.all_out,pool); write_csv(args.top_out,selected)

    print("="*88)
    print("Plausibility candidate extractor v4")
    print("="*88)
    print(f"Generated samples : {len(generated)}")
    print(f"Gold samples      : {len(gold)}")
    print(f"Candidate rows    : {len(pool)}")
    print(f"Selected rows     : {len(selected)}")
    print()
    for cat in DISPLAY_ORDER:
        rows=[r for r in pool if r["category"]==cat]
        top=[r for r in selected if r["category"]==cat]
        ans=sum(r["source_origin"]=="answer" for r in rows)
        prem=sum(r["source_origin"]=="premise" for r in rows)
        print(f"{cat:18s} all={len(rows):3d} (answer={ans:3d}, premise={prem:3d}) selected={len(top):2d}")
    print()
    print(f"[ALL] {args.all_out.resolve()}")
    print(f"[TOP] {args.top_out.resolve()}")

if __name__=="__main__":
    main()
