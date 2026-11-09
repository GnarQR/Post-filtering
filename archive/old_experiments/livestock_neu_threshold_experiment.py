"""
livestock_neu_threshold_experiment.py

40문항 축우 QnA 데이터셋(1차년도_QnA_데이터셋_평가_수정.xlsx에서 추출한
qna_ground_truth.json)을 이용해:

  1. 각 질문에 대해 Gemini 2.5 flash-lite로 "새로운" 답변을 생성
     (CAUcowchat의 기존 답변은 거절/컷이 많이 섞여있어 순수하게
      "GT-답변 표현 거리로 인한 NEU" 신호를 볼 수 없으므로 배제)
  2. GT(premise) vs 새 답변(hypothesis)을 NLI/judge로 비교하되,
     단순 argmax 라벨이 아니라 확률/확신도까지 출력
  3. 여러 threshold에서 "NEU로 판정되어 반려되는 비율"을 집계

중요 - backend별 확신도의 성격이 다름:
  - NLI(klue-roberta 등): softmax 출력이 진짜 확률. threshold 실험에 그대로 사용 가능.
  - Gemini(LLM-judge): 생성형이라 진짜 확률이 없음. 프롬프트에서 "라벨 + 자기 확신도(0~1)"를
    같이 요청하는 방식 -> 모델의 자기 보고치이며 엄밀한 확률이 아님. 해석 시 반드시 이 차이를
    명시할 것.

사용법:
  python livestock_neu_threshold_experiment.py --backend nli --nli-model Huffon/klue-roberta-base-nli
  python livestock_neu_threshold_experiment.py --backend gemini
"""

import os
import requests
import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # .env에서 JEV_API_KEY=... 읽어오기

import re

QNA_PATH = Path("qna_ground_truth.json")

JEV_API_KEY = os.environ["JEV_API_KEY"]
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def split_into_sentences(text: str) -> list:
    """
    간단한 한국어 문장 분리기. 완벽한 구문분석기는 아니지만,
    줄바꿈 + 종결어미(다./요./니다. 등) + 마침표를 기준으로 분리.
    번호 매긴 리스트("1. ...", "① ...")도 최대한 각 항목을 한 문장으로 취급.
    """
    # 먼저 줄바꿈으로 큼직하게 분리
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    sentences = []
    for chunk in chunks:
        # 마침표/물음표/느낌표 뒤에서 분리 (숫자 뒤 소수점 오분리 방지를 위해
        # 마침표 뒤에 공백이나 문장 끝이 오는 경우만 분리 기준으로 삼음)
        parts = re.split(r'(?<=[.!?])\s+', chunk)
        for p in parts:
            p = p.strip()
            # "1. ", "① " 같은 리스트 번호만 남은 조각은 건너뜀
            if p and not re.fullmatch(r'[0-9①-⑳.\)\s]+', p):
                sentences.append(p)
    return sentences if sentences else [text.strip()]


def match_sentence_against_checklist(gt_sentences: list, answer_sentence: str, nli_model: str) -> dict:
    """
    답변 문장 하나를 GT 체크리스트 문장들과 대조하되, "아무 문장이나 CON이면 전체
    CON"이 아니라 -- ENT 확률이 가장 높은 "최적 매칭 GT 문장"을 먼저 찾고,
    그 매칭 하나에 대한 라벨만 최종으로 사용.

    이유: 답변 문장을 체크리스트 전체와 무차별 대조하면, 주제가 전혀 다른
    GT 문장과의 비교(예: "환기 점검" 답변 vs "콧물 확인" GT)에서 나오는
    스퓨리어스(가짜) CON이 체크리스트 길이에 비례해 누적되어, "하나라도
    CON이면 전체 CON" 규칙과 결합했을 때 오탐이 기하급수적으로 늘어남.
    가장 관련 있는 근거 문장 하나만 최종 판정에 쓰면 이 문제를 원천 차단.
    """
    per_gt = []
    for gt_sent in gt_sentences:
        r = call_nli_with_probs(gt_sent, answer_sentence, nli_model)
        per_gt.append({"gt_sentence": gt_sent, "label": r["label"], "probs": r["probs"]})

    # ENT 확률이 가장 높은 GT 문장을 "최적 매칭 근거"로 선정
    best_match = max(per_gt, key=lambda p: p["probs"].get("ENT", 0))
    final = best_match["label"]
    return {"answer_sentence": answer_sentence, "final_label": final,
            "matched_gt_sentence": best_match["gt_sentence"], "per_gt_checks": per_gt}


def run_atomic_to_atomic(nli_model: str, answer_mode: str = "grounded"):
    """
    premise(GT)도 문장 단위로 쪼개서 체크리스트로 만들고, hypothesis(답변)도
    문장 단위로 쪼개서, 답변 문장 하나하나를 체크리스트 전체와 대조.
    문항 단위 집계는 run_atomic_hypothesis와 동일한 규칙(CON>NEU>ENT 우선순위).
    """
    qna = json.loads(QNA_PATH.read_text(encoding="utf-8"))
    cache_path = Path(f"qna_llm_answer_cache_{answer_mode}.json")
    if not cache_path.exists():
        raise RuntimeError(f"{cache_path} 없음 -> 먼저 답변 생성부터 해야 함")
    answers = json.loads(cache_path.read_text(encoding="utf-8"))

    model_tag = nli_model.replace("/", "_")
    results_path = Path(f"qna_atomic_to_atomic_results_{answer_mode}_{model_tag}.jsonl")
    per_question = []
    with results_path.open("a", encoding="utf-8") as f:
        for item in qna:
            num = str(item["번호"])
            gt_sentences = split_into_sentences(item["정답"])
            answer_sentences = split_into_sentences(answers[num])

            sentence_matches = []
            for a_sent in answer_sentences:
                m = match_sentence_against_checklist(gt_sentences, a_sent, nli_model)
                sentence_matches.append(m)
                print(f"  [{num}] '{a_sent[:30]}...' -> {m['final_label']}")

            labels = [m["final_label"] for m in sentence_matches]
            if "CON" in labels:
                agg = "CON"
            elif "NEU" in labels:
                agg = "NEU"
            else:
                agg = "ENT"

            row = {"번호": num, "aggregate_label": agg, "n_gt_sentences": len(gt_sentences),
                   "n_answer_sentences": len(answer_sentences), "sentence_matches": sentence_matches,
                   "nli_model": nli_model}
            print(f"[{num}] GT {len(gt_sentences)}문장 vs 답변 {len(answer_sentences)}문장 -> 집계: {agg}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            per_question.append(row)

    n = len(per_question)
    n_con = sum(1 for r in per_question if r["aggregate_label"] == "CON")
    n_neu = sum(1 for r in per_question if r["aggregate_label"] == "NEU")
    n_ent = sum(1 for r in per_question if r["aggregate_label"] == "ENT")
    print(f"\n=== 문항 단위 집계 (n={n}, model={nli_model}) ===")
    print(f"  CON(환각의심): {n_con} ({n_con/n*100:.1f}%)")
    print(f"  NEU(부분미검증): {n_neu} ({n_neu/n*100:.1f}%)")
    print(f"  ENT(정상): {n_ent} ({n_ent/n*100:.1f}%)")
    print(f"\n결과 저장: {results_path.resolve()}")
    return per_question


def run_atomic_hypothesis(nli_model: str, answer_mode: str = "grounded", backend: str = "nli"):
    """
    premise = GT 전체(원문 그대로), hypothesis = LLM 답변을 문장 단위로 쪼갠 것 각각.
    backend="nli"면 로컬 NLI 모델, backend="gemini"면 실제 팀 파이프라인 방식인
    LLM-as-judge(Gemini)로 판정 -- NLI 실험 결과가 실제 운영 방식에서도
    재현되는지 교차검증하는 용도.
    문항당 여러 문장-레벨 판정이 나오므로, 문항 단위 집계 규칙:
      - 문장 중 하나라도 CON -> 문항 전체 CON(환각 의심)으로 플래그
      - CON 없고 NEU가 하나라도 있으면 -> 문항 전체 NEU(부분 미검증)
      - 전부 ENT -> 문항 전체 ENT
    """
    qna = json.loads(QNA_PATH.read_text(encoding="utf-8"))
    cache_path = Path(f"qna_llm_answer_cache_{answer_mode}.json")
    if not cache_path.exists():
        raise RuntimeError(f"{cache_path} 없음 -> 먼저 --mode run(또는 기존 run)으로 답변 생성부터 해야 함")
    answers = json.loads(cache_path.read_text(encoding="utf-8"))

    model_tag = "gemini-2.5-flash-lite" if backend == "gemini" else nli_model.replace("/", "_")
    results_path = Path(f"qna_atomic_hypothesis_results_{answer_mode}_{model_tag}.jsonl")
    per_question = []
    with results_path.open("a", encoding="utf-8") as f:
        for item in qna:
            num = str(item["번호"])
            gt = item["정답"]
            answer = answers[num]
            sentences = split_into_sentences(answer)

            sentence_results = []
            for sent in sentences:
                if backend == "gemini":
                    r = call_gemini_judge_with_confidence(gt, sent)
                    label = r["label"]
                    extra = {"self_reported_confidence": r["confidence"]}
                    time.sleep(1.0)
                else:
                    r = call_nli_with_probs(gt, sent, nli_model)
                    label = r["label"]
                    extra = {"probs": r["probs"]}
                sentence_results.append({"sentence": sent, "label": label, **extra})
                print(f"  [{num}] '{sent[:30]}...' -> {label}")

            labels = [s["label"] for s in sentence_results]
            if "CON" in labels:
                agg = "CON"
            elif "NEU" in labels:
                agg = "NEU"
            else:
                agg = "ENT"

            row = {"번호": num, "aggregate_label": agg, "n_sentences": len(sentences),
                   "sentence_results": sentence_results, "gt": gt, "answer": answer, "backend": backend}
            print(f"[{num}] 문장 {len(sentences)}개 -> 집계: {agg}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            per_question.append(row)

    n = len(per_question)
    n_con = sum(1 for r in per_question if r["aggregate_label"] == "CON")
    n_neu = sum(1 for r in per_question if r["aggregate_label"] == "NEU")
    n_ent = sum(1 for r in per_question if r["aggregate_label"] == "ENT")
    print(f"\n=== 문항 단위 집계 (n={n}, backend={backend}, model={model_tag}) ===")
    print(f"  CON(환각의심): {n_con} ({n_con/n*100:.1f}%)")
    print(f"  NEU(부분미검증): {n_neu} ({n_neu/n*100:.1f}%)")
    print(f"  ENT(정상): {n_ent} ({n_ent/n*100:.1f}%)")
    print(f"\n결과 저장: {results_path.resolve()}")
    return per_question


def build_answer_prompt(question: str) -> str:
    return f"""다음은 축우(젖소/한우) 사양관리 관련 질문입니다. 당신이 아는 축산학
지식을 바탕으로, 자연스러운 한국어로 답변하세요. 서론이나 부연 설명 없이
답변 내용만 작성하세요.

질문: {question}"""


def build_grounded_answer_prompt(question: str, gt: str) -> str:
    # 이성훈 박사님 파이프라인의 프롬프트 제어 방식 재현: 주어진 근거(GT/트리플)
    # 내용 위주로만 답하고, 임의로 정보를 추가/각색하지 말 것을 명시적으로 지시.
    # 이게 사실상 RAG의 grounding 방식과 동일함.
    return f"""다음은 축우(젖소/한우) 사양관리 관련 질문과, 이를 답변할 때 근거로
삼아야 할 참고 자료입니다. 참고 자료에 있는 내용 위주로만 답변하고,
참고 자료에 없는 내용을 임의로 추가하거나 각색하지 마세요. 서론이나 부연
설명 없이 답변 내용만 작성하세요.

질문: {question}

참고 자료: {gt}"""


def build_grounded_answer_prompt_negation_free(question: str, gt: str) -> str:
    """
    답변 생성 단계(LLM Sentences 생성 단계)에서 미리 포맷을 고정하는 사전처리
    실험용 프롬프트. NLI/LLM-judge 판정 단계에서 부정어(안/않/없) 반전이
    반복적으로 문제를 일으켰으므로, 애초에 부정문을 못 쓰게 강제하고
    참고 자료의 부정 표현도 긍정 서술로 바꿔 쓰도록 명시적으로 지시.
    """
    return f"""다음은 축우(젖소/한우) 사양관리 관련 질문과, 이를 답변할 때 근거로
삼아야 할 참고 자료입니다. 참고 자료에 있는 내용 위주로만 답변하고,
참고 자료에 없는 내용을 임의로 추가하거나 각색하지 마세요. 서론이나 부연
설명 없이 답변 내용만 작성하세요.

**중요한 형식 제약**: 부정문("~하지 않는지", "~이 없는지", "~이 아닌지" 등)을
사용하지 마세요. 참고 자료가 부정형으로 서술되어 있더라도, 의미가 바뀌지
않는 선에서 반드시 긍정 서술형으로 바꿔서 답변하세요.
예: "냄새가 나지 않는지 확인한다" -> "냄새가 없는 상태인지 확인한다" (X, 여전히 부정)
    "냄새가 나지 않는지 확인한다" -> "정상적인 냄새 상태인지 확인한다" (O, 긍정 서술)

질문: {question}

참고 자료: {gt}"""


def build_judge_prompt_with_confidence(premise: str, hypothesis: str) -> str:
    return f"""당신은 자연어 추론(NLI) 판정자입니다. 아래 전제(premise)를 기준으로
가설(hypothesis)이 함의(ENT), 모순(CON), 중립(NEU) 중 무엇인지 판정하고,
그 판정에 대한 스스로의 확신도(0.0~1.0)를 함께 제시하세요.

반드시 아래 형식으로만 출력하세요 (다른 텍스트 없이):
라벨: <ENT|CON|NEU>
확신도: <0.0~1.0 사이 숫자>

전제: {premise}
가설: {hypothesis}"""


def _gemini_model(temperature: float = None):
    import google.generativeai as genai  # type: ignore
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없음")
    genai.configure(api_key=api_key)
    if temperature is not None:
        gen_config = genai.types.GenerationConfig(temperature=temperature)
        return genai.GenerativeModel("gemini-2.5-flash-lite", generation_config=gen_config)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def generate_llm_answer(question: str, gt: str = None, grounded: bool = False, negation_free: bool = False) -> str:
    model = _gemini_model()
    if grounded:
        prompt = (build_grounded_answer_prompt_negation_free(question, gt) if negation_free
                   else build_grounded_answer_prompt(question, gt))
    else:
        prompt = build_answer_prompt(question)
    resp = model.generate_content(prompt)
    return resp.text.strip()


def call_gemini_judge_with_confidence(premise: str, hypothesis: str, temperature: float = 0.0) -> dict:
    """
    반환: {"label": "ENT"|"CON"|"NEU"|"UNPARSED", "confidence": float|None}
    confidence는 모델의 자기 보고치 -> 진짜 확률 아님, 참고용으로만 사용.

    기본값 temperature=0.0: self-consistency 실험 결과, 기본 temperature에서
    나타났던 라벨 흔들림(최대 20%)이 temperature=0에서는 완전히 사라짐(10/10 일관)
    -> 재현성 문제 해결을 위해 실무 파이프라인의 기본값으로 채택.
    단, temperature=0은 "확률 1위 답으로 고정"하는 것일 뿐 정확도를 보장하지
    않음 -> 원래도 틀리던 판정이면 이제 "일관되게" 틀리게 됨. 정확도 문제는
    별개 축(모델 선택, grounding, atomic 분해 등)에서 계속 다뤄야 함.
    """
    model = _gemini_model(temperature=temperature)
    resp = model.generate_content(build_judge_prompt_with_confidence(premise, hypothesis))
    text = resp.text.strip()

    label = "UNPARSED"
    for cand in ("ENT", "CON", "NEU"):
        if cand in text.upper():
            label = cand
            break

    confidence = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("확신도"):
            try:
                confidence = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return {"label": label, "confidence": confidence, "raw": text}


_NLI_MODEL_CACHE = {}


def _get_nli_model(model_name: str, device: str = "cpu"):
    cache_key = (model_name, device)
    if cache_key not in _NLI_MODEL_CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError(f"device={device} 요청했지만 torch.cuda.is_available()=False")
            model = model.to(device)
        print(f"[nli] id2label = {model.config.id2label}  (device={device})")
        _NLI_MODEL_CACHE[cache_key] = (tokenizer, model, torch)
    return _NLI_MODEL_CACHE[cache_key]


_LABEL_MAP = {
    "entailment": "ENT", "contradiction": "CON", "neutral": "NEU",
}


def call_nli_with_probs(premise: str, hypothesis: str, model_name: str, device: str = "cpu") -> dict:
    """
    반환: {"label": "ENT"|"CON"|"NEU", "probs": {"ENT": x, "CON": y, "NEU": z}}
    probs는 softmax를 통과한 실제 확률.
    device="cuda"(또는 "cuda:0" 등)로 GPU 사용 가능 -> CPU 대비 수십 배 빠름
    (특히 DeBERTa-v3-large처럼 큰 모델 + 긴 문서를 다룰 때 체감 크게 다름).
    """
    tokenizer, model, torch = _get_nli_model(model_name, device=device)
    # RoBERTa 계열은 position id를 padding_idx+1부터 세기 시작해서, 토큰을
    # max_position_embeddings 그대로 채우면 위치 인덱스가 테이블 크기를
    # 넘어서 터짐(index out of bounds) -> 여유분 2칸을 미리 빼고 자름
    safe_max_len = getattr(model.config, "max_position_embeddings", 512) - 2
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=safe_max_len)
    inputs.pop("token_type_ids", None)  # RoBERTa 계열 type_vocab_size=1 이슈 회피
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs_tensor = torch.softmax(logits, dim=-1)[0]

    probs = {}
    for idx, p in enumerate(probs_tensor.tolist()):
        raw_label = model.config.id2label[idx].strip().lower()
        mapped = _LABEL_MAP.get(raw_label, raw_label)
        probs[mapped] = p

    best_label = max(probs, key=probs.get)
    return {"label": best_label, "probs": probs}


def get_or_generate_answers(qna: list, answer_mode: str) -> dict:
    cache_path = Path(f"qna_llm_answer_cache_{answer_mode}.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    answers = {}
    for i, item in enumerate(qna):
        num = str(item["번호"])
        print(f"[{i+1}/{len(qna)}] 답변 생성 중... (질문 {num}, mode={answer_mode})")
        if answer_mode == "grounded":
            answers[num] = generate_llm_answer(item["질문"], gt=item["정답"], grounded=True)
        else:
            answers[num] = generate_llm_answer(item["질문"])
        time.sleep(1.0)
        cache_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return answers


def run(backend: str, nli_model: str, answer_mode: str):
    qna = json.loads(QNA_PATH.read_text(encoding="utf-8"))
    answers = get_or_generate_answers(qna, answer_mode)

    results_path = Path(f"qna_neu_threshold_results_{answer_mode}.jsonl")
    results = []
    with results_path.open("a", encoding="utf-8") as f:
        for item in qna:
            num = str(item["번호"])
            premise = item["정답"]         # GT
            hypothesis = answers[num]        # LLM 답변 (independent 또는 grounded)

            try:
                if backend == "nli":
                    r = call_nli_with_probs(premise, hypothesis, nli_model)
                    row = {
                        "번호": num, "backend": "nli", "label": r["label"],
                        "probs": r["probs"],
                    }
                    print(f"[{num}] {r['label']}  probs={ {k: round(v,3) for k,v in r['probs'].items()} }")
                elif backend == "gemini":
                    r = call_gemini_judge_with_confidence(premise, hypothesis)
                    row = {
                        "번호": num, "backend": "gemini", "label": r["label"],
                        "self_reported_confidence": r["confidence"],
                    }
                    print(f"[{num}] {r['label']}  self_conf={r['confidence']}")
                    time.sleep(1.0)
                else:
                    raise ValueError(f"알 수 없는 backend: {backend}")
            except Exception as e:
                row = {"번호": num, "backend": backend, "label": f"ERROR:{e}"}
                print(f"[{num}] ERROR: {e}")

            row["answer_mode"] = answer_mode
            row["premise"] = premise
            row["hypothesis"] = hypothesis
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            results.append(row)

    print(f"\n[answer_mode={answer_mode}] 결과 저장: {results_path.resolve()}")
    print_threshold_summary(results, backend)
    return results


def print_threshold_summary(results: list, backend: str):
    print("\n=== NEU 반려 비율 (threshold별) ===")
    valid = [r for r in results if not str(r.get("label", "")).startswith("ERROR")]
    n_error = len(results) - len(valid)
    n = len(valid)
    if n_error:
        print(f"  (주의: 에러 {n_error}건은 분모에서 제외, 유효 판정 {n}건 기준으로 계산)")

    if backend == "nli":
        thresholds = [0.5, 0.7, 0.9]
        for th in thresholds:
            rejected = sum(1 for r in valid if r.get("probs", {}).get("NEU", 0) >= th)
            print(f"  NEU 확률 >= {th}: {rejected}/{n} 반려 ({rejected/n*100:.1f}%)")
        argmax_neu = sum(1 for r in valid if r.get("label") == "NEU")
        print(f"  (참고) argmax 기준 NEU 비율: {argmax_neu}/{n} ({argmax_neu/n*100:.1f}%)")
    else:
        for th in [0.5, 0.7, 0.9]:
            rejected = sum(1 for r in valid
                           if r.get("label") == "NEU" and (r.get("self_reported_confidence") or 0) >= th)
            print(f"  NEU + 자기확신도 >= {th}: {rejected}/{n} ({rejected/n*100:.1f}%)  [참고용, 진짜 확률 아님]")
        argmax_neu = sum(1 for r in valid if r.get("label") == "NEU")
        print(f"  (참고) 라벨 기준 NEU 비율: {argmax_neu}/{n} ({argmax_neu/n*100:.1f}%)")


def call_jev_judge(premise: str, hypothesis: str, model: str = "jev-latest") -> dict:
    """
    NLI/Gemini 판정 함수들과 동일한 인터페이스로 맞춤:
    반환값 = {"label": "ENT"|"CON"|"NEU", "confidence": float, "probs": {...}}
    """
    payload = {
        "model": model,
        "state": f"[문서]\n{premise}\n\n[답변]\n{hypothesis}",
        "questions": {
            "entailment": {
                "type": "choice",
                "instructions": "문서가 답변을 함의(entail)하는가, 모순(contradict)하는가, 무관(neutral)한가?",
                "criteria": {
                    "entailment": "답변이 문서에 의해 뒷받침됨",
                    "contradiction": "답변이 문서와 모순됨",
                    "neutral": "답변이 문서로부터 판단 불가"
                }
            }
        }
    }
    resp = requests.post(
        JEV_ENDPOINT,
        headers={
            "Authorization": f"Bearer {JEV_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    ans = resp.json()["answers"]["entailment"]

    # 기존 ENT/CON/NEU 라벨 체계와 맞추기 위한 매핑
    label_map = {"entailment": "ENT", "contradiction": "CON", "neutral": "NEU"}
    return {
        "label": label_map[ans["choice"]],
        "confidence": ans.get("confidence"),
        "probs": ans.get("probabilities"),  # 응답 스키마에 있으면 그대로 노출
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pair", "atomic_hypothesis", "atomic_to_atomic"], default="pair",
                         help="pair: GT 전체 vs 답변 전체(문단 대 문단) / "
                              "atomic_hypothesis: GT 전체 vs 답변을 문장 단위로 쪼갠 것 각각 / "
                              "atomic_to_atomic: GT도 문장 단위로 쪼개서 체크리스트 매칭 (SelfCheckGPT+체크리스트 방식)")
    parser.add_argument("--backend", choices=["gemini", "nli"], default="nli")
    parser.add_argument("--nli-model", default="Huffon/klue-roberta-base-nli")
    parser.add_argument("--answer-mode", choices=["independent", "grounded"], default="grounded",
                         help="independent: GT 안 보여주고 질문만 (구버전 방식, 참고용) / "
                              "grounded: GT를 근거로 주고 그 내용 위주로만 답하게 지시 (이성훈 박사님 방식 재현)")
    args = parser.parse_args()

    if args.mode == "atomic_hypothesis":
        run_atomic_hypothesis(args.nli_model, args.answer_mode, backend=args.backend)
    elif args.mode == "atomic_to_atomic":
        if args.backend != "nli":
            print("주의: atomic_to_atomic 모드는 현재 NLI backend만 지원")
        run_atomic_to_atomic(args.nli_model, args.answer_mode)
    else:
        run(args.backend, args.nli_model, args.answer_mode)