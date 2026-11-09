"""
generate_qna.py

sample_pages.py가 만든 sampled_pages.jsonl의 각 샘플(premise_chunks)에
대해, Gemini로 (1) 그 내용으로 답할 수 있는 자연스러운 질문을 하나 생성하고,
(2) 같은 내용을 근거로 grounded 답변을 생성. 답변은 문장 단위로 쪼개서 저장
(사람 라벨링 단계에서 문장별로 지지/환각 표시할 수 있게).

- 질문 생성 시 "원하는 환각 양"을 유도하지 않음(자연 발생 그대로 관찰하는 게 목적) ->
  단, grounded 프롬프트 자체는 "문서에 있는 내용만 근거로 답하라"는 일반적인 지시만 포함
- 결과를 이어하기 지원 (sample_id 기준, 이미 생성된 건 재호출 안 함)
- 문장 분리는 간단한 규칙 기반(정교한 형태소 분석 없이, 종결어미+마침표 패턴)

필요: .env에 GEMINI_API_KEY

사용법:
  python generate_qna.py --samples sampled_pages.jsonl --limit 3   # 먼저 3개만
  python generate_qna.py --samples sampled_pages.jsonl             # 전체
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

Q_PROMPT = """다음은 축산(소 사육) 관련 문서의 일부입니다. 이 내용을 읽고,
실제 농가나 연구자가 물어볼 법한 자연스럽고 구체적인 질문을 하나만 한국어로 작성하세요.

지침:
1. 반드시 이 문서 내용만으로 답할 수 있는 질문이어야 합니다.
2. "예/아니오"로 끝나는 단순 질문이 아니라, 설명이 필요한 질문으로 작성하세요.
3. 질문 하나만 출력하고, 다른 설명이나 번호를 붙이지 마세요.
4. 질문의 모든 핵심 요구사항에 대한 답이 반드시 문서 안에 명시적으로 존재해야 합니다.
   문서에 직접 제시되지 않은 원인, 영향, 결과, 예측, 사례, 개선 효과 등을
   추론하도록 요구하는 질문은 만들지 마세요.
5. 질문을 작성한 뒤, 외부 지식이나 추가적인 추론 없이 제공된 문서의
   문장, 수치, 설명만으로 질문 전체에 답할 수 있는지 스스로 확인하세요.
   일부라도 문서에서 답할 수 없다면 질문을 다시 작성하세요. 
서로 다른 문장이나 사실을 연결하여 문서에 명시되지 않은 인과관계, 계산 관계 또는 조건 관계를 새로 만들어 질문하지 마세요.

[문서]
{context}

[질문]"""

A_PROMPT = """다음은 축산(소 사육) 관련 문서와 질문입니다. 주어진 문서 내용에만
근거하여 질문에 답하세요.

지침:
1. 문서에 있는 내용만 사용하고, 문서에 없는 내용을 추측해서 덧붙이지 마세요.
2. 친절하고 자연스러운 설명체로, 2~5문장 정도로 답하세요.
3. 바로 답변 내용만 출력하세요 (서두 인사말 불필요).
4. 문서에 명시되지 않은 상황을 가정하거나,
   문서에 없는 원인·영향·예측·구체적 사례를 추론해야만
   답할 수 있는 질문은 만들지 마세요.
5. 답은 문서에 명시적으로 제시된 사실, 절차, 수치 또는
   이를 단순히 조합하여 얻을 수 있는 내용이어야 합니다.
6. 문서에 제시된 주체·대상·관계·단위를 그대로 유지하고,
   서로 다른 대상이나 단계의 정보를 바꾸어 연결하지 마세요.

[문서]
{context}

[질문]
{question}

[답변]"""

_model_cache = {}

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

def get_model(model_name: str):
    if model_name not in _model_cache:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model_cache[model_name] = genai.GenerativeModel(
            model_name, generation_config={"temperature": 0.7})
    return _model_cache[model_name]


def call_gemini(prompt: str, model_name: str, max_retries: int = 4) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_model(model_name).generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini 호출 {max_retries}회 실패: {last_err}")


SENT_END = re.compile(r'(?<=[.!?다요음함임])\s*\n|(?<=[.!?])\s+(?=[가-힣A-Z0-9"\'(])')


def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = SENT_END.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    # 너무 짧은 조각(제목/번호 등)은 바로 앞 문장에 합침
    merged = []
    for p in parts:
        if merged and len(p) < 8 and not re.search(r'[.!?]$', p):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def main(samples_path: Path, out_path: Path, model_name: str, limit: int):
    samples = load_jsonl_robust(samples_path)
    if limit:
        samples = samples[:limit]

    done_ids = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done_ids.add(json.loads(l)["sample_id"])
        if done_ids:
            print(f"기존 결과 {len(done_ids)}개 발견 -> 이어서 진행")

    with out_path.open("a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if s["sample_id"] in done_ids:
                continue
            context = "\n\n".join(s["premise_chunks"])

            question = call_gemini(Q_PROMPT.format(context=context), model_name)
            time.sleep(0.5)
            answer = call_gemini(A_PROMPT.format(context=context, question=question), model_name)
            time.sleep(0.5)

            sentences = split_sentences(answer)
            row = dict(s)
            row["question"] = question
            row["answer"] = answer
            row["sentences"] = sentences
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(samples)}] {s['sample_id']} ({s['domain']})")
            print(f"  Q: {question}")
            print(f"  A: {answer[:80]}... ({len(sentences)}문장)\n")

    print(f"완료: {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=Path("sampled_pages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("generated_qna.jsonl"))
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.samples, a.out, a.model, a.limit)
