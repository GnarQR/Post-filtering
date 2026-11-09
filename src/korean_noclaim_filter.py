"""
korean_noclaim_filter.py

생성된 답변 문장 중 "비주장"(검증할 내용이 없는 문장)을 한국어 패턴으로 탐지.
영어 RAGTruth와의 핵심 차이: 한국어는 "~명시되어 있지 않습니다" 같은 표현이
독립 문장이 아니라 "~않으나, <진짜 주장>" 처럼 복합문에 섞여 나오는 경우가 많음.
그래서 fullmatch(문장 전체가 패턴과 일치)만 비주장으로 처리 -> 복합문은
자동으로 안 걸러짐(패턴이 문장 전체를 덮지 못하므로).

또한 "~없다"는 그 자체가 틀릴 수 있는 주장이라(실제론 문서에 있는데 모델이
거짓으로 "없다"고 할 수도 있음), 영어 때처럼 무조건 통과(ENT) 처리하지 않고
"힌트로만 표시 - 최종 판단은 사람이" 하는 용도로 둠. label_livestock_qna.py가
이 플래그를 보여주되 라벨링을 자동으로 건너뛰지는 않음.

사용법:
  python korean_noclaim_filter.py --input jev_scouted_qna.jsonl --out noclaim_flags.jsonl
  (jev_scouted_qna.jsonl 대신 generated_qna.jsonl도 가능 - sentences 필드만 있으면 됨)
"""

import argparse
import json
import re
from pathlib import Path

_CONN = r'(?:그러나|하지만|다만|따라서|또한|결론적으로)?\s*'
_TOPIC = (
    r'(?:(?!지만,|으나,|며,|다고\s|라고\s|한다\s|합니다\s).){0,150}'
    r'(?:에\s*대한|에\s*관한|관련된)?\s*(?:구체적인\s*)?'
)
_NOUN = r'(?:정보|내용|설명|수치|사례|기준|근거|방법|이유|사항)(?:은|는|이|가)?\s*'
_ABSENT = (
    r'(?:(?:문서에)?\s*(?:명시|언급|기재|제공)되어\s*있지\s*않(?:습니다|으며|으나|다)'
    r'|(?:문서에)?\s*나와\s*있지\s*않(?:습니다|으며|으나|다)'
    r'|(?:문서에)?\s*포함되어\s*있지\s*않(?:습니다|으며|으나|다))\.?'
)

PATTERNS = [
    re.compile(_CONN + _TOPIC + _NOUN + _ABSENT + r'$'),
    re.compile(r'^(?:따라서\s*)?.{0,40}?답변(?:을|은)?\s*(?:드리기|하기)\s*어렵습니다\.?$'),
    re.compile(r'^.{0,30}?(?:확인|파악)하기\s*어렵습니다\.?$'),
    re.compile(r'^.{0,30}?(?:정보가|내용이)\s*(?:부족|불충분)합니다\.?$'),
]


def is_noclaim_ko(sent: str) -> bool:
    s = sent.strip()
    return any(p.fullmatch(s) for p in PATTERNS)


def main(input_path: Path, out_path: Path):
    rows = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = []
    n_flagged = 0
    for r in rows:
        sentences = r.get("sentences") or [js["sent"] for js in r.get("jev_sentences", [])]
        flags = [is_noclaim_ko(s) for s in sentences]
        n_flagged += sum(flags)
        out.append({"sample_id": r.get("sample_id"), "sentences": sentences, "noclaim_flags": flags})

    out_path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")

    print(f"전체 {sum(len(o['sentences']) for o in out)}문장 중 {n_flagged}개를 비주장으로 플래그\n")
    print("=== 플래그된 문장 (전체 목록 - 오탐 없는지 직접 확인) ===")
    for o in out:
        for s, f in zip(o["sentences"], o["noclaim_flags"]):
            if f:
                print(f"  [{o['sample_id']}] {s}")

    print(f"\n저장 완료: {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("jev_scouted_qna.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("noclaim_flags.jsonl"))
    a = ap.parse_args()
    main(a.input, a.out)
