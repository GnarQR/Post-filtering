import time
import requests

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

LABEL_MAP = {
    "entailment": "ENT",
    "contradiction": "CON",
    "neutral": "NEU",
}


def call_jev(
    premise: str,
    hypothesis: str,
    api_key: str,
    max_retries: int = 5,
) -> dict:

    payload = {
        "model": "jev-1.14.0",
        "state": f"[문서]\n{premise}\n\n[답변]\n{hypothesis}",
        "questions": {
            "entailment": {
                "type": "choice",
                "instructions": (
                    "문서가 답변을 함의(entail)하는가, "
                    "모순(contradict)하는가, 무관(neutral)한가?"
                ),
                "criteria": {
                    "entailment": "답변이 문서에 의해 뒷받침됨",
                    "contradiction": "답변이 문서와 모순됨",
                    "neutral": "답변이 문서로부터 판단 불가",
                },
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                JEV_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if resp.status_code in (429, 500, 502, 503, 529):
                wait = 2 ** attempt
                print(
                    f"  [JEV {resp.status_code}, "
                    f"{wait}초 후 재시도 {attempt + 1}/{max_retries}]"
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()

            data = resp.json()
            ans = data["answers"]["entailment"]
            probs = ans.get("probabilities", {})

            return {
                "label": LABEL_MAP[ans["choice"]],
                "p_entailment": probs.get("entailment"),
                "p_contradiction": probs.get("contradiction"),
                "p_neutral": probs.get("neutral"),
                "resolved_model": data.get("model"),
            }

        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"JEV 호출 {max_retries}회 실패: {last_err}"
    )