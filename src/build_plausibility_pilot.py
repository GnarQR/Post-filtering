"""
build_plausibility_pilot_v2.py

v1 오류 수정:
- TMP_003 교체용 Temporal 후보를 v4_all.csv에서 찾지 않음.
- 교체 후보를 코드 내부 controlled source claim으로 직접 정의.
- 따라서 plausibility_candidates_v4_top20.csv만 있으면 실행 가능.

입력:
  results/plausibility_candidates_v4_top20.csv

출력:
  results/plausibility_pilot20_curated.csv
  data/plausibility_pilot80.jsonl

실행:
  python ./src/build_plausibility_pilot_v2.py
"""

import csv
import json
from pathlib import Path


TOP_PATH = Path("results/plausibility_candidates_v4_top20.csv")

CURATED_OUT = Path("results/plausibility_pilot20_curated.csv")
CHALLENGE_OUT = Path("data/plausibility_pilot80.jsonl")


# -------------------------------------------------------------------
# v4 top20에서 제외할 후보
# -------------------------------------------------------------------

DROP_IDS = {
    "TMP_003",
}


# -------------------------------------------------------------------
# 교체용 Temporal claim
#
# v4 후보 pool에 없어도 직접 사용한다.
# 기존 축우 문서에서 확인했던 문장을 기반으로 한 controlled pilot source.
# -------------------------------------------------------------------

TEMP_REPLACEMENT_ROW = {
    "candidate_id": "__TEMP_REPLACEMENT__",
    "category": "temporal_order",
    "sample_id": "06_농업기술길라잡이_가축위생과_질병(소)_p24",
    "domain": "질병",
    "source_origin": "answer",
    "sentence_index": "0",

    "source_claim":
        "송아지가 분만 후 12시간이 지난 후에 초유를 섭취하면, "
        "분만 직후 섭취했을 때보다 체내 면역글로불린 농도가 "
        "절반 정도로 떨어집니다.",

    "question":
        "송아지가 분만 후 12시간이 지난 후에 초유를 섭취했을 때, "
        "분만 직후 섭취했을 때에 비해 체내 면역글로불린 농도가 "
        "절반 정도로 떨어지는 이유는 무엇인가요?",

    "answer":
        "송아지가 분만 후 12시간이 지난 후에 초유를 섭취하면, "
        "분만 직후 섭취했을 때보다 체내 면역글로불린 농도가 "
        "절반 정도로 떨어집니다.",

    "sample_reference_label": "Supported",
    "sentence_reference_label": "Supported",
}


# -------------------------------------------------------------------
# 각 후보의 controlled corruption
# -------------------------------------------------------------------

CORRUPTIONS = {
    # ===============================================================
    # Biological
    # ===============================================================
    "BIO_001": {
        "corrupted_claim":
            "소 수정란 이식(embryo transfer, ET)은 우수한 특성을 가진 암소에서 "
            "채취한 수정란을 다른 수소에 이식하여 임신시키는 기술입니다.",
        "violation": "male_cattle_pregnancy_via_embryo_transfer",
        "edit_type": "recipient_sex_swap",
    },

    "BIO_002": {
        "corrupted_claim":
            "문서에 따르면 수소의 임신 시기에 필요한 구리(Cu)의 요구량은 "
            "10.00 mg/kg이며, 구리의 최대 한계 농도는 40.00 mg/kg입니다.",
        "violation": "male_cattle_pregnancy",
        "edit_type": "female_to_male",
    },

    "BIO_003": {
        "corrupted_claim":
            "임신한 수소의 경우 0.5~0.7%, 그리고 비유 중인 암소의 경우에는 "
            "임신한 수소보다 좀 더 높은 수준의 칼륨이 요구됩니다.",
        "violation": "male_cattle_pregnancy",
        "edit_type": "pregnant_cow_to_bull",
    },

    "BIO_004": {
        "corrupted_claim":
            "번식용 수소의 주요관리에서 임신한 수소에 대한 주요관리로서 "
            "임신 초기의 개체관리와 임신기간 중의 운동, 그리고 분만 직전의 "
            "조치사항들을 들 수 있습니다.",
        "violation": "pregnant_and_calving_bull",
        "edit_type": "cow_to_bull_role_swap",
    },

    # ===============================================================
    # Temporal
    # ===============================================================
    "TMP_001": {
        "corrupted_claim":
            "어린 송아지사료는 출생 10일 전부터 섭취 훈련을 시작하여 "
            "생후 20~30일령 이후에는 하루 300g 이상 섭취하도록 하고, "
            "양질 목건초는 출생 직후부터 자유롭게 먹도록 해야 합니다.",
        "violation": "feeding_before_birth",
        "edit_type": "post_birth_to_pre_birth",
    },

    "TMP_002": {
        "corrupted_claim":
            "송아지는 태어나기 2~3개월 전부터 젖이나 액상사료를 먹으며 "
            "이러한 소화 기능을 사용합니다.",
        "violation": "oral_feeding_before_birth",
        "edit_type": "post_birth_function_to_pre_birth_feeding",
    },

    "TMP_004": {
        "corrupted_claim":
            "출생 4주 전의 송아지에게 1일 2회 급여하는 액상사료량은 "
            "생시체중이 23~29kg일 때 2.3kg입니다.",
        "violation": "feeding_before_birth",
        "edit_type": "age_to_pre_birth",
    },

    "__TEMP_REPLACEMENT__": {
        "corrupted_claim":
            "송아지가 분만 12시간 전에 초유를 섭취하면, "
            "분만 직후 섭취했을 때보다 체내 면역글로불린 농도가 "
            "절반 정도로 떨어집니다.",
        "violation": "colostrum_before_birth",
        "edit_type": "post_calving_to_pre_calving",
    },

    # ===============================================================
    # Relational
    # ===============================================================
    "REL_001": {
        "corrupted_claim":
            "암소의 아비가 그 암소 자신이라고 하고 육종가가 10이라고 하면, "
            "그 암소는 자기 자신의 딸이 됩니다.",
        "violation": "self_parent_relation",
        "edit_type": "father_to_self",
    },

    "REL_002": {
        "corrupted_claim":
            "암소의 외조부가 그 암소 자신이고 외조모 능력은 0으로 가정하면, "
            "암소는 자기 자신의 외손녀가 됩니다.",
        "violation": "self_grandparent_relation",
        "edit_type": "maternal_grandsire_to_self",
    },

    "REL_003": {
        "corrupted_claim":
            "어미소의 송아지가 그 어미소의 어미이기도 하며, "
            "분만 후 생후 3개월까지 모유에 대한 의존도가 매우 높습니다.",
        "violation": "cyclic_parent_child_relation",
        "edit_type": "child_becomes_parent",
    },

    "REL_004": {
        "corrupted_claim":
            "딸 소는 자기 자신의 아비이자 어미이며, "
            "딸 소의 육종가는 자기 자신의 육종가를 이용하여 계산됩니다.",
        "violation": "self_parent_relation",
        "edit_type": "parents_to_self",
    },

    # ===============================================================
    # Numeric / physical
    # ===============================================================
    "NUM_001": {
        "corrupted_claim":
            "옥수수 담근먹이를 급여했을 때 일당증체량은 0.75kg이며, "
            "1kg 증체당 농후사료소요량은 -5.6kg, 도체율은 61.6%로 나타났습니다.",
        "violation": "negative_feed_requirement",
        "edit_type": "positive_quantity_to_negative",
    },

    "NUM_002": {
        "corrupted_claim":
            "젖소(홀스타인)는 27°C 이상의 고온 환경에서 "
            "사료 섭취량이 107~112% 감소하고, 유량은 120~130% 줄어드는 것으로 나타납니다.",
        "violation": "decrease_over_100_percent",
        "edit_type": "percentage_beyond_physical_bound",
    },

    "NUM_003": {
        "corrupted_claim":
            "반추위가 충분히 발달되지 않은 어린 송아지는 콜린을 생성하는 능력이 "
            "떨어지기 때문에 우유의 0.26% 또는 체중 1kg당 -21mg을 "
            "급여하도록 권장합니다.",
        "violation": "negative_dose",
        "edit_type": "positive_dose_to_negative",
    },

    "NUM_004": {
        "corrupted_claim":
            "반추동물은 생애 전체 사료 섭취량의 -30% 이상을 "
            "조사료를 포함하여 섭취해야 합니다.",
        "violation": "negative_feed_proportion",
        "edit_type": "positive_proportion_to_negative",
    },

    # ===============================================================
    # Unit / scale
    # ===============================================================
    "UNT_001": {
        "corrupted_claim":
            "체중 500톤인 수소가 일당 1.0kg의 증체를 목표로 할 때, "
            "필요한 건물량은 9.5kg입니다.",
        "violation": "cattle_weight_in_tons",
        "edit_type": "kg_to_ton",
    },

    "UNT_002": {
        "corrupted_claim":
            "특히 임신 시기에 따라 요구량이 증가하는데, 임신 초기에는 하루 0.5kg, "
            "임신 100일에서 225일 사이에는 하루 1.5kg, "
            "임신 225일 이상이 되면 하루 2.0kg의 구리가 필요합니다.",
        "violation": "trace_mineral_mg_to_kg",
        "edit_type": "mg_to_kg",
    },

    "UNT_003": {
        "corrupted_claim":
            "2007년부터 2017년까지 젖소의 연간 유전적 개량량 추세에서 "
            "유량은 매년 26.25톤, 유지방량은 0.96톤, "
            "유단백량은 0.71톤씩 증가했습니다.",
        "violation": "genetic_gain_kg_to_ton",
        "edit_type": "kg_to_ton",
    },

    "UNT_004": {
        "corrupted_claim":
            "체중 150톤인 송아지의 대사단백질의 정미단백질로의 전환 효율은 "
            "66.3%이며, 체중 300톤인 육성단계 거세우의 전환 효율은 49.2%입니다.",
        "violation": "cattle_weight_kg_to_ton",
        "edit_type": "kg_to_ton",
    },
}


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def apply_curations(top_rows):
    final_rows = []

    # ---------------------------------------------------------------
    # v4 top20
    # ---------------------------------------------------------------
    for row in top_rows:
        cid = row["candidate_id"]

        if cid in DROP_IDS:
            continue

        if cid not in CORRUPTIONS:
            raise RuntimeError(
                f"corruption rule 없음: {cid}"
            )

        info = CORRUPTIONS[cid]

        out = dict(row)
        out["manual_keep"] = "Y"
        out["manual_corrupted_claim"] = info["corrupted_claim"]
        out["violation"] = info["violation"]
        out["edit_type"] = info["edit_type"]

        final_rows.append(out)

    # ---------------------------------------------------------------
    # temporal replacement를 직접 삽입
    # ---------------------------------------------------------------
    replacement = dict(TEMP_REPLACEMENT_ROW)
    info = CORRUPTIONS["__TEMP_REPLACEMENT__"]

    replacement["manual_keep"] = "Y"
    replacement["manual_corrupted_claim"] = info["corrupted_claim"]
    replacement["violation"] = info["violation"]
    replacement["edit_type"] = info["edit_type"]

    final_rows.append(replacement)

    # ---------------------------------------------------------------
    # category별 정렬 + 새 pilot_id
    # ---------------------------------------------------------------
    order = {
        "biological_role": 0,
        "temporal_order": 1,
        "relational": 2,
        "numeric_physical": 3,
        "unit_scale": 4,
    }

    prefix = {
        "biological_role": "BIO",
        "temporal_order": "TMP",
        "relational": "REL",
        "numeric_physical": "NUM",
        "unit_scale": "UNT",
    }

    final_rows.sort(
        key=lambda r: (
            order[r["category"]],
            r["sample_id"],
            str(r.get("sentence_index", "")),
        )
    )

    counters = {
        cat: 0
        for cat in order
    }

    for row in final_rows:
        cat = row["category"]
        counters[cat] += 1

        row["pilot_id"] = (
            f"{prefix[cat]}_{counters[cat]:03d}"
        )

    # ---------------------------------------------------------------
    # sanity check
    # ---------------------------------------------------------------
    if len(final_rows) != 20:
        raise RuntimeError(
            f"최종 source claim 수가 20이 아님: {len(final_rows)}"
        )

    for cat in order:
        n = sum(
            row["category"] == cat
            for row in final_rows
        )

        if n != 4:
            raise RuntimeError(
                f"{cat} 개수 오류: {n}"
            )

    return final_rows


def write_curated_csv(rows):
    CURATED_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "pilot_id",
        "category",
        "sample_id",
        "domain",
        "source_origin",
        "sentence_index",
        "source_claim",
        "manual_corrupted_claim",
        "violation",
        "edit_type",
        "question",
        "answer",
        "manual_keep",
    ]

    with CURATED_OUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def make_challenge_rows(curated_rows):
    conditions = {
        "A": {
            "name": "supported_plausible",
            "premise": "normal",
            "answer": "normal",
            "evidence_gold": "Supported",
            "plausibility_gold": "Plausible",
        },

        "B": {
            "name": "unsupported_implausible",
            "premise": "normal",
            "answer": "corrupt",
            "evidence_gold": "Unsupported",
            "plausibility_gold": "Implausible",
        },

        "C": {
            "name": "supported_implausible",
            "premise": "corrupt",
            "answer": "corrupt",
            "evidence_gold": "Supported",
            "plausibility_gold": "Implausible",
        },

        "D": {
            "name": "unsupported_plausible",
            "premise": "corrupt",
            "answer": "normal",
            "evidence_gold": "Unsupported",
            "plausibility_gold": "Plausible",
        },
    }

    output = []

    for row in curated_rows:
        normal = row["source_claim"]
        corrupt = row["manual_corrupted_claim"]

        for letter, cond in conditions.items():

            premise = (
                normal
                if cond["premise"] == "normal"
                else corrupt
            )

            answer = (
                normal
                if cond["answer"] == "normal"
                else corrupt
            )

            output.append({
                "challenge_id":
                    f"{row['pilot_id']}_{letter}",

                "group_id":
                    row["pilot_id"],

                "condition":
                    letter,

                "condition_name":
                    cond["name"],

                "category":
                    row["category"],

                "violation":
                    row["violation"],

                "edit_type":
                    row["edit_type"],

                "source_sample_id":
                    row["sample_id"],

                "source_origin":
                    row["source_origin"],

                "normal_claim":
                    normal,

                "corrupted_claim":
                    corrupt,

                "premise":
                    premise,

                "answer":
                    answer,

                "evidence_gold":
                    cond["evidence_gold"],

                "plausibility_gold":
                    cond["plausibility_gold"],
            })

    return output


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    if not TOP_PATH.exists():
        raise FileNotFoundError(
            TOP_PATH
        )

    top_rows = read_csv(
        TOP_PATH
    )

    curated = apply_curations(
        top_rows
    )

    challenge = make_challenge_rows(
        curated
    )

    write_curated_csv(
        curated
    )

    write_jsonl(
        CHALLENGE_OUT,
        challenge,
    )

    print("=" * 82)
    print("Plausibility pilot challenge set 생성 완료")
    print("=" * 82)

    print(
        f"Curated source claims : {len(curated)}"
    )

    print(
        f"A/B/C/D samples      : {len(challenge)}"
    )

    print()

    categories = [
        "biological_role",
        "temporal_order",
        "relational",
        "numeric_physical",
        "unit_scale",
    ]

    for cat in categories:
        n = sum(
            x["category"] == cat
            for x in curated
        )

        print(
            f"{cat:18s}: {n}"
        )

    print()

    for cond in ["A", "B", "C", "D"]:
        n = sum(
            x["condition"] == cond
            for x in challenge
        )

        print(
            f"Condition {cond}: {n}"
        )

    print()
    print(
        f"[20 claims] {CURATED_OUT.resolve()}"
    )
    print(
        f"[80 cases ] {CHALLENGE_OUT.resolve()}"
    )

    print()
    print(
        "다음 실험 핵심: condition C 20개가 "
        "JEV에서 ENT로 통과하는 비율 확인"
    )


if __name__ == "__main__":
    main()
