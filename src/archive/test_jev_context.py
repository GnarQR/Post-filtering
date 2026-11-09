import os
from dotenv import load_dotenv

from jev_client import call_jev

load_dotenv()

key = os.environ["JEV_API_KEY"]

premise = """
2쌍의 염색체를 가지는 동물을 예로 들어 설명해 보자.
염색체가 2쌍인 경우에는 4가지 경우가 생긴다(2^2=4).
따라서 염색체쌍이 2쌍인 경우 서로 다른 정자 또는 난자가
4종류가 생기므로 이들이 서로 만나 만드는 조합은 4x4=16가지가 된다.

소는 염색체가 30쌍이므로 서로 다른 정자 또는 난자를 만들 경우의 수는
2^30=1,073,741,824가지이다.
""".strip()

s1 = """
한우의 염색체가 30쌍일 경우, 서로 다른 정자 또는 난자를 만들 경우의 수는
2의 30제곱인 1,073,741,824가지입니다.
""".strip()

s2 = """
이들이 서로 만나 각기 다른 형태의 조합을 만들 확률은
4x4=16가지가 됩니다.
""".strip()


tests = {
    "A_source_to_s2": (
        premise,
        s2,
    ),

    "B_s1_to_s2": (
        s1,
        s2,
    ),

    "C_source_plus_s1_to_s2": (
        premise + "\n\n[직전 답변]\n" + s1,
        s2,
    ),

    "D_source_to_full_answer": (
        premise,
        s1 + " " + s2,
    ),
}


for name, (p, h) in tests.items():
    result = call_jev(p, h, key)

    print(f"\n{name}")
    print(f"  label = {result['label']}")
    print(f"  p_entailment = {result['p_entailment']}")