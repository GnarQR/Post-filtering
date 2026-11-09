# E8: 축우 AI 안전 권고 3모델 비교 벤치마크

## 0. 핵심 설계

- **인간 Gold**: PASS / VIOLATION (이진). 업로드된 수동 검수 103개, dev 40 / test 63.
- **모델 예측**: PASS / REVIEW / VIOLATION (3분류). REVIEW는 **판정 유보**, 자동 PASS나 VIOLATION으로 강제 변환하지 않습니다.
- **모델**: TypeSafe JEV (`jev-latest`), Gemini 2.5 Flash-Lite, Gemini 2.5 Flash.
- **입력**: 질문·답변만. 생성 모델의 Gold/risk_reason/시나리오 라벨/수정 메모는 전송하지 않음.
- **제외**: 전문 수치·법규의 정밀 검증 및 RAG 문서-답변 NLI. E8의 JEV는 안전성 `choice` 분류기이며 기존 Semantic JEV와 다른 과제.

**연구적 제한**: 안전성 평가 기준은 연구자가 정의한 `일반 상식으로 식별 가능한 명백한 위험 권고`임. PASS는 수의학적 안전성을 보증하지 않음. 데이터는 Gemini로 **합성 생성**되었으므로 Gemini 자체 평가에서 생성기-평가기 연관 편향이 있을 수 있음. 독립 인간/전문가 다중 라벨 검증이 없는 연구자 단독 검수라는 한계를 논문에 보고해야 함. `final_qc_flag=true`인 4개 사례는 사용자의 판정을 임의로 변경하지 않고 포함하여 보고하며, 별도 민감도 분석을 권장.

## 1. 설치

압축을 현재 프로젝트의 **루트**에 풉니다.

```text
src/run_e8_safety_benchmark.py
data/e8_human_screened_103_provisional.jsonl
requirements_e8_benchmark.txt
README_E8_Benchmark_KO.md
tests/test_e8_benchmark.py
```

Windows PowerShell에서 프로젝트 루트 기준:

```powershell
python -m pip install -r .\requirements_e8_benchmark.txt
```

프로젝트 루트의 `.env` (파일은 절대 공유/업로드/커밋하지 않음):

```dotenv
GEMINI_API_KEY=본인_키
JEV_API_KEY=본인_키
```

JEV는 `TYPESAFE_API_KEY` 환경변수도 허용합니다. Gemini API는 `generateContent` REST, JEV는 TypeSafe의 `/v1/systemone` 호출을 사용합니다. 실제 권한·쿼터·과금은 사용자의 API 계정에 따라 달라집니다.

## 2. 순서 (Dev만 먼저)

```powershell
# 검증: 키 없이 가능. API 호출도 파일 쓰기도 없음
python .\src\run_e8_safety_benchmark.py --dry-run --pilot

# PASS와 VIOLATION 각 1개 × 3모델 = 최대 6회
python .\src\run_e8_safety_benchmark.py --pilot

# Dev 나머지까지 포함해 40개 × 3모델 = 총 120회 (6회는 SKIP)
python .\src\run_e8_safety_benchmark.py

# 저장 결과만 재집계 (API 추가 호출 없음)
python .\src\run_e8_safety_benchmark.py --evaluate-only
```

개별 모델만 시험:

```powershell
python .\src\run_e8_safety_benchmark.py --engine jev --limit 2
python .\src\run_e8_safety_benchmark.py --engine lite --limit 2
python .\src\run_e8_safety_benchmark.py --engine flash --limit 2
```

개발용 결과로 **프롬프트·생성 스키마·모델 버전·REVIEW 취급 및 평가식**을 확정한 다음, Test에서는 추가 탐색을 하지 않습니다.

```powershell
# Test 63개 × 3모델 = 최대 189회. 파일 별도 저장.
python .\src\run_e8_safety_benchmark.py --split test --final-test

# Test 요약 재생성 (API 없음)
python .\src\run_e8_safety_benchmark.py --split test --evaluate-only
```

기본 출력:

```text
results/e8_safety_dev_predictions.jsonl
results/e8_safety_dev_predictions.csv
results/e8_safety_dev_comparison.csv       # 같은 QA의 3모델 예측 나란히 비교
results/e8_safety_dev_summary.json
results/e8_safety_dev_errors.jsonl      # 오류 발생 시에만 생성
results/e8_safety_test_predictions.jsonl
results/e8_safety_test_predictions.csv
results/e8_safety_test_comparison.csv
results/e8_safety_test_summary.json
```

결과는 각 요청 성공 즉시 JSONL에 저장되고, 다시 실행하면 **성공 건만 SKIP**합니다. 일부 실패 시에는 오류 목록이 남고 종료 코드가 2이며, 재실행 시 그 건만 재호출합니다. **설정/프롬프트/원본 QA·Gold가 바뀌면 이전 결과와 섞는 대신 실행을 중단합니다**. 새로운 실험으로 바꾸려면 `--out`, `--csv`, `--paired-csv`, `--summary`, `--errors`를 모두 새 경로로 지정하거나 결과 파일을 별도로 보존하세요.

## 3. 지표 정의

Gold가 PASS/VIOLATION이고 예측이 PASS/REVIEW/VIOLATION이므로 **2 × 3 혼동행렬**을 기본으로 출력합니다.

- **Strict VIOLATION recall**: Gold VIOLATION 중 예측 VIOLATION의 비율. 예측 REVIEW는 *미탐지*로 취급하지만 FN(PASS)과 분리 기록.
- **Strict precision/F1**: 예측 VIOLATION만 양성으로 취급, REVIEW는 양성 예측에서 제외. 분모에는 VIOLATION Gold에서 REVIEW 예측한 건을 반영.
- **False-block rate**: Gold PASS인데 예측 VIOLATION한 비율.
- **REVIEW rate**: 전체 예측 중 REVIEW 비율.
- **Coverage**: REVIEW가 아닌 판정 비율.
- **Selective accuracy**: REVIEW를 제외한 자동 판정에서의 정확도.
- **Triage recall**: Gold VIOLATION 중 예측 VIOLATION **또는 REVIEW** 비율 (자동 통과시키지 않은 위험 응답).
- **Triage false-alert rate**: Gold PASS 중 예측 VIOLATION 또는 REVIEW 비율.

세 모델의 같은 case_id 예측을 비교하고, 전체 및 카테고리별 지표를 생성합니다. 모델 오류/호출 미완료가 있으면 `complete=false`로 표시되므로 불완전 표본을 최종 논문 성적으로 사용하지 마세요. 자동 분류와 사후 사람 검토를 분리해야 한다는 점에 유의하세요.

## 4. 설정 및 주의

기본 온도 `0`, Flash-Lite thinking budget `0`, Flash thinking budget `1024`, 최초 출력 상한 `4096`입니다. Gemini가 `MAX_TOKENS`로 끊기면 `8192`, `16384`까지 자동 재시도하고 **모든 시도 사용량**을 합산합니다. 이는 동일한 추론 예산을 강제하는 모델 공정 비교가 아니라 실제 운용 설정을 비교합니다. 공정성의 다른 정의가 필요하면 Dev에서 예산을 별도로 고정하세요.

`jev-latest`는 별칭이므로 JEV 모델 업데이트 시 실제 모델이 달라질 수 있습니다. 논문을 위한 최종 실행에서는 `--jev-model`에 API 계정에서 확인한 **고정 버전**을 주고 `model_resolved`를 기록하세요. Flash/Lite도 사용한 모델 ID와 응답의 `modelVersion`을 저장합니다.

`--pilot`은 개발용에만 허용됩니다. Test에 `--final-test`를 요구해 우발적인 테스트 튜닝을 예방합니다.

**중요**: 본 스크립트는 실제 Gemini/JEV 계정 API 키가 없으므로, 제공 시점에는 라이브 API 성능을 검증하지 못했습니다. 오프라인 모의 요청 단위 테스트와 Dev dry-run만 검증했습니다.

## 5. 코드 테스트 (API 불필요)

```powershell
python -m unittest discover -s tests -p "test_e8_benchmark.py" -v
```
