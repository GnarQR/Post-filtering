# E8 축우 위험 권고 평가 데이터셋 생성기

Gemini 2.5 Flash로 한우·젖소 상담에 등장할 법한 **위험한 관리 권고와 안전한 대조 답변**을 합성합니다. E6의 상식적 불가능성 판정과 달리, 이 데이터셋은 **실제로 가능한 행동을 부적절하게 권장하는 답변**을 다룹니다.

## 0. 압축 해제

프로젝트 폴더 `C:\Users\hanjw\Coding Project\축우 연구용 폴더` **루트에 ZIP 내용을 압축 해제**합니다.

```
축우 연구용 폴더/
├─ .env
├─ src/
│  ├─ generate_e8_flash_dataset.py
│  └─ finalize_e8_review.py
├─ data/
│  └─ e8_seed_plan.json
├─ requirements_e8.txt
├─ tests/
│  └─ test_generate_e8_flash_dataset.py
└─ results/  (실행 중 자동 생성)
```

기존 E6·E7 파일은 덮어쓰지 않습니다.

## 1. 설치

```powershell
python -m pip install -r .\requirements_e8.txt
```

`.env`:

```dotenv
GEMINI_API_KEY=여기에_본인_API_키
```

`.env`를 Git에 올리지 마세요.

## 2. 실행 순서 (비용 절약)

```powershell
# API 호출 없음, 계획 확인
python .\src\generate_e8_flash_dataset.py --dry-run --limit 2

# 처음 2개 시나리오에서 6개 QA 생성 (API 최대 2회)
python .\src\generate_e8_flash_dataset.py --limit 2

# 1차 검수 후 처음 8개 작업까지 확대 (기존 2개는 SKIP)
python .\src\generate_e8_flash_dataset.py --limit 8

# 전체 생성: 총 36개 API 작업 (30개 시나리오 + 6개 REVIEW 배치)
python .\src\generate_e8_flash_dataset.py
```

기본 모델은 `gemini-2.5-flash`, 사고 토큰 설정은 `--thinking-budget 1024`, 생성 상한은 `--max-output-tokens 4096`입니다. `MAX_TOKENS`가 발생하면 해당 작업만 8192/16384까지 자동 재시도합니다. 재시도 호출은 추가 비용이 발생할 수 있습니다.

기본 API 호출 수는 **36회**입니다(호출 재시도 제외). REVIEW 3개는 각 카테고리당 1회 호출에서 묶어 생성합니다.

다른 모델을 원하면 `--model ...`로 바꿀 수 있지만, **중간에 모델/temperature/프롬프트/seed를 바꾸면 이어하기가 차단**됩니다. 다른 설정은 별도 `--raw`, `--data-out`, `--csv`, `--summary`, `--review-sheet` 경로를 사용하세요.

한 카테고리만 시험하려면 `--category FOOD_SAFETY --limit 2`를 사용할 수 있습니다. 테스트 용도로 2개만 시행할 때 비용을 예상할 수 있습니다.

## 3. 데이터 설계

| 구분 | 단위 | 개수 | 라벨 초안 |
|---|---|---:|---|
| 독립 위험 시나리오 | 6영역 × 5상황 | 30 | 각 상황마다 VIOLATION 1 + PASS 2 |
| 파생 QA 대조 답변 | 30 × 3 | 90 | VIOLATION 30, PASS 60 |
| 별도 불확실한 상황 | 6영역 × 3상황 | 18 | REVIEW 18 |
| 합계 | 독립 상황 48개에서 파생 | **108** | PASS 60 / REVIEW 18 / VIOLATION 30 |

6가지 영역: `BREEDING`, `FOOD_SAFETY`, `ANIMAL_WELFARE`, `FEEDING`, `BIOSECURITY`, `FRAUD`.

**시나리오 단위 분할**: 영역당 위험 시나리오 2개 × 3답변 + REVIEW 1개 = dev 7개, 영역당 3개 × 3답변 + REVIEW 2개 = test 11개. 전체 **dev 42 / test 66**. 같은 위험 시나리오의 대조 답변이 두 분할에 섞이지 않습니다. 그러나 주제 범주는 두 분할에서 공유되므로 완전한 도메인 외삽 평가는 아닙니다.

`--limit`으로 먼저 본 생성 데이터는 프롬프트 검토에 사용하므로, **최종 holdout에 들어갈 주제나 답변을 보며 프롬프트를 조정하지 않도록** 주의하세요. 가장 엄격한 설계를 원한다면 검수 후 새로운 주제의 독립 평가셋을 별도 작성하세요. 현 `test` 표시는 시나리오 단위 사전 분할을 의미하며, 이미 검토해 본 문항이 있으면 이를 최종 블라인드 성능으로 주장하면 안 됩니다.

## 4. 출력 파일

| 경로 | 내용 |
|---|---|
| `results/e8_generation_raw.jsonl` | 각 API 작업의 JSON 결과 및 토큰, 원 설정 해시 (중간 저장 및 SKIP) |
| `data/e8_safety_108_draft.jsonl` | 평가용 QA 초안 및 분할/위험 근거 |
| `results/e8_safety_108_draft.csv` | 사람이 검토하는 간편 CSV |
| `results/e8_generation_summary.json` | 생성 진행률 및 비용 분석용 토큰 수 |
| `results/e8_human_review_sheet.csv` | **108건 생성이 완료됐을 때** 만들어지는 사람 검수용 시트 |

각 QA에는 `case_id`, `scenario_id`, `category`, `split`, `question`, `answer`, `gold_draft`, `risk_reason`, `quality_flags`, `reviewer_check`, `suggested_source_query` 등이 기록됩니다. `quality_flags`는 휴리스틱 주의 신호이며 자동 정답 검증이 아닙니다.

**중요: 생성형 모델이 직접 쓴 `gold_draft`는 최종 Gold Label이 아닙니다.** 모델이 만든 위험한 권고 자체가 실제로 타당한 E8 검증 사례인지, 안전한 대조문은 정말 PASS인지 별도 자료를 확인해야 합니다. `suggested_source_query`는 출처가 아니라 자료 검색어입니다.

## 5. 사람 검수 후 최종 QA 저장

`results/e8_human_review_sheet.csv`에서 다음 열을 채웁니다.

- `human_gold`: PASS / REVIEW / VIOLATION
- `fact_verified`: YES (해당 사례의 위험성/라벨 검토를 완료했을 때만)
- `source_url`: 확인한 기관·학술 근거 링크
- `reviewer`: 검수자 식별명 (실제 전문가 검수 여부는 별도로 기록)
- `review_notes`: Gold 변경 이유, 적용 범위 등

그다음:

```powershell
# 완전한 시나리오 대조묶음이 검수된 경우만 저장 (부분 가능)
python .\src\finalize_e8_review.py

# 전체 108건을 모두 확인해야만 완료
python .\src\finalize_e8_review.py --require-all
```

검수가 완료되지 않은 행은 결과에서 제외됩니다. `data/e8_safety_verified.jsonl`의 `gold`만 최종 평가에 사용하세요.

**검수 시트는 기존 파일이 있으면 덮어쓰지 않습니다.** 이미 작성한 내용을 유지합니다. 의도적으로 재생성하려면 `--reset-review-sheet`가 필요하며, 기존 검수 내용이 사라질 수 있습니다.

## 6. 자동 테스트

```powershell
python -m unittest discover -s tests -v
```

테스트는 **모의 응답**을 사용하며 실제 API를 호출하지 않습니다. JSON 필드 검사, 중단/재개, fingerprint, 108개/42·66/30·60·18 분포를 확인합니다.

## 7. 연구상 주의

- 같은 상황에서 파생된 3개 답변은 **서로 독립 표본이 아닙니다**. `scenario_id` 기준으로 통계·분할을 관리하세요.
- 금지어/강한 부정 표현 등에 과적합하지 않도록, 사례를 직접 검토하세요.
- 합성 데이터셋만으로 실제 LLM이 위험 답변을 얼마나 자주 생성한다고 추정할 수 없습니다.
- E8은 **부적절한 관리 권고를 선별하는 기능**으로, E6의 명백한 상식·논리적 불가능성 환각 탐지와 구분해 보고하세요.
- 실제 모델 검증 성능은 별도 평가 코드와 독립 데이터로 측정해야 합니다. 생성기 자체는 검증기가 아닙니다.
- 위험 권고 문장은 테스트 목적의 비실행적 한두 문장으로 제한되어 있습니다. 실제 현장에 적용하지 마세요.
