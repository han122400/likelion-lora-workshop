# Qwen3.5-2B에 LoRA 적용하기

학생은 제공된 `train.py` 상단의 **LEARNING_RATE, EPOCHS, LORA_RANK** 중 하나를 바꾸고, 새로 실행한 결과를 비교합니다. 모델을 호출하는 유료 API나 API 키는 사용하지 않습니다. 공개 모델을 내려받아 GPU에서 직접 학습합니다.

> **검증 범위.** 이전 데이터로 명령줄 전체 실행(`python train.py`, Qwen3.5-2B, Colab T4)이 약 9분, GPU 메모리 약 6GB로 완료되는 것을 확인했습니다. 현재 데이터와 노트북의 단계별 실행은 GPU에서 아직 확인하지 않았고, 가짜 모델로 연결만 테스트했습니다. 수업 전에 진행자가 Colab에서 노트북을 처음부터 끝까지 한 번 실행해야 합니다. 학습 시간과 성능 향상은 보장하지 않습니다.

## 0. Colab에서 실습하기 (기본)

실습 사이트의 **Colab 열기**로 새 노트북을 열고, ‘LoRA 실습과 비교’ 챕터의 코드 블록을 순서대로 새 코드 셀에 붙여 넣어 실행합니다. 단계마다 입력·출력·점수가 카드, 표, 그래프로 표시됩니다.

1. 실습 사이트에서 `lora-workshop.zip` 다운로드
2. Colab 열기 → 런타임 → 런타임 유형 변경 → **T4 GPU**
3. 1-2 코드 실행 → ‘파일 선택’에서 `lora-workshop.zip` 업로드
4. 1-3 설치 → **런타임 → 세션 다시 시작** → 1-4부터 계속

같은 코드를 한 파일로 모은 `lora_workshop.ipynb`도 ZIP에 들어 있습니다. Colab의 파일 → 노트북 업로드로 열면 셀을 붙여 넣지 않고 진행할 수 있습니다.

| 단계 | 노트북 셀 | 하는 일 | 화면에 보이는 것 |
| --- | --- | --- | --- |
| 1 환경 세팅 | 1-1 ~ 1-4 | GPU 확인, 파일 받기, 설치 | GPU 이름, 패키지 버전, 데이터 구성표 |
| 2 기본 모델 | 2-1 필수 · 2-2 선택 | 모델 불러오기·질문·평가를 한 셀로 실행, (선택) 직접 질문 | 가이드 5개 답변 카드·표, 검증 10문항 점수·표 |
| 3 LoRA 학습 | 3-1 ~ 3-3 | 기본 변수로 학습 | 학습 파라미터 비율, loss 그래프 |
| 4 LoRA 모델 | 4-1 필수 · 4-2 선택 | 학습한 모델로 같은 질문·평가를 한 셀로 실행, (선택) 직접 질문 | 가이드 5개 답변 카드·표, 검증 10문항 점수·표 |
| 5 비교 | 5-1 ~ 5-4 | 학습 전후 비교, test로 마지막 확인, 결과 저장 | ▲개선·▼악화 비교표, 지표 막대, test 점수 |
| 6 과제 | 6-1 ~ 6-4 | 빈칸에 내 문장 3개 쓰기 → 변수 바꿔 학습 → 자유 질문 | 내 문장 정확 일치·비슷한 정도, 실험 순위표 |

2단계는 **2-1**, 4단계는 **4-1**만 실행하면 질문부터 평가까지 끝납니다. 각 셀은 validation 10문항에 한 번씩만 답하고, 그중 `guide.json`에 정한 5개의 결과를 순서대로 골라 카드와 표로 보여 줍니다. 같은 답을 다시 생성하지 않으며, 5단계 비교에서도 이 결과를 그대로 씁니다. 직접 질문은 선택 셀 **2-2 / 4-2**에서 해 볼 수 있습니다. 앞 셀을 건너뛰면 '먼저 ○-○ 셀을 실행하세요' 안내가 나옵니다.

| 파일 | 역할 |
| --- | --- |
| `lora_workshop.ipynb` | 웹 가이드 코드를 한 파일로 모은 6단계 노트북(웹이 이 파일의 코드를 그대로 보여 줌) |
| `train.py` | 모델 불러오기·질문·학습·평가·실험 함수(노트북에서 `import train as lab`), 명령줄 실행 |
| `lab_view.py` | 답변 카드·표·막대·loss 그래프를 그리는 함수(노트북에서 `import lab_view as view`) |
| `guide.json` | 2단계·4단계에 넣는 웹 가이드 입력 |
| `train.jsonl` · `validation.jsonl` · `test.jsonl` · `knowledge.json` | 실습 데이터 |
| `requirements.txt` | 설치 목록 |

### 6단계 과제: 나만의 출력 학습

1~5단계가 **답하는 형식**(근거 ID, 답 보류)을 학습했다면, 과제는 **문장 내용 자체**를 학습합니다. 6-1 예시 코드의 `"____"` 빈칸에 모델이 글자 그대로 말하길 원하는 문장 3개와, 문장마다 질문 a·b(학습용)·c(시험용)를 씁니다. 문서는 주지 않습니다.

- 학습 예시는 6개(문장 3 × 질문 a·b)이고, 예시 하나마다 바로 업데이트합니다(누적 1).
- 채점: 학습 질문 정확 일치, **처음 보는 질문(c) 정확 일치**, **관계없는 질문 보존**(내 문장이 엉뚱하게 섞이지 않는지). 정확 일치는 공백만 무시하고 글자가 모두 같아야 합니다.
- 순위: ① 처음 보는 질문 정확 일치 ② 관계없는 질문 보존 ③ 학습 질문 정확 일치 ④ 스텝이 적을수록 ⑤ 학습 시간이 짧을수록.
- 결과: `outputs/custom-날짜-시간/`, 기록: `outputs/custom_experiments.jsonl`.
- 기본값(epoch 2)으로는 대부분 외우지 못합니다. epoch·학습률을 올리며 정확히 나오는 최소 설정을 찾습니다. 실제 필요한 값과 시간은 GPU에서 확인하지 않았습니다.

## 1. 어떤 실험인가요?

이번 실습에서는 **LoRA 학습 전후의 답변**을 비교합니다. 지난 시간에 배운 RAG는 슬라이드에서 개념만 비교하며, RAG를 적용하거나 적용 전후의 답변을 비교하는 실습은 하지 않습니다.

**멋사 중부대 14기** 운영 안내문을 사용합니다. 정규 세미나 시간과 운영진 정보는 실제 14기 정보이고, 나머지 규정·일정·금액은 교육용 예시입니다. 실제 정보가 들어 있는 문서는 T01(정규 세미나), T11(대표·부대표), V03(총무), F04(홍보부장)입니다.

| 조건 | 모델 | 질문에 제공하는 자료 |
| --- | --- | --- |
| 학습 전 (`base`) | 원본 Qwen3.5-2B | 질문 + 예제에 미리 담긴 문서 |
| 학습 후 (`lora`) | 같은 기본 모델에 LoRA 어댑터 적용 | 학습 전과 같은 질문 + 같은 문서 |

**기본 모델을 먼저 평가하고, LoRA를 학습한 뒤 같은 문항으로 다시 평가합니다.** 두 평가에는 같은 질문과 문서, 시스템 지시문, no-thinking 채팅 템플릿, greedy 생성 설정과 출력 길이를 사용합니다. 문서 내용이 같을 때 학습 전후로 답변 형식과 근거 활용이 어떻게 달라지는지 살펴봅니다.

문서는 각 JSONL 예제의 `document`에 미리 들어 있습니다. 실행 중 검색으로 문서를 고르지 않습니다. `knowledge.json`은 이 문서들의 원문을 보관하고, 예제의 문서 내용과 ID가 맞는지 확인하는 데 사용합니다.

## 2. 실행 환경

- 이 스크립트의 학습 경로는 **NVIDIA CUDA GPU**용입니다. CPU나 내장 그래픽으로 Qwen3.5-2B 학습이 가능하다고 가정하지 않습니다.
- Python 3.11 또는 3.12와 CUDA 지원 PyTorch 환경을 권장합니다.
- Colab T4(약 15 GiB)에서 실행했을 때 최대 사용량은 약 6 GiB였습니다. 전체 12 GiB 또는 사용 가능 8 GiB 미만이면 메모리 안내를 표시하되 총용량만으로 실행을 차단하지 않습니다. 이 수치는 주의를 위한 기준이며 공식 최소 사양이나 성공 보장이 아닙니다. 실제 모델 로딩·학습 중 메모리가 부족하면 오류로 중단합니다. rank, 입력 길이, 커널, 패키지 버전에 따라 필요량이 달라집니다.
- 모델을 처음 내려받을 때 인터넷 연결과 디스크 공간이 필요합니다.
- Qwen3.5 공식 Unsloth 안내에 맞춰 **16bit LoRA**를 사용합니다. 4bit QLoRA는 기본 구성으로 사용하지 않습니다. Ampere 이상 GPU에서는 bf16을 요청하고, 학습 정밀도는 실제로 로딩된 가중치에 맞춥니다. 예를 들어 Colab T4에서는 Unsloth가 Qwen3.5를 float32로 올리므로 혼합 정밀도 없이 학습합니다. 실제 값은 `config.json`의 `model_dtype`, `train_precision`에 기록됩니다.
- AMD·Mac·CPU용 학습 경로는 이 배포본에서 구현하거나 검증하지 않았습니다.

Unsloth는 Qwen3.5에서 Transformers v5 사용을 안내하고, 해당 계열의 4bit QLoRA를 권장하지 않습니다. 처음 커널을 컴파일할 때 시간이 추가될 수 있습니다. [공식 Qwen3.5 파인튜닝 안내](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

### (선택) Windows CMD 예시 · 내 PC의 NVIDIA GPU

압축을 푼 폴더에서 실행합니다. 이미 CUDA 지원 PyTorch와 Unsloth가 준비된 환경이 있으면 그 환경을 사용해도 됩니다.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --check
python train.py
```

`requirements.txt`는 호환 범위를 지정한 목록이며 GPU 통합 실행을 검증한 lockfile이 아닙니다. PyTorch가 CPU 전용으로 설치됐거나 CUDA를 인식하지 못하면 진행하지 않습니다. 그 경우 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS와 GPU에 맞는 명령을 확인하고, [Unsloth 공식 설치 안내](https://unsloth.ai/docs/get-started/install)와 대조합니다. NVIDIA 드라이버와 패키지 설치를 세미나 시작 전에 확인해 주세요.

### GPU를 배정받을 별도 환경이 필요한 경우

기본 경로는 위의 Colab 노트북입니다. **무료 GPU 할당과 사용 시간은 보장되지 않습니다.** 코랩 사용이나 유료 결제는 이 자료의 필수 절차가 아닙니다. [Colab 자원 제한 안내](https://research.google.com/colaboratory/faq.html)

## 3. 학생이 바꿀 세 값

```python
LEARNING_RATE = 2e-4
EPOCHS = 2
LORA_RANK = 8
```

| 변수 | 의미 | 예시 비교값 |
| --- | --- | --- |
| LEARNING_RATE | 한 업데이트에서 가중치를 바꾸는 크기 | 5e-5 / 1e-4 / 2e-4 |
| EPOCHS | 학습 데이터를 반복해서 보는 횟수 | 1 / 2 / 4 |
| LORA_RANK | 추가 학습 행렬의 rank | 4 / 8 / 16 |

파일을 고치지 않고 실행 옵션으로 바꿀 수도 있습니다. 옵션을 주지 않은 값은 파일 상단 값을 그대로 씁니다. Colab에서는 앞에 `!`를 붙입니다.

```bat
python train.py --lr 1e-4
python train.py --epochs 4
python train.py --rank 16
```

예시값은 교육용 비교 후보이며 최적값이 아닙니다. **한 번에 한 값만 바꾸고 `python train.py`를 새로 실행**하세요. 실제로 사용한 값은 실행 첫 줄과 `config.json`에 기록됩니다. 매번 원본 모델에서 새 어댑터를 만들므로 이전 실험의 학습이 누적되지 않습니다. alpha는 `2 × rank`로 두어 alpha/r 비율을 고정했습니다. seed, 데이터 분할, 배치, 길이, 생성 설정은 고정합니다.

`max_steps`를 따로 지정하지 않으므로 EPOCHS를 바꾸면 실제 학습량이 달라집니다. 데이터는 학습 36개로 작게 준비했습니다. 이 분량은 실습 흐름을 관찰하기 위한 것이며 충분한 서비스 품질을 보장하지 않습니다.

## 4. 답변 형식과 학습 데이터

답이 문서에 있을 때:

```json
{"answer":"화요일","evidence":["T01"],"status":"answered"}
```

문서에서 확인할 수 없을 때:

```json
{"answer":null,"evidence":[],"status":"insufficient_information"}
```

`answer`에는 질문에 필요한 짧은 값만 넣습니다. `evidence`는 문서 ID의 배열입니다. `status`는 `answered` 또는 `insufficient_information`만 사용합니다. 설명문이나 Markdown 코드블록을 덧붙이면 엄격한 JSON 형식 점수는 통과하지 못합니다.

각 JSONL 예제에는 `question`, `document`, `context_ids`, `expected`, `accepted_answers`가 들어 있습니다. `expected`는 학습 정답이며 `accepted_answers`는 평가에서 인정할 표현을 명시한 목록입니다. `answerable=false` 예제는 관련 주제의 문서가 있어도 질문에 필요한 사실이 없는 경우입니다.

학습에는 문서+질문을 입력으로 주고 정답 JSON 부분에만 loss를 계산합니다. 토큰화 후 길이가 한도를 넘으면 내용을 조용히 잘라내지 않고 중단합니다.

## 5. 데이터 분할과 평가 순서

| 파일 | 개수 | 사용 목적 |
| --- | --- | --- |
| train.jsonl | 36 | LoRA 가중치 학습. 문서 T01~T12만 사용 |
| validation.jsonl | 10 | 학습 중 validation loss 및 설정 비교. 문서 V01~V04 |
| test.jsonl | 10 | 설정을 확정한 뒤 최종 평가. 문서 F01~F04 |
| knowledge.json | 문서 20개 | 예제에 제공할 문서 원문과 데이터 일치 여부 확인 |

문서와 질문 그룹이 분할을 넘지 않도록 구성했습니다. 같은 문서를 표현만 바꾼 질문이 train과 test에 동시에 들어가지 않습니다. 스크립트는 중복 질문·ID, 그룹 누수, 문서 분할, 정답 형식을 검사합니다.

`knowledge.json`에는 평가 문서도 들어 있습니다. **평가할 때는 각 예제에 미리 정해 둔 문서를 질문과 함께 제공합니다.** 이 평가 문서와 정답은 LoRA 가중치 학습에 넣지 않습니다. `document`와 `context_ids`로 입력 문서를 정하며, 정답의 `evidence` 필드로 문서를 고르지 않습니다.

기본 실행은 validation 결과로 설정을 비교합니다.

```bat
python train.py
```

실험 설정을 고른 뒤 최종 평가는 한 번 진행합니다. 이 명령도 선택한 설정으로 원본 모델에서 새로 학습합니다.

```bat
python train.py --final-test
```

최종 실행에서도 학습 전의 `base`와 학습 후의 `lora`에 같은 test 질문과 문서를 제공합니다. test 결과를 본 뒤 그 결과에 맞춰 반복적으로 값을 고르면 더 이상 독립적인 최종 평가가 아닙니다. 별도의 새 test가 필요합니다.

## 6. 결과 읽기

실행마다 `outputs/run-날짜-시간/` 폴더를 새로 만듭니다.

- `comparison.json`, `comparison.csv`: 학습 전후의 **실제 생성 문장**, 질문, 정답, 제공 문서 ID(`context_ids`), 개별 평가 결과. `mode`가 `base`이면 학습 전, `lora`이면 학습 후입니다.
- `summary.json`: `base`와 `lora` 각각의 형식 준수율, 정답 일치율, 근거 일치율, 근거를 갖춘 성공률, 제공 문서에 따른 응답 정책 준수율.
- `loss_history.json`: 학습 loss와 validation loss 등 Trainer 기록.
- `training_metrics.json`: 실제 학습 시간과 학습 기록.
- `contexts.json`: 학습 전후 평가에 공통으로 제공한 문서 ID와 순서.
- `config.json`: 세 실험 변수, 패키지 버전, GPU, 정밀도, 데이터 해시.
- `adapter/`: 학습한 LoRA 어댑터와 토크나이저. 원본 2B 모델 전체를 복제한 폴더가 아닙니다.
- `run_status.json`: `running`, `completed`, `failed` 상태. 일부 결과가 있다고 전체 실행 성공은 아닙니다.

### 지표의 의미

| 지표 | 무엇을 확인하는가 |
| --- | --- |
| format_ok | JSON 하나이며 answer/evidence/status의 형식과 필드가 맞는가 |
| answer_correct | 명시적 정답 또는 허용 표현과 일치하는가 |
| source_correct | 요구되는 근거 문서 ID가 정확한가 |
| grounded_success | 형식·정답·근거·status가 모두 맞고, 답할 근거를 실제로 제공받았는가 |
| context_policy_ok | 제공된 문서에 근거가 있으면 정확히 답하고, 없으면 답변을 보류하는가 |

**학습 전후 모두 문서를 받지만, 모든 질문의 답이 문서에 있는 것은 아닙니다.** 답할 근거가 없는 질문에서 추측하지 않고 정해진 정보 부족 응답을 반환하면 `context_policy_ok`를 통과할 수 있습니다. 같은 문서와 질문을 기준으로 답의 내용, 형식, 근거 활용이 각각 어떻게 달라졌는지 확인하세요.

loss는 정답 토큰을 예측하는 학습 지표입니다. 질문에 대한 사실 정확도와 같은 값이 아닙니다. loss가 낮아져도 형식이나 근거 점수가 좋아지지 않을 수 있습니다. 정답 표현 평가는 짧은 문자열 일치 방식이므로 의미가 같아도 허용 목록에 없는 표현은 틀렸다고 셀 수 있습니다. CSV의 원문도 함께 읽으세요. 10개 평가 사례의 결과를 일반적인 모델 성능으로 확대 해석하지 마세요.

이 자료는 결과를 임의로 채우거나 학습 성공을 시뮬레이션하지 않습니다. 실행 중단 전까지 얻은 실제 출력만 남습니다. 추가 명령 없이 외부 서비스에 결과나 어댑터를 업로드하지 않습니다.

## 7. 사전 확인 명령

```bat
python train.py --help
python train.py --check-data
python train.py --check
```

`--check-data`는 Python 표준 라이브러리만 사용합니다. `--check`는 데이터와 CUDA GPU/메모리를 확인하며 모델을 다운로드하지 않습니다. 이 점검 통과는 모델 로딩·학습 성공을 증명하지 않으므로 실제 전체 실행이 한 번 더 필요합니다.

처음 설치한 환경에서 CUDA GPU가 잡히지 않으면 명확한 오류와 종료 코드 1을 반환합니다. 메모리 사전 점검은 안내만 표시하며, 실제 GPU 메모리 부족으로 중간에 멈추면 종료 코드 1과 실패 상태를 남기고 이미 생성된 출력은 보존합니다. 학생들이 동일한 데이터와 생성 설정을 쓰는지, 실행이 `completed`인지 확인한 뒤 비교하세요.

## 8. 공식 참고 자료

- [Unsloth Qwen3.5 텍스트 SFT·16bit LoRA 예제](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Qwen3.5-2B 모델 카드와 non-thinking 설정](https://huggingface.co/Qwen/Qwen3.5-2B)
- [TRL SFTTrainer: prompt-completion과 completion-only loss](https://huggingface.co/docs/trl/en/sft_trainer)
- [Hugging Face PEFT LoRA 개념](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
- [Unsloth 설치 안내](https://unsloth.ai/docs/get-started/install)

작성 시 확인한 공식 예제에 맞추되, 모델·라이브러리가 갱신되면 API나 지원 환경이 바뀔 수 있습니다. 이 배포본은 버전 정보를 실행 결과에 기록합니다.
