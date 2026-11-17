# LIKELION · LoRA 웹 워크숍

발표와 개인 학습을 함께 지원하는 한국어 24장 웹 슬라이드입니다. 지난 시간에 다룬 RAG와는 개념만 비교하고, 이번 실습에서는 LoRA 학습 전후를 살펴봅니다.

## 개발

```sh
npm ci
npm run dev
npm run build
```

React + Vite. `dist/`는 정적 파일이므로 Vercel에서 그대로 제공할 수 있습니다. 서버·환경 변수·유료 모델 API는 사용하지 않습니다.

## 수정할 곳

- `lesson-content.json`: 슬라이드 제목, 본문, 표, 학생용 해설 `learnerNote`, 발표자 메모 `speakerNote`
- `src/SlideStage.jsx`: 표지와 개념 시각화
- `src/styles.css`: 레이아웃, 색상, 반응형/발표 스타일
- `src/data.js`: 목차와 공식 참고 자료
- `workshop-assets/`: 실습 원본. `lora_workshop.ipynb`(6단계 Colab 노트북), `train.py`(학습·평가 함수와 명령줄 실행), `lab_view.py`(카드·표·그래프), `guide.json`(2·4단계 가이드 입력, 웹 슬라이드도 같은 파일을 읽음), 데이터 파일
- `public/downloads/`: 사이트에서 다운로드하는 실습 ZIP, 노트북, README

실습 원본을 바꿨다면 `python scripts/pack_workshop.py`로 다운로드 파일을 다시 만든 뒤 `npm run build` 하세요. 테스트는 `python -m pytest tests`(GPU 없이 가짜 모델로 노트북 흐름까지 확인)입니다. 이전 데이터로 한 명령줄 전체 실행은 Colab T4에서 확인했지만, 현재 노트북의 단계별 실행은 수업 전 진행자가 Colab에서 한 번 실행해 확인해야 합니다.

## 이용 방법

- 학습 모드: 챕터 목차, 현재 페이지 번호를 눌러 전체 슬라이드 선택, 설명과 공식 출처, 학습 완료 체크, 코드 복사.
- 발표 모드: 방향키 또는 Space로 넘기기, Home/End로 처음/끝, F 전체 화면, N 발표자 메모. Esc는 브라우저 전체 화면을 먼저 종료하며 일반 발표 화면에서는 학습 모드로 돌아갑니다.
- `?present=1#s07`처럼 모드와 슬라이드를 URL로 지정할 수 있습니다.
- 학습 완료 상태만 현재 브라우저의 localStorage에 저장합니다. 계정·서버 저장소는 없습니다.
- 사이트에서 모델을 학습하지 않습니다. 제공된 노트북을 Colab(T4 GPU)에서 실행합니다.

## 실험

Qwen3.5-2B에 같은 질문과 미리 준비한 문서를 제공하고, LoRA 학습 전후의 답변을 비교합니다. 실행 중 문서를 검색하거나 RAG 적용 여부를 비교하지 않습니다. 한 변수를 바꾸며 validation 결과를 비교하고, 설정을 고른 뒤 test로 최종 평가합니다.

실습 데이터는 멋사 중부대 14기 운영 안내입니다. 정규 세미나 시간과 운영진 정보는 실제 14기 정보이고, 나머지 규정·일정·금액은 교육용 예시입니다. 하드웨어 조건과 설치는 `workshop-assets/README.ko.md`에서 확인하세요.
