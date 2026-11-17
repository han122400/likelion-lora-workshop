import lesson from '../lesson-content.json';
import guide from '../workshop-assets/guide.json';

export const slides = lesson.slides;
// 2단계·4단계에 똑같이 넣는 웹 가이드 입력(노트북의 guide.json과 같은 파일)
export const guideInputs = guide.inputs;
export const chapterNames = ['RAG 복습과 학습 목표', 'LoRA 이해하기', '개념 비교와 활용', '학습의 원리', '데이터 준비', 'LoRA 실습과 비교'];
export const sources = [
  { name: 'LoRA: Low-Rank Adaptation', detail: '원본 논문 · 저랭크 가중치 업데이트', url: 'https://arxiv.org/abs/2106.09685', chapters: [2, 3, 4] },
  { name: 'QLoRA: Efficient Finetuning', detail: '원본 논문 · 양자화한 기반 모델과 LoRA', url: 'https://arxiv.org/abs/2305.14314', chapters: [2] },
  { name: 'Hugging Face · PEFT LoRA', detail: 'LoRA 개념과 어댑터', url: 'https://huggingface.co/docs/peft/en/conceptual_guides/lora', chapters: [2, 3, 4] },
  { name: 'Hugging Face · RAG Cookbook', detail: '검색 문맥과 생성 모델 연결', url: 'https://huggingface.co/learn/cookbook/advanced_rag', chapters: [1, 3] },
  { name: 'Unsloth · Qwen3.5 학습 가이드', detail: '16bit LoRA · 설치와 모델 설정', url: 'https://unsloth.ai/docs/models/qwen3.5/fine-tune', chapters: [2, 6] },
  { name: 'Qwen3.5-2B 모델 카드', detail: '모델 정보 · 텍스트 입력 · no-thinking', url: 'https://huggingface.co/Qwen/Qwen3.5-2B', chapters: [6] },
  { name: 'Hugging Face · SFTTrainer', detail: '지도 미세조정 · completion-only loss', url: 'https://huggingface.co/docs/trl/en/sft_trainer', chapters: [4, 5, 6] },
  { name: 'PyTorch · AdamW', detail: '기울기 기반 업데이트와 weight decay', url: 'https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html', chapters: [4] },
  { name: 'PyTorch · SGD', detail: '기울기와 학습률로 가중치를 업데이트하는 원리', url: 'https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html', chapters: [4] },
  { name: 'SciPy · 뉴턴법과 Hessian', detail: '기울기·곡률을 이용한 업데이트와 Hessian-vector 곱', url: 'https://docs.scipy.org/doc/scipy/tutorial/optimize.html#newton-conjugate-gradient-algorithm-method-newton-cg', chapters: [4] },
  { name: 'PyTorch · Hessian', detail: '선택한 입력에 대한 2차 미분 행렬', url: 'https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html', chapters: [4] },
  { name: 'Optuna · 첫 번째 실험', detail: '선택 심화 · 하이퍼파라미터 탐색', url: 'https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html', chapters: [4] },
  { name: 'Optuna · 설정을 제안하는 방법', detail: 'sampler 종류와 이전 실험 결과의 활용', url: 'https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html', chapters: [4] },
];

export const setupCommand = 'python -m venv .venv\n.venv\\Scripts\\activate\npython -m pip install --upgrade pip\npython -m pip install -r requirements.txt';
export const configCode = 'LEARNING_RATE = 2e-4\nEPOCHS = 2\nLORA_RANK = 8';
export const outputExample = '{\n  "answer": "화요일",\n  "evidence": ["T01"],\n  "status": "answered"\n}';

export const questions = {
  1: '문서를 한 번 읽히는 것과 모델을 학습시키는 것은 어떻게 다를까요?',
  2: 'LoRA에서 고정되는 값과 업데이트되는 값은 무엇인가요?',
  3: '정확한 문서를 줘도 출력 형식을 계속 어긴다면, 어디를 개선할까요?',
  4: '학습 loss는 낮아졌는데 새 질문의 답이 틀렸다면 무엇을 확인할까요?',
  5: 'AI가 만들어 준 정답을 그대로 학습 데이터로 써도 될까요?',
  6: '같은 입력에 대한 답변이 LoRA 학습 전후에 어떻게 달라졌나요?',
};
