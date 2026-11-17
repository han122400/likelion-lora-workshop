import React, { useId, useRef, useState } from 'react';
import { ArrowRight, LockKeyhole, RotateCcw } from 'lucide-react';
import './training-example.css';

const steps = [
  ['데이터 준비', '질문과 모범 답변'],
  ['다음 토큰 예측', '정답에 준 확률 확인'],
  ['Loss 계산', '정답 확률로 손실 계산'],
  ['LoRA 업데이트', '기울기로 A·B 조정'],
  ['어댑터 저장', '반복 학습 후 저장'],
];

function StepDetail({ step }) {
  switch (step) {
    case 0: return <>
      <p>질문과 정답 예시를 모델에 제공합니다. SFT에서는 모범 답변이 학습 목표가 됩니다.</p>
      <div className="seoul-data-pair"><span>대한민국의 수도는?</span><ArrowRight aria-hidden="true"/><strong>서울입니다.</strong></div>
      <p className="seoul-detail-note">이런 입력·정답 쌍을 여러 개 준비해 데이터셋을 만듭니다.</p>
    </>;
    case 1: return <>
      <p>정답 문장의 첫 토큰을 예측한다고 가정해 봅시다.</p>
      <ul className="seoul-probabilities" aria-label="설명을 위한 가상 예측 확률">
        {[['서울', 20], ['부산', 50], ['인천', 30]].map(([city, probability]) => <li key={city} className={city === '서울' ? 'seoul-correct' : ''}>
          <span>{city}{city === '서울' && <small>정답</small>}</span>
          <span className="seoul-probability-track" aria-hidden="true"><i style={{ width: `${probability}%` }}/></span>
          <strong>{probability}%</strong>
        </li>)}
      </ul>
      <p className="seoul-detail-note">정답인 ‘서울’에 준 확률은 20%입니다. 도시 이름을 토큰 하나로 단순화한 가상 예시이며, 실제 토큰 분할과 확률은 다릅니다.</p>
    </>;
    case 2: return <>
      <p>정답 토큰에 얼마나 낮은 확률을 줬는지 손실로 계산합니다.</p>
      <div className="seoul-loss-formula" role="math" aria-label="손실 L은 정답 토큰 확률의 자연로그에 마이너스를 붙인 값입니다.">L = −ln P(정답 토큰)</div>
      <p className="seoul-loss-example">앞선 20% 예시: <strong>−ln(0.20) ≈ 1.609</strong></p>
      <p className="seoul-detail-note">정답 확률이 낮으면 Loss가 커지고, 높으면 작아집니다. 실제로는 여러 정답 토큰의 손실을 합산하거나 평균 냅니다.</p>
    </>;
    case 3: return <>
      <p>Loss를 줄이려면 A와 B를 어떻게 바꿔야 할까요?</p>
      <ol className="seoul-update-actions">
        <li><strong>역전파</strong><span>A·B에 대한 기울기(Gradient) 계산</span></li>
        <li><strong>AdamW 등 옵티마이저</strong><span>기울기를 사용해 A·B 업데이트</span></li>
      </ol>
      <p className="seoul-detail-note">정답 ‘서울’의 확률을 높이는 방향으로 학습합니다. 단, 업데이트할 때마다 반드시 개선되는 것은 아닙니다.</p>
    </>;
    case 4: return <>
      <div className="seoul-repeat"><RotateCcw aria-hidden="true"/><strong>② 예측 → ③ Loss → ④ 업데이트 반복</strong></div>
      <p>데이터셋에 걸쳐 학습을 반복한 뒤, 학습한 A와 B를 LoRA 어댑터로 저장합니다.</p>
      <div className="seoul-save-equation"><span>기본 모델</span><b>+</b><span>학습한 어댑터</span><ArrowRight aria-hidden="true"/><strong>추론에 사용</strong></div>
      <p className="seoul-detail-note">어댑터는 같은 기본 모델에 적용해 사용합니다.</p>
    </>;
    default: return null;
  }
}

export default function TrainingExample() {
  const [step, setStep] = useState(1);
  const [probability, setProbability] = useState(25);
  const id = useId();
  const tabRefs = useRef([]);
  const loss = -Math.log(probability / 100);

  function handleTabKey(event, index) {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % steps.length;
    else if (event.key === 'ArrowLeft') next = (index + steps.length - 1) % steps.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = steps.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    setStep(next);
    tabRefs.current[next]?.focus();
  }

  return <section className="seoul-training" aria-label="대한민국 수도 예시로 배우는 LoRA 학습">
    <div className="seoul-training-data"><strong>학습 데이터</strong><span>질문 <b>대한민국의 수도는?</b></span><ArrowRight aria-hidden="true"/><span>정답 <b>서울입니다.</b></span></div>
    <div className="seoul-step-tabs" role="tablist" aria-label="학습 단계 자세히 보기">
      {steps.map(([title, summary], index) => <button key={title} type="button" role="tab"
        id={`${id}-step-${index}`} aria-selected={step === index} aria-controls={`${id}-detail`}
        tabIndex={step === index ? 0 : -1} ref={element => { tabRefs.current[index] = element; }}
        onClick={() => setStep(index)} onKeyDown={event => handleTabKey(event, index)}>
        <span className="seoul-step-title"><b>{index + 1}</b>{title}</span><small>{summary}</small>
      </button>)}
    </div>
    <div className="seoul-example-grid">
      <section className="seoul-step-detail" role="tabpanel" id={`${id}-detail`} aria-labelledby={`${id}-step-${step}`} tabIndex={0}>
        <div className="seoul-detail-heading"><h3>{step + 1}. {steps[step][0]}</h3><span>위 단계를 눌러 확인하세요</span></div>
        <StepDetail step={step}/>
      </section>
      <section className="seoul-loss-control" aria-label="정답 확률과 손실을 직접 조작하는 별도 계산 예시">
        <h3>직접 조작해 보기 <small>별도 계산 예시</small></h3>
        <label htmlFor={`${id}-probability`}>정답 ‘서울’의 예측 확률 <strong>{probability}%</strong></label>
        <input id={`${id}-probability`} type="range" min="1" max="99" step="1" value={probability}
          aria-valuetext={`${probability}%`} onChange={event => setProbability(Number(event.target.value))}/>
        <div className="seoul-loss-readout" aria-live="polite"><span>Cross-entropy Loss</span><strong>{loss.toFixed(3)}</strong></div>
        <p className="seoul-learning-goal"><span>학습 목표</span><strong>정답 확률 높이기</strong></p>
        <p className="seoul-slider-note">슬라이더는 손실 계산용입니다. 실제 학습에서는 확률을 직접 조절하지 않고, 가중치를 업데이트한 결과로 확률이 변합니다.</p>
      </section>
    </div>
    <p className="seoul-frozen"><LockKeyhole size={16} aria-hidden="true"/><span>모든 단계에서 <strong>원본 W는 고정</strong> · <strong>A와 B만 학습</strong></span></p>
  </section>;
}
