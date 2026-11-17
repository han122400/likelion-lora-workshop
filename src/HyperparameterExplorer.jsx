import React, { useId, useState } from 'react';
import './hyperparameter-explorer.css';

const rates = [
  { value: 0.01, label: '0.01', description: '작게' },
  { value: 0.1, label: '0.1', description: '중간 예시' },
  { value: 1, label: '1.0', description: '크게' },
];
const formatCount = value => value.toLocaleString('en-US');

export default function HyperparameterExplorer() {
  const [learningRate, setLearningRate] = useState(1);
  const [epochs, setEpochs] = useState(4);
  const [rank, setRank] = useState(11);
  const id = useId();
  const change = learningRate * 2;
  const updatedWeight = 5 - change;
  const rateLabel = rates.find(rate => rate.value === learningRate).label;

  return <section className="hp-explorer" aria-label="학습률, 에포크, LoRA rank를 직접 바꾸는 개념 예시">
    <p className="hp-example-label">개념 설명용 예시 · 학습률 계산은 단순 경사하강법(SGD)이며, AdamW의 실제 업데이트 식과 다릅니다.</p>
    <div className="hp-cards">
      <section className="hp-card hp-rate" aria-labelledby={`${id}-rate-title`}>
        <h3 id={`${id}-rate-title`}>Learning Rate <span>학습률</span></h3>
        <p className="hp-question">가중치를 얼마나 크게 수정할까?</p>
        <p className="hp-rate-given"><span>앞 그래프와 별도 예시</span><strong>w = 5 · g = 2</strong></p>
        <fieldset className="hp-rate-options">
          <legend>학습률 선택</legend>
          {rates.map(rate => <label key={rate.value}>
            <input type="radio" name={`${id}-rate`} value={rate.value} checked={learningRate === rate.value} onChange={() => setLearningRate(rate.value)}/>
            <span><strong>{rate.label}</strong><small>{rate.description}</small></span>
          </label>)}
        </fieldset>
        <div className="hp-rate-equation" role="math" aria-label={`새 가중치는 5 빼기 ${rateLabel} 곱하기 2, ${updatedWeight.toFixed(2)}입니다.`}>
          5 − <strong>{rateLabel}</strong> × 2 = <strong>{updatedWeight.toFixed(2)}</strong>
        </div>
        <div className="hp-two-readouts" aria-live="polite">
          <div><span>수정 크기</span><strong>{change.toFixed(2)}</strong></div>
          <div><span>새 가중치</span><strong>{updatedWeight.toFixed(2)}</strong></div>
        </div>
        <p className="hp-card-note">너무 작으면 느리고, 너무 크면 불안정할 수 있습니다. <strong>실제 학습률 설정값이 아닌 계산 예시입니다.</strong></p>
      </section>

      <section className="hp-card hp-epochs" aria-labelledby={`${id}-epochs-title`}>
        <h3 id={`${id}-epochs-title`}>Epochs <span>반복 횟수</span></h3>
        <p className="hp-question">전체 데이터를 몇 번 반복해서 볼까?</p>
        <label className="hp-slider-label" htmlFor={`${id}-epochs`}>전체 데이터 <strong>1,000개</strong><span><b>{epochs}</b> epochs</span></label>
        <input className="hp-slider" id={`${id}-epochs`} type="range" min="1" max="6" step="1" value={epochs} aria-valuetext={`전체 데이터를 ${epochs}번 반복`} onChange={event => setEpochs(Number(event.target.value))}/>
        <ol className="hp-epoch-passes" aria-label={`1,000개 데이터를 ${epochs}번 반복`}>
          {Array.from({ length: epochs }, (_, index) => <li key={index}><span>{index + 1}회차</span><strong>1,000개</strong></li>)}
        </ol>
        <div className="hp-total" aria-live="polite"><span>샘플 기준 학습 횟수</span><strong>{formatCount(epochs * 1000)}<small>회</small></strong></div>
        <p className="hp-count-note">서로 다른 데이터 수나 업데이트 횟수가 아닙니다.</p>
        <p className="hp-card-note">적게 반복하면 과소적합, 많이 반복하면 과적합이 생길 수 있습니다. <strong>학습·검증 Loss를 함께 확인합니다.</strong></p>
      </section>

      <section className="hp-card hp-rank" aria-labelledby={`${id}-rank-title`}>
        <h3 id={`${id}-rank-title`}>LoRA Rank <span>중간 차원</span></h3>
        <p className="hp-question">LoRA의 중간 차원을 얼마나 둘까?</p>
        <label className="hp-slider-label" htmlFor={`${id}-rank`}>Rank <strong>r</strong><span><b>{rank}</b></span></label>
        <input className="hp-slider" id={`${id}-rank`} type="range" min="1" max="32" step="1" value={rank} aria-valuetext={`중간 차원 ${rank}`} onChange={event => setRank(Number(event.target.value))}/>
        <div className="hp-matrix-dimensions" aria-label={`B는 1000 곱하기 ${rank}, A는 ${rank} 곱하기 1000 크기입니다.`}>
          <div><span className="hp-matrix-shape hp-matrix-b" style={{ width: `${8 + rank * 0.55}px` }} aria-hidden="true"/><p>B <span>1,000 × <strong>{rank}</strong></span></p></div>
          <span className="hp-matrix-times" aria-hidden="true">×</span>
          <div><span className="hp-matrix-shape hp-matrix-a" style={{ height: `${8 + rank * 0.55}px` }} aria-hidden="true"/><p>A <span><strong>{rank}</strong> × 1,000</span></p></div>
        </div>
        <div className="hp-rank-parameters" aria-live="polite"><span>학습할 A·B 파라미터<br/>2,000 × r</span><strong>{formatCount(2000 * rank)}<small>개</small></strong></div>
        <p className="hp-count-note">원본 W 한 개: 1,000 × 1,000 = 1,000,000개<br/>전체 Qwen 모델의 파라미터 수가 아닙니다.</p>
        <p className="hp-card-note">r이 크면 표현 용량과 파라미터·메모리·계산량이 늘어납니다. <strong>성능이 더 좋아진다는 보장은 없습니다.</strong></p>
      </section>
    </div>
  </section>;
}
