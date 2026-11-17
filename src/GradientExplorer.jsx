import React, { useId, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import './gradient-explorer.css';

const LEARNING_RATE = 0.1;
const lossAt = weight => (weight - 3) ** 2;
const gradientAt = weight => 2 * (weight - 3);
const updateWeight = weight => weight - LEARNING_RATE * gradientAt(weight);
const clampWeight = weight => Math.max(0, Math.min(6, weight));
const xAt = weight => 48 + weight / 6 * 504;
const yAt = loss => 162 - loss / 9 * 132;
const curve = Array.from({ length: 121 }, (_, i) => {
  const weight = i / 20;
  return `${i === 0 ? 'M' : 'L'}${xAt(weight).toFixed(2)},${yAt(lossAt(weight)).toFixed(2)}`;
}).join(' ');

function number(value, minimumDecimals = 0) {
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
  return value.toLocaleString('en-US', { minimumFractionDigits: minimumDecimals, maximumFractionDigits: 4 });
}

const weightLabel = value => value !== 3 && Math.abs(value - 3) < 0.0001 ? '≈ 3' : number(value);

export default function GradientExplorer() {
  const [weight, setWeight] = useState(3.2);
  const id = useId();
  const loss = lossAt(weight);
  const gradient = gradientAt(weight);
  const nextWeight = updateWeight(weight);
  const atMinimum = gradient === 0;
  const almostMinimum = !atMinimum && Math.abs(weight - 3) < 0.0001;
  const direction = atMinimum
    ? '기울기가 0입니다. w = 3에서 Loss가 가장 작습니다.'
    : almostMinimum
      ? '기울기가 거의 0입니다. 최솟값에 가까워졌습니다.'
      : gradient > 0
        ? '기울기가 양수입니다. w를 줄이는 방향으로 움직이면 Loss를 줄일 수 있습니다.'
        : '기울기가 음수입니다. w를 늘리는 방향으로 움직이면 Loss를 줄일 수 있습니다.';

  function handleWeightKey(event) {
    const moves = { ArrowLeft: -0.1, ArrowDown: -0.1, ArrowRight: 0.1, ArrowUp: 0.1, PageDown: -0.5, PageUp: 0.5 };
    if (!(event.key in moves) && event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault();
    event.stopPropagation();
    setWeight(current => event.key === 'Home' ? 0 : event.key === 'End' ? 6 : clampWeight(Number((current + moves[event.key]).toFixed(2))));
  }

  return <section className="gradient-explorer" aria-label="가중치를 직접 바꿔 보는 손실과 기울기 실험">
    <div className="gradient-main-grid">
      <section className="gradient-chart-panel">
        <div className="gradient-chart-heading"><h3>Loss = (w − 3)²</h3><span>w = 3에서 최솟값</span></div>
        <label htmlFor={`${id}-weight`}>현재 가중치 w <strong>{weightLabel(weight)}</strong></label>
        <input id={`${id}-weight`} type="range" min="0" max="6" step="any" value={weight}
          aria-valuetext={weightLabel(weight)} onKeyDown={handleWeightKey}
          onChange={event => setWeight(Number(Number(event.target.value).toFixed(2)))}/>
        <svg className="gradient-chart" viewBox="0 0 600 200" role="img" aria-labelledby={`${id}-title ${id}-description`}>
          <title id={`${id}-title`}>L = (w − 3)²의 그래프와 현재 위치</title>
          <desc id={`${id}-description`}>가중치 {weightLabel(weight)}, 손실 {number(loss, 2)}, 기울기 {number(gradient, 2)}. 최솟값은 w가 3일 때 손실 0입니다.</desc>
          <path className="gradient-axis" d="M48,25 V162 H552"/>
          <text x="48" y="17" className="gradient-axis-label">Loss</text>
          <text x="31" y="35" textAnchor="end">9</text>
          <text x="31" y="166" textAnchor="end">0</text>
          {[0, 3, 6].map(tick => <g key={tick}>
            <line className="gradient-tick" x1={xAt(tick)} x2={xAt(tick)} y1="162" y2="167"/>
            <text x={xAt(tick)} y="190" textAnchor="middle">{tick === 3 ? '3 (최솟값)' : tick}</text>
          </g>)}
          <text x="578" y="190" textAnchor="middle">w</text>
          <path className="gradient-curve" d={curve}/>
          <line className="gradient-guide" x1={xAt(weight)} x2={xAt(weight)} y1={yAt(loss)} y2="162"/>
          <circle className="gradient-dot" cx={xAt(weight)} cy={yAt(loss)} r="8"/>
        </svg>
      </section>
      <section className="gradient-control-panel" aria-label="기울기로 한 번 업데이트하기">
        <div className="gradient-readouts" aria-live="polite">
          <div><span>현재 Loss</span><strong>{number(loss, 2)}</strong></div>
          <div><span>기울기 dL/dw</span><strong>{number(gradient, 2)}</strong></div>
        </div>
        <p className="gradient-direction" aria-live="polite">{direction}</p>
        <div className="gradient-update-rule">
          <p>학습률 <strong>0.1</strong>로 한 번 업데이트하면</p>
          <div className="gradient-equation" role="math" aria-label="새 가중치는 현재 가중치에서 0.1 곱하기 기울기를 뺀 값입니다.">
            <span><i>w</i><sub>new</sub></span> = <i>w</i> − 0.1 × <span className="gradient-fraction"><span>dL</span><span>dw</span></span>
          </div>
          <div className="gradient-next" aria-live="polite"><span>업데이트 후 가중치</span><strong>{weightLabel(nextWeight)}</strong></div>
        </div>
        <div className="gradient-actions">
          <button className="gradient-step-button" type="button" disabled={atMinimum || almostMinimum} onClick={() => setWeight(updateWeight)}>한 번 학습시키기</button>
          <button className="gradient-reset" type="button" onClick={() => setWeight(3.2)} aria-label="가중치를 초기값 3.2로 되돌리기"><RotateCcw size={15} aria-hidden="true"/>초기값</button>
        </div>
      </section>
    </div>
    <ol className="gradient-role-strip">
      <li><strong>순전파</strong><span>현재 값으로 예측·Loss 계산</span></li>
      <li><strong>역전파</strong><span>Loss의 기울기 계산</span></li>
      <li><strong>옵티마이저</strong><span>가중치를 실제로 업데이트</span></li>
    </ol>
    <p className="gradient-caveat">가중치 하나의 단순한 경사하강법 예시입니다. 실제 LoRA는 원본 W를 고정하고 A·B를 학습합니다.</p>
  </section>;
}
