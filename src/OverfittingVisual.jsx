import React, { useId } from 'react';
import { ArrowRight, TrendingDown, TrendingUp } from 'lucide-react';
import './overfitting-visual.css';

function LossSketch({ compact, id }) {
  const width = compact ? 420 : 680;
  const end = compact ? 275 : 490;
  const turn = compact ? 160 : 270;
  const start = 48;
  const trainPath = compact
    ? 'M48 52 C78 112 112 154 160 184 S234 221 275 230'
    : 'M48 52 C112 116 190 160 270 184 S418 223 490 230';
  const validationPath = compact
    ? 'M48 62 C85 123 118 165 160 166 S225 132 275 72'
    : 'M48 62 C120 125 200 167 270 166 S404 128 490 72';

  return (
    <svg
      className={`overfit-chart ${compact ? 'overfit-chart-mobile' : 'overfit-chart-wide'}`}
      viewBox={`0 0 ${width} 290`}
      role="img"
      aria-labelledby={`${id}-title ${id}-desc`}
    >
      <title id={`${id}-title`}>학습 loss와 검증 loss가 다르게 움직이는 과적합의 예</title>
      <desc id={`${id}-desc`}>
        가로축은 학습 진행, 세로축은 loss입니다. 학습 loss는 계속 낮아지지만
        검증 loss는 낮아졌다가 다시 높아집니다. 과적합을 의심할 수 있는 개념도이며
        실제 측정값이나 최적 학습 횟수를 나타내지 않습니다.
      </desc>
      <rect className="overfit-caution-area" x={turn} y="25" width={end - turn + 10} height="220" rx="8" />
      <path className="overfit-axis" d={`M${start} 22 V249 H${end + 14}`} />
      <path className="overfit-axis-tip" d={`M${start - 4} 29 L${start} 21 L${start + 4} 29 M${end + 7} 245 L${end + 15} 249 L${end + 7} 253`} />
      <text className="overfit-axis-label" x="12" y="18">loss</text>
      <text className="overfit-axis-label" x={end + 8} y="279" textAnchor="end">학습 진행</text>
      <text className="overfit-region-label" x={turn + 12} y="47">과적합 의심</text>
      <path className="overfit-train-line" d={trainPath} />
      <path className="overfit-validation-line" d={validationPath} />
      <circle className="overfit-validation-dot" cx={end} cy="72" r="4" />
      <text className="overfit-direct-label overfit-validation-label" x={end + 12} y="78">검증 loss ↑</text>
      <text className="overfit-direct-label overfit-train-label" x={end + 12} y="236">학습 loss ↓</text>
    </svg>
  );
}

export default function OverfittingVisual() {
  const id = useId();
  return (
    <figure className="overfit-visual">
      <div className="overfit-body">
        <div className="overfit-plot">
          <LossSketch id={`${id}-wide`} compact={false} />
          <LossSketch id={`${id}-mobile`} compact />
        </div>
        <div className="overfit-explanation">
          <div className="overfit-observation">
            <TrendingDown aria-hidden="true" />
            <div><h3>학습 loss는 계속 감소</h3><p>학습 예시의 정답 토큰을<br />점점 더 잘 예측합니다.</p></div>
          </div>
          <div className="overfit-observation overfit-warning">
            <TrendingUp aria-hidden="true" />
            <div><h3>검증 loss는 다시 증가</h3><p>학습에 없던 예시에서는<br />예측이 나빠질 수 있습니다.</p></div>
          </div>
          <p className="overfit-action"><ArrowRight aria-hidden="true" /><span>EPOCHS를 줄여 다시 비교</span></p>
        </div>
      </div>
      <figcaption className="overfit-caption">개념 설명용 예시 · 실제 학습 결과 아님<span>loss와 실제 답변을 함께 확인하세요.</span></figcaption>
    </figure>
  );
}
