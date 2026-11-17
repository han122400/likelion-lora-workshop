import React, { useId } from 'react';
import { ArrowRight } from 'lucide-react';
import './newton-hessian-panels.css';

const chartX = weight => 34 + (weight - 1) / 4 * 340;
const chartY = loss => 162 - loss / 4 * 128;
const parabola = Array.from({ length: 81 }, (_, index) => {
  const weight = 1 + index / 20;
  return `${index === 0 ? 'M' : 'L'}${chartX(weight).toFixed(2)},${chartY((weight - 3) ** 2).toFixed(2)}`;
}).join(' ');

function Fraction({ numerator, denominator }) {
  return <span className="nh-fraction"><span>{numerator}</span><span>{denominator}</span></span>;
}

export function NewtonPanel() {
  const id = useId();
  return <section className="nh-panel nh-newton" aria-label="기울기와 곡률로 업데이트하는 뉴턴법 예시">
    <figure className="nh-newton-chart">
      <figcaption><b>L = (w − 3)<sup>2</sup></b><span>설명을 위한 이차함수</span></figcaption>
      <svg viewBox="0 0 420 200" role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>이차함수에서 뉴턴법으로 가중치 5를 3으로 업데이트</title>
        <desc id={`${id}-description`}>손실은 w가 5일 때 4이고, w가 3일 때 최솟값 0입니다. 이 함수에서는 뉴턴법 한 번으로 3에 도달합니다.</desc>
        <defs><marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#a8430b"/></marker></defs>
        <path className="nh-chart-axis" d="M34,25 V162 H390"/>
        <text x="34" y="15" className="nh-chart-axis-label">Loss</text>
        <text x="399" y="167" className="nh-chart-axis-label">w</text>
        <text x="22" y="166" textAnchor="end">0</text>
        <text x="22" y="39" textAnchor="end">4</text>
        {[1, 3, 5].map(weight => <g key={weight}><line className="nh-chart-axis" x1={chartX(weight)} x2={chartX(weight)} y1="162" y2="167"/><text x={chartX(weight)} y="188" textAnchor="middle">{weight === 3 ? '3 · 최솟값' : weight}</text></g>)}
        <path className="nh-chart-curve" d={parabola}/>
        <path className="nh-chart-jump" d="M363,48 L219,148" markerEnd={`url(#${id}-arrow)`}/>
        <circle className="nh-chart-start" cx={chartX(5)} cy={chartY(4)} r="6"/>
        <circle className="nh-chart-end" cx={chartX(3)} cy={chartY(0)} r="7"/>
        <text x="361" y="25" textAnchor="end" className="nh-chart-point-label">시작 w = 5</text>
      </svg>
    </figure>
    <div className="nh-newton-explanation">
      <dl className="nh-derivative-roles">
        <div><dt>1차 미분 L′(w) · 기울기</dt><dd>Loss가 어느 방향으로 변하나?</dd></div>
        <div><dt>2차 미분 L″(w) · 곡률</dt><dd>얼마나 휘어 있나?</dd></div>
      </dl>
      <div className="nh-newton-formula" role="math" aria-label="새 가중치는 현재 가중치에서 손실의 일차 미분을 이차 미분으로 나눈 값을 뺀 것입니다.">
        <span>w<sub>new</sub></span><span>=</span><span>w</span><span>−</span><Fraction numerator="L′(w)" denominator="L″(w)"/>
      </div>
      <div className="nh-newton-calculation">
        <p>w = 5일 때 <strong>기울기 4</strong> · <strong>곡률 2</strong></p>
        <div role="math" aria-label="5 빼기 4 나누기 2는 3입니다."><span>5 −</span><Fraction numerator="4" denominator="2"/><span>=</span><strong>3</strong></div>
      </div>
      <p className="nh-newton-note">기본식은 L″(w) ≠ 0일 때 적용합니다. 이 이차함수에서는 한 번에 도달하지만,<br/>일반적으로 한 번에 최솟값에 도달하거나 Loss가 줄어든다는 보장은 없습니다.</p>
    </div>
  </section>;
}

export function HessianPanel() {
  return <section className="nh-panel nh-hessian" aria-label="여러 가중치의 이차 미분 정보를 모은 Hessian과 행렬의 크기">
    <div className="nh-hessian-grid">
      <div className="nh-hessian-definition">
        <p className="nh-definition">여러 가중치의 <strong>2차 편미분 정보</strong>를 모은 행렬입니다.</p>
        <div className="nh-hessian-equation" role="math" aria-label="가중치 두 개의 Hessian은 이차 편미분 네 개를 담은 2 곱하기 2 행렬입니다.">
          <span>H =</span><div className="nh-hessian-matrix">
            <Fraction numerator="∂²L" denominator="∂w₁²"/>
            <Fraction numerator="∂²L" denominator="∂w₁∂w₂"/>
            <Fraction numerator="∂²L" denominator="∂w₂∂w₁"/>
            <Fraction numerator="∂²L" denominator="∂w₂²"/>
          </div>
        </div>
        <p className="nh-matrix-dimension">미분 대상이 <strong>n개</strong>이면 <strong>n × n</strong> 행렬<br/>뉴턴법에서 여러 가중치의 곡률을 다룰 때 사용합니다.</p>
      </div>
      <div className="nh-hessian-cost">
        <p className="nh-cost-label">미분 대상 파라미터를 10억 개로 잡은 이론 예시</p>
        <div className="nh-cost-comparison"><div><span>파라미터</span><strong>10억 개</strong></div><ArrowRight aria-hidden="true"/><div><span>Hessian 원소</span><strong>10<sup>18</sup>개</strong></div></div>
        <p className="nh-cost-conclusion">전체 행렬을 구성하고 저장하려면<br/><strong>계산·메모리 비용이 큽니다.</strong></p>
        <p className="nh-adamw-note">LoRA 학습에서는 <strong>AdamW</strong>를 흔히 사용합니다.</p>
      </div>
    </div>
    <p className="nh-hessian-caveat">Hessian의 크기는 미분 대상 파라미터 수로 결정됩니다. LoRA에서 고정한 원본 W가 학습 대상에 자동으로 포함되는 것은 아닙니다.</p>
  </section>;
}
