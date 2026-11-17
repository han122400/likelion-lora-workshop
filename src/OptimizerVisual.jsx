import React from 'react';
import { ArrowRight } from 'lucide-react';
import './optimizer-visual.css';

export default function OptimizerVisual() {
  return <section className="optimizer-visual" aria-label="옵티마이저가 기울기를 이용해 LoRA 가중치 한 칸을 업데이트하는 예시">
    <div className="optimizer-example-heading">
      <h3>LoRA의 A 또는 B에 들어 있는 값 하나</h3>
      <p>단순 경사하강법(SGD) 예시 · 실제 학습 결과 아님</p>
    </div>
    <div className="optimizer-formula" role="math" aria-label="새 가중치는 현재 가중치에서 학습률 곱하기 기울기를 뺀 값입니다.">
      <span>새 가중치</span><b>=</b><span>현재 가중치</span><b>−</b><span className="optimizer-formula-factor">학습률 × 기울기</span>
    </div>
    <ol className="optimizer-value-flow">
      <li>
        <h3>현재값</h3>
        <strong>1.00</strong>
        <p>업데이트 전</p>
        <ArrowRight className="optimizer-connector" aria-hidden="true"/>
      </li>
      <li className="optimizer-change">
        <h3>업데이트량</h3>
        <strong>−0.02</strong>
        <p>−<span>0.10</span> × <span>0.20</span></p>
        <ArrowRight className="optimizer-connector" aria-hidden="true"/>
      </li>
      <li className="optimizer-result">
        <h3>새값</h3>
        <strong>0.98</strong>
        <p>1.00 − 0.02</p>
      </li>
    </ol>
    <p className="optimizer-direction"><strong>기울기가 +0.20</strong>이면, 현재 가중치를 조금 높일 때 loss가 증가하는 방향입니다.<br className="optimizer-direction-break"/> 여기서는 가중치를 낮추는 방향으로 조정합니다.</p>
    <div className="optimizer-concepts">
      <div><h3>학습률 0.10</h3><p>업데이트 크기의 기준 · 설명용 값</p></div>
      <div><h3>AdamW</h3><p>이전 기울기의 통계까지 활용</p></div>
    </div>
  </section>;
}
