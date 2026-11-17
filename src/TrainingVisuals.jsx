import React from 'react';
import { ArrowRight, Calculator, LockKeyhole, MoveDown, RotateCcw, SlidersHorizontal } from 'lucide-react';
import './training-visuals.css';

function FlowNode({ number, icon: Icon, title, children, accent = false, last = false }) {
  return <li className={`training-node ${accent ? 'training-node-accent' : ''}`}>
    <div className="training-node-top"><span>{number}</span><Icon aria-hidden="true" /></div>
    <h3>{title}</h3>
    <div className="training-node-copy">{children}</div>
    {!last && <ArrowRight className="training-connector" aria-hidden="true" />}
  </li>;
}

export function BackpropVisual() {
  return <section className="training-visual" aria-label="역전파로 기울기를 계산하고 AdamW로 LoRA 행렬을 업데이트하는 과정">
    <ol className="training-flow training-flow-four">
      <FlowNode number="01" icon={Calculator} title="loss">
        <p>현재 예측의 오차</p>
      </FlowNode>
      <FlowNode number="02" icon={RotateCcw} title="역전파">
        <p>파라미터 변화에 따른<br/>loss의 변화율 계산</p>
      </FlowNode>
      <FlowNode number="03" icon={MoveDown} title="기울기">
        <p>A·B를 조정할 때<br/>사용할 변화 방향과 크기</p>
      </FlowNode>
      <FlowNode number="04" icon={SlidersHorizontal} title="AdamW" accent last>
        <p>기울기와 누적 통계로<br/>A·B 값 업데이트</p>
      </FlowNode>
    </ol>
    <div className="training-weight-pair">
      <div className="training-weight-fixed"><span className="training-weight-symbol">W</span><div><strong><LockKeyhole size={16} aria-hidden="true"/> 원본 가중치</strong><p>계산에 사용하지만 값은 고정</p></div></div>
      <div className="training-weight-changing"><span className="training-weight-symbol">A · B</span><div><strong><SlidersHorizontal size={16} aria-hidden="true"/> LoRA 행렬</strong><p>optimizer가 값을 조정</p></div></div>
    </div>
    <p className="training-schematic">역전파는 기울기 계산 · 파라미터 업데이트는 optimizer의 역할</p>
  </section>;
}
