import React from 'react';
import { ArrowRight, BrainCircuit, LockKeyhole, Undo2 } from 'lucide-react';
import './forward-backward.css';

export default function ForwardBackwardVisual() {
  return <section className="fb-visual" aria-label="순전파로 예측하고 역전파로 기울기를 계산하는 과정">
    <div className="fb-forward-heading"><h3>순전파</h3><p>질문을 읽고, 다음 토큰의 확률을 계산합니다.</p></div>
    <ol className="fb-forward-flow">
      <li className="fb-input">
        <h3>질문</h3>
        <p>대한민국의 수도는?</p>
        <div className="fb-target"><span>정답</span><strong>서울</strong></div>
        <ArrowRight className="fb-connector" aria-hidden="true"/>
      </li>
      <li className="fb-model">
        <BrainCircuit aria-hidden="true"/>
        <h3>Qwen</h3>
        <p>기존 가중치 W + LoRA</p>
        <span>다음 토큰의 확률 예측</span>
        <ArrowRight className="fb-connector" aria-hidden="true"/>
      </li>
      <li className="fb-output">
        <h3>예측 확률</h3>
        <ul className="fb-probabilities" aria-label="가상 예측 확률: 서울 70%, 부산 20%, 대전 10%">
          {[['서울', 70], ['부산', 20], ['대전', 10]].map(([city, probability]) => <li key={city} className={city === '서울' ? 'fb-correct' : ''}>
            <span>{city}{city === '서울' && <small>정답</small>}</span>
            <span className="fb-probability-track" aria-hidden="true"><i style={{ width: `${probability}%` }}/></span>
            <strong>{probability}%</strong>
          </li>)}
        </ul>
        <div className="fb-loss"><span>Loss</span><strong>−ln(0.7) ≈ 0.357</strong></div>
      </li>
    </ol>
    <div className="fb-backward-strip">
      <div className="fb-backward-explanation">
        <h3><Undo2 aria-hidden="true"/>역전파: 무엇을 얼마나 바꾸면 될까?</h3>
        <p>Loss를 가중치로 미분해, 조금 바꿨을 때 Loss가 얼마나 변하는지 <strong>(기울기)</strong>를 계산합니다.</p>
      </div>
      <div className="fb-optimizer-role"><h3>옵티마이저</h3><p>계산된 기울기를 이용해<br/>실제 A·B 값을 수정합니다.</p></div>
    </div>
    <div className="fb-notes">
      <p className="fb-frozen"><LockKeyhole aria-hidden="true"/><span>원본 <strong>W는 고정</strong> · <strong>A와 B만 학습</strong></span></p>
      <p className="fb-example-note">도시 이름을 토큰 하나로 단순화한 가상 예시 · 실제 모델 출력 아님</p>
    </div>
  </section>;
}
