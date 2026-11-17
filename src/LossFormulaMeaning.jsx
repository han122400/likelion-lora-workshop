import React from 'react';
import { ArrowRight } from 'lucide-react';
import './loss-formula-meaning.css';

const learningSteps = [
  ['순전파', '현재 가중치로 예측합니다.'],
  ['Loss 계산', '예측을 정답 데이터와 비교합니다.'],
  ['역전파', '가중치에 대한 Loss의 기울기를 구합니다.'],
  ['옵티마이저', '기울기를 이용해 가중치를 조정합니다.'],
];

export default function LossFormulaMeaning({ table }) {
  return <section className="loss-meaning" aria-label="예시 손실 함수의 의미와 실제 AI 학습의 차이">
    <div className="loss-meaning-toy">
      <h3 className="loss-meaning-formula"><span role="math" aria-label="손실 L은 가중치 w에서 3을 뺀 값의 제곱입니다.">L = (w − 3)<sup>2</sup></span></h3>
      <p className="loss-meaning-caveat">실제 LLM의 Loss가 아닌, 경사하강법 설명용 예시</p>
      <dl className="loss-meaning-symbols">
        <div><dt>L</dt><dd>Loss · 손실</dd></div>
        <div><dt>w</dt><dd>현재 가중치</dd></div>
        <div><dt>3</dt><dd>Loss가 최소인 w<br/><small>이 예시에서 정한 값</small></dd></div>
        <div><dt>²</dt><dd>차이를 제곱</dd></div>
      </dl>
      <table className="loss-meaning-table">
        <caption>가중치에 따라 달라지는 Loss</caption>
        <thead><tr>{table.headers.map(header => <th key={header} scope="col">{header}</th>)}</tr></thead>
        <tbody>{table.rows.map((row, index) => <tr key={index} className={String(row[0]).trim() === '3' ? 'loss-meaning-minimum' : ''}>
          {row.map((cell, cellIndex) => cellIndex === 0 ? <th scope="row" key={cellIndex}>{cell}</th> : <td key={cellIndex}>{cell}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    <div className="loss-meaning-real">
      <h3>실제 AI는 정답 가중치를 미리 알까요?</h3>
      <p className="loss-meaning-answer"><strong>정답 데이터는 주지만,<br/>정답 가중치는 주지 않습니다.</strong></p>
      <ol className="loss-meaning-steps">
        {learningSteps.map(([title, explanation], index) => <li key={title}>
          <span aria-hidden="true">{index + 1}</span><div><strong>{title}</strong><p>{explanation}</p></div>
        </li>)}
      </ol>
      <p className="loss-meaning-lora"><strong>LoRA에서는</strong><span>A·B 조정 <ArrowRight aria-hidden="true"/> 정답 토큰의 확률을 높이도록 학습</span></p>
    </div>
  </section>;
}
