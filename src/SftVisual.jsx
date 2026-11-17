import React from 'react';
import { ArrowDown, BrainCircuit, FileText, Target } from 'lucide-react';
import './sft-visual.css';

const targetAnswer = '{"answer":"화요일",\n "evidence":["T01"],\n "status":"answered"}';

export default function SftVisual({ table }) {
  return <section className="sft-visual" aria-label="SFT 학습 방식 비교와 입력·모범 답변 예시">
    <div className="sft-comparison">
      <table className="sft-method-table">
        <caption>어떤 가중치를 학습할까요?</caption>
        <thead><tr>{table.headers.map(header => <th key={header} scope="col">{header}</th>)}</tr></thead>
        <tbody>{table.rows.map(([name, description]) => {
          const current = name.trim().startsWith('LoRA');
          return <tr key={name} className={current ? 'sft-current-method' : undefined}>
            <th scope="row"><span>{name}</span>{current && <small>이번 실습</small>}</th>
            <td>{description}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>

    <div className="sft-example-flow" aria-label="기존 모델에 입력과 모범 답변을 제공해 학습하는 흐름">
      <div className="sft-base-model">
        <BrainCircuit aria-hidden="true" />
        <h3>기존 모델</h3><strong>Qwen3.5-2B</strong>
      </div>
      <ArrowDown className="sft-flow-arrow" aria-hidden="true" />
      <section className="sft-example">
        <h3><FileText aria-hidden="true" />SFT 학습 예시</h3>
        <dl className="sft-inputs">
          <div><dt>질문</dt><dd>정규 세미나는 무슨 요일이야?</dd></div>
          <div><dt>참고 문서</dt><dd>[T01] 매주 화요일 오후 7시에 열립니다.</dd></div>
        </dl>
        <div className="sft-target-answer"><span>모범 답변</span><pre><code>{targetAnswer}</code></pre></div>
      </section>
      <ArrowDown className="sft-flow-arrow" aria-hidden="true" />
      <div className="sft-training-goal">
        <Target aria-hidden="true" />
        <div><h3>학습 목표</h3><p>문서에 근거해 정해진 형식으로 답하도록 학습</p></div>
      </div>
    </div>
  </section>;
}
