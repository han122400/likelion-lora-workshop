import React, { useId, useRef, useState } from 'react';
import { ArrowRight, Check, SlidersHorizontal, RefreshCw } from 'lucide-react';
import { NewtonPanel, HessianPanel } from './NewtonHessianPanels.jsx';
import './optimization-roles.css';

const topics = [
  ['뉴턴법', '기울기 + 곡률로 이동'],
  ['Hessian', '곡률 정보를 모은 행렬'],
  ['AdamW와 Optuna', '가중치 수정 vs 설정 탐색'],
  ['Optuna 실험', '검증 점수로 조합 비교'],
];

const trials = [
  ['1', '0.0001', '3', '8', '0.42'],
  ['2', '0.0002', '3', '16', '0.31'],
  ['3', '0.0003', '5', '32', '0.38'],
  ['4', '0.00015', '4', '16', '0.27'],
];

function RoleComparison() {
  const rows = [
    ['역할', '가중치 업데이트', '하이퍼파라미터 탐색'],
    ['조절 대상', 'LoRA의 A, B', 'Learning Rate · Epochs · Rank'],
    ['사용하는 정보', '기울기와 기울기 제곱의 이동평균', '실험별 검증 점수'],
    ['작동 시점', '한 번의 모델 학습 안에서', '여러 학습 실험을 관리할 때'],
  ];
  return <section className="opt-role-comparison" aria-label="AdamW와 Optuna의 역할 비교">
    <table className="opt-role-table">
      <thead><tr><th scope="col">구분</th><th scope="col">AdamW</th><th scope="col">Optuna</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row[0]}><th scope="row">{row[0]}</th><td>{row[1]}</td><td>{row[2]}</td></tr>)}</tbody>
    </table>
    <div className="opt-role-connection">
      <span><SlidersHorizontal aria-hidden="true"/><b>Optuna</b>가 설정 선택</span>
      <ArrowRight className="opt-connection-arrow" aria-hidden="true"/>
      <span><RefreshCw aria-hidden="true"/><b>AdamW</b>가 A·B 업데이트</span>
    </div>
    <p className="opt-footnote">Optuna는 가중치를 직접 수정하지 않습니다. 각 실험에서 사용할 학습 설정을 탐색합니다.</p>
  </section>;
}

function OptunaTrials() {
  return <section className="opt-trials" aria-label="Optuna의 가상 실험 예시">
    <div className="opt-trial-grid">
      <div className="opt-trial-results">
        <table className="opt-trial-table">
          <caption>Qwen을 LoRA로 학습한다면 <span>가상의 실험 결과</span></caption>
          <thead><tr>{['실험', 'Learning Rate', 'Epochs', 'Rank', '검증 Loss ↓'].map(header => <th key={header} scope="col">{header}</th>)}</tr></thead>
          <tbody>{trials.map((row, index) => <tr key={row[0]} className={index === 3 ? 'opt-best-trial' : undefined}>
            <th scope="row">{row[0]}{index === 3 && <Check size={14} aria-label="최저 손실"/>}</th>
            {row.slice(1).map((value, column) => <td key={column}>{value}</td>)}
          </tr>)}</tbody>
        </table>
        <p className="opt-best-summary"><strong>현재 4개 실험 중 최선: 4번</strong><span>학습률 0.00015 · Epochs 4 · Rank 16</span></p>
      </div>
      <ol className="opt-trial-flow" aria-label="Optuna 실험 과정">
        {[
          ['설정 제안', '학습률 · Epochs · Rank 선택'],
          ['LoRA 학습', 'AdamW로 A·B 업데이트'],
          ['검증 Loss 계산', '동일한 검증 데이터로 평가'],
          ['다른 설정 시도', '이전 결과를 활용해 제안 가능'],
          ['가장 좋은 결과 선택', '주어진 실험 횟수 안에서 비교'],
        ].map(([title, detail], index) => <li key={title}><span>{index + 1}</span><div><strong>{title}</strong><p>{detail}</p></div></li>)}
      </ol>
    </div>
    <p className="opt-footnote">전역 최적값을 보장하지 않습니다. 다음 조합의 제안 방식은 탐색 알고리즘에 따라 다르며, 테스트 데이터는 최종 평가용으로 남겨 둡니다.</p>
  </section>;
}

export default function OptimizationRoles() {
  const [topic, setTopic] = useState(0);
  const id = useId();
  const tabs = useRef([]);

  function handleTabKey(event, index) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % topics.length;
    else if (event.key === 'ArrowLeft') next = (index + topics.length - 1) % topics.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = topics.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    setTopic(next);
    tabs.current[next]?.focus();
  }

  return <section className="optimization-roles" aria-label="뉴턴법, Hessian, Optuna의 역할 살펴보기">
    <div className="opt-topic-tabs" role="tablist" aria-label="심화 개념 선택">
      {topics.map(([title, detail], index) => <button key={title} type="button" role="tab"
        id={`${id}-tab-${index}`} aria-controls={`${id}-panel-${index}`} aria-selected={topic === index}
        tabIndex={topic === index ? 0 : -1} ref={element => { tabs.current[index] = element; }}
        onClick={() => setTopic(index)} onKeyDown={event => handleTabKey(event, index)}>
        <strong><span>{index + 1}</span>{title}</strong><small>{detail}</small>
      </button>)}
    </div>
    {[NewtonPanel, HessianPanel, RoleComparison, OptunaTrials].map((Panel, index) => <div key={index}
      className="opt-topic-panel" role="tabpanel" id={`${id}-panel-${index}`}
      aria-labelledby={`${id}-tab-${index}`} hidden={topic !== index} tabIndex={0}>
      <Panel/>
    </div>)}
  </section>;
}
