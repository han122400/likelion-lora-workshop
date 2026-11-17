import React from 'react';
import { ArrowLeftRight, Database, ChartNoAxesCombined } from 'lucide-react';
import './hyperparameter-experiments.css';

const rowNotes = ['비교 기준', '학습률만 변경', '반복 횟수만 변경', 'rank만 변경'];

export default function HyperparameterExperiments({ table }) {
  const baseline = table.rows[0];

  return (
    <section className="hp-experiments" aria-label="한 번에 한 설정만 바꾸는 실험 예시">
      <div className="hp-table-wrap">
        <table className="hp-comparison-table">
          <caption>B·C·D는 모두 A를 기준으로 비교합니다.</caption>
          <thead>
            <tr>{table.headers.map((header) => <th key={header} scope="col">{header}</th>)}</tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={row[0]} className={rowIndex === 0 ? 'hp-baseline-row' : undefined}>
                <th scope="row">
                  <div className="hp-experiment-name"><strong>{row[0]}</strong><span>{rowNotes[rowIndex]}</span></div>
                </th>
                {row.slice(1).map((value, offset) => {
                  const changed = rowIndex > 0 && value !== baseline[offset + 1];
                  return (
                    <td key={table.headers[offset + 1]} className={changed ? 'hp-cell-changed' : undefined}>
                      <span className="hp-cell-value">{value}</span>
                      {changed && <span className="hp-change-label">변경</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hp-example-note">설정값은 실험 예시이며, 최적값을 뜻하지 않습니다.</p>
      <div className="hp-comparison-rules">
        <div className="hp-comparison-rule">
          <ArrowLeftRight aria-hidden="true" />
          <div><h3>A와 비교</h3><p>B·C·D에서 하나씩만 변경</p></div>
        </div>
        <div className="hp-comparison-rule">
          <Database aria-hidden="true" />
          <div><h3>같은 검증 데이터</h3><p>나머지 조건도 동일하게 유지</p></div>
        </div>
        <div className="hp-comparison-rule hp-performance-rule">
          <ChartNoAxesCombined aria-hidden="true" />
          <div><h3>Loss + 실제 작업 성능</h3><p>정답 정확도 · JSON 출력 성공률 · 근거 일치</p></div>
        </div>
      </div>
    </section>
  );
}
