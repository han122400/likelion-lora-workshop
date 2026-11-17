import React from 'react';
import { ArrowRight } from 'lucide-react';
import './practice-goal.css';

export default function PracticeGoal() {
  return <section className="practice-goal" aria-label="이번 실습의 입력과 목표 답변 예시">
    <div className="practice-goal-example">
      <section className="practice-goal-input">
        <h3>이런 입력을 주고</h3>
        <dl>
          <div><dt>참고 문서 T01</dt><dd>정규 세미나는 매주 화요일에 열립니다.</dd></div>
          <div><dt>질문</dt><dd><strong>세미나는 무슨 요일이야?</strong></dd></div>
        </dl>
      </section>
      <ArrowRight className="practice-goal-arrow" aria-hidden="true"/>
      <section className="practice-goal-output">
        <h3>이렇게 답하도록 학습합니다 <small>모범 답변 예시</small></h3>
        <div className="practice-goal-answer">
          <pre><code>{'{\n  "answer": "화요일",\n  "evidence": ["T01"],\n  "status": "answered"\n}'}</code></pre>
          <dl>
            <div><dt>answer</dt><dd>답변</dd></div>
            <div><dt>evidence</dt><dd>근거 문서 번호</dd></div>
            <div><dt>status</dt><dd>답변 가능 여부</dd></div>
          </dl>
        </div>
      </section>
    </div>
    <div className="practice-goal-unknown"><strong>문서에 답이 없다면?</strong><p>“몇 명이 참석해?”처럼 알 수 없는 질문에는 추측하지 않고 정보 부족으로 답합니다.<br/><code>answer: null · evidence: [] · status: insufficient_information</code></p></div>
  </section>;
}
