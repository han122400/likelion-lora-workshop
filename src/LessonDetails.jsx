import React, {useState} from 'react';
import { Check, Copy, ArrowUpRight, Download, ChevronDown } from 'lucide-react';
import { sources, questions, configCode, outputExample } from './data.js';
import notebookRaw from '../workshop-assets/lora_workshop.ipynb?raw';

export const COLAB_URL = 'https://colab.research.google.com/#create=true';

// 노트북(lora_workshop.ipynb)의 코드 셀을 '## N단계' 제목 기준으로 묶습니다. 웹과 노트북 코드가 항상 같도록 노트북 파일을 직접 읽습니다.
const STEPS = (() => {
  const steps = {};
  let step = 0;
  for (const cell of JSON.parse(notebookRaw).cells) {
    const src = cell.source.join('');
    if (cell.cell_type === 'markdown') {
      const m = src.match(/##\s*(\d)단계\s*·\s*(.+)/);
      if (m) { step = +m[1]; steps[step] = { title: m[2].trim(), cells: [] }; }
    } else if (step) {
      steps[step].cells.push({ title: (src.match(/#@title\s*(.+)/) || [])[1] || '코드', code: src });
    }
  }
  return steps;
})();
const SLIDE_STEPS = { s19: [1], s20: [2], s21: [3, 4], s22: [5, 6] };

function ColabSteps({ ids, onToast }) {
  return ids.filter((n) => STEPS[n]).map((n) => <div key={n} className="colab-step">
    <h3>{n}단계 · {STEPS[n].title}</h3>
    {STEPS[n].cells.map((c) => <CopyCode key={c.title} label={`새 코드 셀에 붙여 넣기 · ${c.title}`} code={c.code} onToast={onToast}/>)}
  </div>);
}

export function CopyCode({code,label='CMD',onToast}) {
  const [copied,setCopied]=useState(false);
  const copy=async()=>{try{await navigator.clipboard.writeText(code);setCopied(true);setTimeout(()=>setCopied(false),2000);}catch{onToast?.('복사 권한이 없어요. 코드 내용을 선택해 복사해 주세요.');}};
  return <div className="code-block"><div className="code-toolbar"><span>{label}</span><button onClick={copy} aria-label={`${label} 코드 복사`}>{copied?<Check size={16}/>:<Copy size={16}/>} {copied?'복사했어요':'복사'}</button></div><pre><code>{code}</code></pre></div>;
}

function RankExplorer() {
  const [rank,setRank]=useState(8);
  const full=4096*4096;
  const small=2*4096*rank;
  return <section className="rank-explorer"><div><h3>rank를 바꾸면 얼마나 달라질까요?</h3><p>4096 × 4096 가중치 행렬 하나로 계산한 예시입니다.<br/>실제 Qwen 전체 파라미터 수나 성능 측정값이 아닙니다.</p></div><div className="rank-controls"><label htmlFor="rank">LoRA rank <strong>r = {rank}</strong></label><input id="rank" type="range" min="0" max="3" step="1" value={[4,8,16,32].indexOf(rank)} onChange={e=>setRank([4,8,16,32][+e.target.value])}/><div className="range-labels"><span>4</span><span>8</span><span>16</span><span>32</span></div><div className="rank-readout"><span>추가로 학습하는 파라미터<strong>{small.toLocaleString()}개</strong></span><span>원본 행렬 대비<strong>{(100*small/full).toFixed(2)}%</strong></span></div><small>A와 B의 파라미터 수 = r × (4096 + 4096)</small></div></section>;
}

export default function LessonDetails({slide,index,onToast,onResources}) {
  const note=slide.learnerNote || slide.lead;
  const isPractice=slide.chapter===6;
  return <div className="lesson-details">
    <section className="reading-panel">
      <div className="lesson-explanation"><h2>{index===0?'이번 시간에 할 일':'조금 더 이해하기'}</h2><p>{note}</p><ul>{slide.bullets.map(b=><li key={b}>{b}</li>)}</ul></div>
      <aside className="reflection"><h2>핵심 질문</h2><span className="question-number">0{slide.chapter}</span><p>{questions[slide.chapter]}</p><details><summary>기억할 문장 <ChevronDown size={16}/></summary><p>{slide.takeAway}</p></details></aside>
    </section>
    {slide.id==='s05'&&<RankExplorer/>}
    {['s10','s16'].includes(slide.id)&&<section className="example-section"><div><h3>정답 예시의 출력 형식</h3><p>제공 데이터의 멋사 중부대 14기 운영 안내 FAQ 예시입니다.<br/>모델이 실제로 생성한 결과가 아닙니다.</p><p className="document-excerpt">[T01] 멋사 중부대 14기 정규 세미나는 매주 화요일 오후 7시에 열린다.</p><strong>질문: 정규 세미나는 무슨 요일에 열려?</strong></div><CopyCode label="학습 데이터의 정답 JSON" code={outputExample} onToast={onToast}/></section>}
    {slide.id==='s13'&&<CopyCode label="train.py · 학생이 바꿀 값" code={configCode} onToast={onToast}/>}
    {isPractice&&<section className="practice-details"><div className="section-title"><div><h2>Colab에서 따라하기</h2><p>Colab 새 노트북을 열고, 아래 코드 블록을 하나씩 복사해 새 코드 셀에 붙여 넣고 실행하세요.</p></div><div className="download-bar"><a className="button primary" href="/downloads/lora-workshop.zip" download="lora-workshop.zip"><Download size={17}/> 실습 파일 ZIP</a><a className="button" href={COLAB_URL} target="_blank" rel="noreferrer">Colab 열기 <ArrowUpRight size={16}/></a></div></div>
      {slide.id==='s19'&&<div className="environment-note"><strong>시작 전에</strong><p>① 실습 파일 ZIP 다운로드 ② Colab 열기 → 새 노트북 ③ 메뉴 런타임 → 런타임 유형 변경 → T4 GPU ④ 아래 코드를 순서대로 새 코드 셀(+ 코드)에 붙여 넣고 Ctrl+Enter로 실행. 1-2를 실행하면 나오는 ‘파일 선택’에서 lora-workshop.zip을 고르세요. 1-3 설치가 끝나면 런타임 → 세션 다시 시작을 누른 뒤 1-4부터 이어서 실행합니다. 무료 GPU 배정은 보장되지 않으므로, GPU를 받지 못하면 조원의 화면으로 함께 진행하세요.</p></div>}
      {['s20','s21'].includes(slide.id)&&<p className="caption">{slide.id==='s20'?'2-2':'4-1'} ①~⑤ 코드에는 가이드 입력 1~5의 질문과 문서가 이미 들어 있습니다. 하나씩 새 코드 셀에 붙여 넣고 실행만 하세요. 입력 칸은 고치지 않아도 됩니다.</p>}
      {SLIDE_STEPS[slide.id]&&<ColabSteps ids={SLIDE_STEPS[slide.id]} onToast={onToast}/>}
      <p className="caption">1 환경 세팅 → 2 기본 모델에 질문 → 3 LoRA 학습 → 4 LoRA 모델에 같은 질문 → 5 비교 → 6 과제: 최적 변수 찾기 · 결과는 Colab의 outputs/ 폴더에 저장</p>
    </section>}
    <section className="references"><h3>더 읽어보기</h3><div>{sources.filter(s=>s.chapters.includes(slide.chapter)).map(s=><a key={s.url} href={s.url} target="_blank" rel="noreferrer">{s.name}<ArrowUpRight size={15}/></a>)}</div></section>
  </div>;
}
