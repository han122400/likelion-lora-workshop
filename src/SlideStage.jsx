import React from 'react';
import { ArrowRight, FileText, Search, MessageSquareText, LockKeyhole } from 'lucide-react';
import ForwardBackwardVisual from './ForwardBackwardVisual.jsx';
import GradientExplorer from './GradientExplorer.jsx';
import LossFormulaMeaning from './LossFormulaMeaning.jsx';
import HyperparameterExperiments from './HyperparameterExperiments.jsx';
import HyperparameterExplorer from './HyperparameterExplorer.jsx';
import TrainingExample from './TrainingExample.jsx';
import OverfittingVisual from './OverfittingVisual.jsx';
import OptimizationRoles from './OptimizationRoles.jsx';
import SftVisual from './SftVisual.jsx';

export function Matrix({ compact = false, rank = 2 }) {
  return <div className={`matrix-composition ${compact ? 'compact' : ''}`} role="img" aria-label="고정된 가중치 W에 작은 행렬 B와 A의 곱을 더하는 LoRA 구조">
    <div className="matrix-part"><span className="matrix-letter">W</span><div className="matrix-grid base" style={{ '--cols': 6, '--rows': 6 }}>{Array.from({length:36},(_,i)=><i key={i}/>)}</div><span className="matrix-caption">고정</span></div>
    <span className="math-operator">+</span>
    <div className="matrix-part"><span className="matrix-letter">B</span><div className="matrix-grid adapter vertical" style={{ '--cols': rank, '--rows': 6, '--rank': rank }}>{Array.from({length:6*rank},(_,i)=><i key={i}/>)}</div><span className="matrix-caption">학습</span></div>
    <span className="math-operator">×</span>
    <div className="matrix-part"><span className="matrix-letter">A</span><div className="matrix-grid adapter horizontal" style={{ '--cols': 6, '--rows': rank, '--rank': rank }}>{Array.from({length:6*rank},(_,i)=><i key={i}/>)}</div><span className="matrix-caption">학습</span></div>
  </div>;
}

function Cover() {
  return <>
    <div className="cover-layout">
      <div className="cover-copy">
        <h1><em>LoRA</em>로 학습하기<br/><span>같은 모델, 달라지는 답변.</span></h1>
        <p>지난 시간의 RAG와 차이를 짚고,<br className="cover-break"/> 이번에는 LoRA로 모델을 학습합니다.</p>
        <div className="cover-tags"><span>Qwen3.5-2B</span><span>Colab 실습</span></div>
      </div>
      <Matrix compact />
    </div>
    <footer className="cover-footer"><span>멋쟁이사자처럼 · AI 실습 세미나</span><span>지난 시간 RAG <ArrowRight size={17}/> 이번 시간 LoRA</span></footer>
  </>;
}

function RagFlow() {
  const items = [
    [Search, '관련 자료 검색', '질문에 필요한 부분 찾기'],
    [FileText, '문맥 구성', '질문과 근거를 함께 전달'],
    [MessageSquareText, '답변 생성', '제공된 자료를 읽고 답하기'],
  ];
  return <div className="flow-diagram">{items.map(([Icon,title,text],i)=><React.Fragment key={title}>
    {i>0&&<ArrowRight className="flow-arrow"/>}<div className="flow-step"><Icon/><span className="flow-number">0{i+1}</span><h3>{title}</h3><p>{text}</p></div>
  </React.Fragment>)}</div>;
}

function SlideTable({table, className = ''}) {
  return <div className={`table-wrap ${className}`}><table className={`slide-table cols-${table.headers.length}`}>
    <thead><tr>{table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead>
    <tbody>{table.rows.map((row,i)=><tr key={i}>{row.map((cell,j)=>j===0?<th key={j} scope="row">{cell}</th>:<td key={j}>{cell}</td>)}</tr>)}</tbody>
  </table></div>;
}

function SlideBody({slide}) {
  switch (slide.id) {
    case 's02': return <RagFlow/>;
    case 's03': return <SlideTable table={slide.table} className="method-comparison"/>;
    case 's04': return <SftVisual table={slide.table}/>;
    case 's05': return <div className="lora-visual">
      <Matrix/>
      <p className="lora-formula" role="math" aria-label="LoRA를 적용한 가중치 W new는 원본 가중치 W original에 B와 A의 곱을 더한 값입니다. 스케일 계수는 생략한 개념식입니다.">
        <span>W<sub>new</sub></span> = <span>W<sub>original</sub></span> + <em>BA</em>
      </p>
      <p className="diagram-note"><LockKeyhole size={15}/> 원본 W는 고정 · 작은 행렬 A와 B만 학습</p>
      <p className="lora-scale-note">스케일을 생략한 개념식입니다.<br className="lora-note-break"/> 실제로는 <span>(α/r)BA</span>를 더합니다.</p>
    </div>;
    case 's06': return <TrainingExample/>;
    case 's11': return <ForwardBackwardVisual/>;
    case 's12': return <GradientExplorer/>;
    case 's12-meaning': return <LossFormulaMeaning table={slide.table}/>;
    case 's13': return <HyperparameterExplorer/>;
    case 's13-experiments': return <HyperparameterExperiments table={slide.table}/>;
    case 's14': return <OverfittingVisual/>;
    case 's15': return <OptimizationRoles/>;
    default: return slide.table ? <SlideTable table={slide.table}/> : <ol className="key-points">{slide.bullets.map((b,i)=><li key={b}><span>{String(i+1).padStart(2,'0')}</span><p>{b}</p></li>)}</ol>;
  }
}

export default function SlideStage({slide,index}) {
  const isCover = index===0;
  return <article className={`slide-stage ${isCover?'cover':'content-slide'} ${slide.table?'has-table':''}`} aria-label={`${index+1}번째 슬라이드`}>
    <span className="slide-number">{String(index+1).padStart(2,'0')}{slide.id==='s15'&&<span className="optional-label">선택 심화</span>}</span>
    {isCover?<Cover/>:<>
      <header className="slide-heading"><h1>{slide.title.split(/(LoRA)/g).map((part,i)=>part==='LoRA'?<em key={i}>{part}</em>:part)}</h1><p>{slide.lead}</p></header>
      <div className="slide-body">
        <SlideBody slide={slide}/>
      </div>
      <footer className="slide-takeaway"><span className="takeaway-rule"/><p>{slide.takeAway}</p></footer>
    </>}
  </article>;
}
