import React, {useCallback, useEffect, useRef, useState} from 'react';
import {ArrowLeft, ArrowRight, Check, Circle, Download, Expand, Menu, Monitor, NotebookPen, X, BookOpen, Link} from 'lucide-react';
import {slides, chapterNames} from './data.js';
import SlideStage from './SlideStage.jsx';
import LessonDetails from './LessonDetails.jsx';
import Resources from './Resources.jsx';

const PROGRESS_KEY='likelion-lora-progress-v1';
function initialIndex(){const i=slides.findIndex(s=>`#${s.id}`===window.location.hash);return Math.max(0,i);}
function loadProgress(){try{const p=JSON.parse(localStorage.getItem(PROGRESS_KEY)||'[]');return Array.isArray(p)?p.filter(id=>slides.some(s=>s.id===id)):[];}catch{return [];}}

function Sidebar({slide,completed,onNavigate,onResources,onClose}) {
  return <aside className="sidebar" aria-label="워크숍 목차"><div className="sidebar-title"><h2>워크숍 목차</h2><button className="icon-button mobile-close" onClick={onClose} aria-label="목차 닫기"><X/></button></div>
    <nav>{chapterNames.map((name,i)=>{const start=slides.findIndex(s=>s.chapter===i+1);return <button key={name} className={`chapter-link ${slide.chapter===i+1?'active':''}`} aria-current={slide.chapter===i+1?'step':undefined} onClick={()=>onNavigate(start)}><span>{String(i+1).padStart(2,'0')}</span><strong>{name}</strong></button>;})}</nav>
    <button className="sidebar-resources" onClick={onResources}><Download size={18}/> 실습 파일 · 참고 자료<ArrowRight size={16}/></button>
    <div className="progress-block"><span>내 학습 진행</span><div className="progress-row"><progress max={slides.length} value={completed.length} aria-label="학습 완료 슬라이드"/><span>{completed.length} / {slides.length}</span></div><small>이 브라우저에 자동 저장됩니다.</small></div>
  </aside>;
}

export default function App(){
  const [index,setIndex]=useState(initialIndex);
  const [present,setPresent]=useState(()=>new URLSearchParams(location.search).get('present')==='1');
  const [completed,setCompleted]=useState(loadProgress);
  const [resources,setResources]=useState(false);
  const [menu,setMenu]=useState(false);
  const [notes,setNotes]=useState(false);
  const [toast,setToast]=useState('');
  const toastTimer=useRef(null);
  const slide=slides[index];
  const notify=useCallback(message=>{setToast(message);clearTimeout(toastTimer.current);toastTimer.current=setTimeout(()=>setToast(''),3500);},[]);
  const go=useCallback(next=>{const n=Math.max(0,Math.min(slides.length-1,next));setIndex(n);setMenu(false);history.replaceState(null,'',`${location.pathname}${location.search}#${slides[n].id}`);window.scrollTo({top:0,behavior:'instant'});},[]);
  const changeMode=useCallback(value=>{setPresent(value);setNotes(false);setMenu(false);const url=new URL(location.href);value?url.searchParams.set('present','1'):url.searchParams.delete('present');history.replaceState(null,'',url);if(!value&&document.fullscreenElement)document.exitFullscreen().catch(()=>{});window.scrollTo({top:0,behavior:'instant'});},[]);
  const fullscreen=useCallback(async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();else notify('이 브라우저에서는 전체 화면을 지원하지 않아요.');}catch{notify('브라우저의 전체 화면 메뉴를 사용해 주세요.');}},[notify]);
  useEffect(()=>{document.title=`${index+1}. ${slide.title.replace('\n',' · ')} — LIKELION`;},[index,slide.title]);
  useEffect(()=>{try{localStorage.setItem(PROGRESS_KEY,JSON.stringify(completed));}catch{}},[completed]);
  useEffect(()=>{const onHash=()=>setIndex(initialIndex());window.addEventListener('hashchange',onHash);return()=>window.removeEventListener('hashchange',onHash);},[]);
  useEffect(()=>{const handle=e=>{if(resources)return;if(e.key==='Escape'){setMenu(false);setNotes(false);if(present&&!document.fullscreenElement)changeMode(false);return;}if(menu||e.altKey||e.ctrlKey||e.metaKey||/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)||e.target.isContentEditable)return;const key=e.key.toLowerCase();if(['arrowright','pagedown'].includes(key)||(key===' '&&e.target.tagName!=='BUTTON')){e.preventDefault();go(index+1);}else if(['arrowleft','pageup'].includes(key)){e.preventDefault();go(index-1);}else if(key==='home'){e.preventDefault();go(0);}else if(key==='end'){e.preventDefault();go(slides.length-1);}else if(key==='f'&&present){e.preventDefault();fullscreen();}else if(key==='n'&&present){e.preventDefault();setNotes(n=>!n);}};window.addEventListener('keydown',handle);return()=>window.removeEventListener('keydown',handle);},[index,present,resources,menu,go,changeMode,fullscreen]);
  useEffect(()=>()=>clearTimeout(toastTimer.current),[]);
  const mark=()=>setCompleted(prev=>prev.includes(slide.id)?prev.filter(id=>id!==slide.id):[...prev,slide.id]);
  const share=async()=>{try{const url=new URL(location.href);url.searchParams.delete('present');url.hash=slide.id;await navigator.clipboard.writeText(url.href);notify('현재 슬라이드 링크를 복사했어요.');}catch{notify('주소창의 링크를 복사해 주세요.');}};
  return <div className={`workshop-app ${present?'presentation-mode':'learning-mode'} ${menu?'menu-open':''}`}>
    <a className="skip-link" href="#lesson" onClick={e=>{e.preventDefault();document.getElementById('lesson')?.focus();}}>본문으로 이동</a>
    <header className="app-header"><div className="brand-group"><button className="icon-button menu-button" onClick={()=>setMenu(true)} aria-label="워크숍 목차 열기"><Menu/></button><button className="brand" onClick={()=>go(0)} aria-label="첫 슬라이드로 이동"><i/>LIKELION</button><span className="brand-divider"/><span className="brand-subtitle">AI WORKSHOP</span></div>
      <div className="header-actions">{present?<><button className="button return-button" onClick={()=>changeMode(false)}><BookOpen size={18}/><span>학습 모드로 돌아가기</span></button><button className="icon-button" onClick={fullscreen} aria-label="전체 화면" title="전체 화면 · F"><Expand/></button></>:<><div className="mode-switch" aria-label="화면 모드"><button className="selected" aria-pressed="true" onClick={()=>changeMode(false)}>학습 모드</button><button aria-pressed="false" onClick={()=>changeMode(true)}><Monitor size={18}/>발표 모드</button></div><button className="icon-button download-button" onClick={()=>setResources(true)} aria-label="실습 자료 열기" title="실습 자료"><Download/></button></>}</div>
    </header>
    {!present&&<><button className="sidebar-backdrop" aria-label="목차 닫기" onClick={()=>setMenu(false)}/><Sidebar slide={slide} completed={completed} onNavigate={go} onResources={()=>{setMenu(false);setResources(true);}} onClose={()=>setMenu(false)}/></>}
    <main id="lesson" className="main-content" tabIndex={-1}>
      {!present&&<div className="slide-meta"><div><span>{String(slide.chapter).padStart(2,'0')}</span> {chapterNames[slide.chapter-1]}</div><div className="slide-meta-right"><button className="share-button" onClick={share} aria-label="현재 슬라이드 링크 복사"><Link size={16}/></button><select value={index} onChange={e=>go(+e.target.value)} aria-label="슬라이드 바로가기">{slides.map((s,i)=><option key={s.id} value={i}>{i+1} / {slides.length} · {s.title.replace('\n',' ')}</option>)}</select><span className="slide-count" aria-hidden="true">{index+1} / {slides.length}</span></div></div>}
      <SlideStage slide={slide} index={index}/>
      <div className="slide-controls">
        {present?<div className="presentation-shortcuts"><span>← → 이동 · F 전체 화면</span><button onClick={()=>setNotes(n=>!n)} className={notes?'active':''} aria-pressed={notes}><NotebookPen size={16}/> 발표자 노트 <kbd>N</kbd></button></div>:<button className={`button complete-button ${completed.includes(slide.id)?'completed':''}`} onClick={mark} aria-pressed={completed.includes(slide.id)}>{completed.includes(slide.id)?<Check size={19}/>:<Circle size={19}/>}학습 완료</button>}
        {present?<div className="presentation-progress"><progress max={slides.length} value={index+1} aria-label="발표 진행"/><span>{index+1} / {slides.length}</span></div>:<span className="keyboard-hint"><ArrowLeft size={17}/><ArrowRight size={17}/> 슬라이드 이동</span>}
        <div className="navigation-buttons"><button className="button previous-button" onClick={()=>go(index-1)} disabled={index===0} aria-label="이전 슬라이드"><ArrowLeft size={20}/></button><button className="button primary next-button" onClick={()=>index===slides.length-1?setResources(true):go(index+1)}>{index===slides.length-1?'실습 자료':'다음'}<ArrowRight size={20}/></button></div>
      </div>
      {present&&notes&&<aside className="speaker-notes"><div><h2>발표자 노트</h2><button className="icon-button" onClick={()=>setNotes(false)} aria-label="발표자 노트 닫기"><X size={20}/></button></div><p>{slide.speakerNote.split('참고:')[0]}</p></aside>}
      {!present&&<LessonDetails slide={slide} index={index} onToast={notify} onResources={()=>setResources(true)}/>}
    </main>
    <Resources open={resources} onClose={()=>setResources(false)} onToast={notify}/>
    {toast&&<div className="toast" role="status"><Check size={17}/>{toast}</div>}
  </div>;
}
