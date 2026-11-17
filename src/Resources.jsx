import React, { useEffect, useId, useRef } from 'react';
import { ArrowUpRight, Download, FileText, X } from 'lucide-react';
import { CopyCode, COLAB_URL } from './LessonDetails.jsx';
import { sources, setupCommand, guideInputs } from './data.js';

export default function Resources({ open, onClose, onToast }) {
  const dialogRef = useRef(null);
  const returnFocusRef = useRef(null);
  const openRef = useRef(open);
  const onCloseRef = useRef(onClose);
  const titleId = useId();
  const descriptionId = useId();

  openRef.current = open;
  onCloseRef.current = onClose;

  const restoreFocus = () => {
    const target = returnFocusRef.current;
    returnFocusRef.current = null;
    if (target?.isConnected && typeof target.focus === 'function') {
      target.focus({ preventScroll: true });
    }
  };

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open && !dialog.open) {
      returnFocusRef.current = document.activeElement;
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    const dialog = dialogRef.current;
    return () => {
      if (dialog?.open) dialog.close();
      restoreFocus();
    };
  }, []);

  const closeDialog = () => {
    if (dialogRef.current?.open) dialogRef.current.close();
  };

  const handleNativeClose = () => {
    // A queued close event can arrive after a rapid reopen.
    if (dialogRef.current?.open) return;
    try {
      if (openRef.current) onCloseRef.current?.();
    } finally {
      restoreFocus();
    }
  };

  const handleBackdropClick = (event) => {
    if (event.target !== event.currentTarget) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const outside =
      event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom;
    if (outside) closeDialog();
  };

  return (
    <dialog
      ref={dialogRef}
      className="resource-dialog"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      aria-modal="true"
      onClose={handleNativeClose}
      onCancel={(event) => event.stopPropagation()}
      onClick={handleBackdropClick}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <header className="dialog-heading">
        <div>
          <h2 id={titleId}>실습 자료</h2>
          <p id={descriptionId}>웹에서 설명과 가이드 입력을 보고, Colab 노트북에서 실습 코드를 실행하세요.</p>
        </div>
        <button
          type="button"
          className="dialog-close"
          aria-label="실습 자료 닫기"
          onClick={closeDialog}
        >
          <X size={22} aria-hidden="true" />
        </button>
      </header>

      <div className="resource-body">
        <div className="download-bar">
          <a
            className="button primary"
            href="/downloads/lora-workshop.zip"
            download="lora-workshop.zip"
          >
            <Download size={18} aria-hidden="true" />
            실습 파일 ZIP (10개)
          </a>
          <a
            className="button"
            href={COLAB_URL}
            target="_blank"
            rel="noreferrer"
          >
            Colab 열기
            <ArrowUpRight size={15} aria-hidden="true" />
          </a>
          <a
            className="button"
            href="/downloads/README.ko.md"
            target="_blank"
            rel="noreferrer"
          >
            <FileText size={18} aria-hidden="true" />
            README 읽기
            <ArrowUpRight size={15} aria-hidden="true" />
          </a>
        </div>

        <section className="resource-section">
          <h3>01. Colab에서 시작하기</h3>
          <ol>
            <li>위의 <strong>실습 파일 ZIP</strong>을 내려받습니다(압축은 풀지 않아도 됩니다).</li>
            <li><strong>Colab 열기</strong>를 눌러 새 노트북을 엽니다(Google 계정 로그인 필요).</li>
            <li>런타임 → 런타임 유형 변경 → <strong>T4 GPU</strong>를 선택합니다.</li>
            <li>‘LoRA 실습과 비교’ 챕터의 코드를 순서대로 복사해 새 코드 셀에 붙여 넣고 실행합니다. 2단계와 4단계는 각각 필수 코드 블록 하나로 진행합니다.</li>
            <li>1-2 코드를 실행하면 나오는 ‘파일 선택’에서 lora-workshop.zip을 올립니다. 1-3 설치 뒤에는 세션을 다시 시작하고 1-4부터 이어 갑니다.</li>
          </ol>
          <p className="caption">
            Qwen3.5-2B를 직접 학습하므로 NVIDIA GPU가 필요합니다. 무료 Colab GPU 배정은 보장되지 않으니,
            GPU를 받지 못하면 조원의 화면으로 함께 진행하세요. 이 웹페이지에서 모델 학습이 실행되지는 않습니다.
          </p>
        </section>

        <section className="resource-section">
          <h3>02. 노트북 6단계</h3>
          <ol>
            <li>환경 세팅: GPU · 패키지 버전 · 데이터 구성표</li>
            <li>기본 모델에 질문: 2-1 한 번 실행 → 가이드 답변 카드 · 검증 점수</li>
            <li>기본 변수로 LoRA 학습: 학습 파라미터 비율 · loss 그래프</li>
            <li>LoRA 모델에 같은 질문: 4-1 한 번 실행 → 가이드 답변 카드 · 검증 점수</li>
            <li>최종 비교: 질문별 ▲개선 · ▼악화 표 · 지표 막대 · test로 마지막 확인</li>
            <li>과제: 6-1 빈칸에 내 문장 3개 쓰기 → 변수 하나씩 바꿔 학습 → 정확히 출력되는 최소 설정 찾기</li>
          </ol>
          <CopyCode
            code={guideInputs.map((g) => `입력 ${g.no} · 문서_ID ${g.document_id} · ${g.question}`).join('\n')}
            label="2단계 · 4단계에 똑같이 넣을 가이드 입력"
            onToast={onToast}
          />
        </section>

        <section className="resource-section">
          <h3>03. (선택) 내 PC의 NVIDIA GPU에서 CMD로 실행</h3>
          <p>
            CUDA 지원 PyTorch가 준비된 PC라면 압축을 푼 폴더에서 한 번에 실행할 수도 있습니다.
            기본 실행은 학습 전 평가 → LoRA 학습 → 학습 후 평가 순서이고, 변수는 실행 옵션으로 하나만 바꿉니다.
          </p>
          <CopyCode code={setupCommand} label="Windows CMD · 설치" onToast={onToast} />
          <CopyCode
            code={'python train.py --check\npython train.py\npython train.py --lr 1e-4\npython train.py --final-test'}
            label="CMD · 점검 → 기본 실행 → 한 값 바꾸기 → 최종 평가"
            onToast={onToast}
          />
          <p className="caption">
            Windows에서 requirements.txt만 설치하면 CPU용 PyTorch가 깔릴 수 있습니다. README의 PyTorch 설치 안내를 먼저 확인하세요.
          </p>
        </section>

        <section className="resource-section">
          <h3>실제 결과는 어디에 있나요?</h3>
          <p>
            노트북 화면에 단계별 카드·표·그래프로 바로 표시되고, 파일로도 저장됩니다.
            5-3 셀은 <code>outputs/stepwise-날짜-시간/</code>, 6단계 실험은 <code>outputs/run-날짜-시간/</code>에 저장되며,
            실험 순위의 원본은 <code>outputs/experiments.jsonl</code>입니다.
            <code> comparison.csv</code>에서 LoRA 학습 전후의 실제 답변,
            <code> summary.json</code>에서 평가 지표,
            <code> loss_history.json</code>에서 학습 기록을 확인하세요.
          </p>
          <p className="caption">
            <code>run_status.json</code>의 상태가 <code>completed</code>인지 확인하세요.
            일부 출력만 남아 있으면 전체 실행이 성공한 것은 아닙니다.
            데이터는 학습 36개·검증 10개·테스트 10개, 참고 문서 원문 20개입니다.
            정규 세미나 시간과 운영진 정보는 실제 14기 정보이고, 나머지 규정·일정·금액은 교육용 예시입니다.
          </p>
        </section>

        <section className="resource-section">
          <h3>공식 참고 자료</h3>
          <div className="resource-links">
            {sources.map((source) => (
              <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                <span>
                  <strong>{source.name}</strong>
                  <span className="caption">{source.detail}</span>
                </span>
                <ArrowUpRight size={16} aria-hidden="true" />
              </a>
            ))}
          </div>
        </section>
      </div>
    </dialog>
  );
}
