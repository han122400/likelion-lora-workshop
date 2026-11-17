"""노트북 화면에 입력·출력·수치를 보기 좋게 그려 주는 함수 모음.

`import lab_view as view` 후 view.show_...() 를 호출합니다.
- *_html() 함수: HTML 문자열을 만듭니다(테스트하기 쉽도록 화면 출력과 분리).
- show_*() 함수: 만든 HTML을 노트북 출력 칸에 표시합니다.
모델이나 GPU를 건드리지 않으므로 이 파일만으로는 아무것도 학습하지 않습니다.
"""
from __future__ import annotations

import html
import json
import math

# 화면을 만드는 흐름: 결과 딕셔너리·목록 → *_html()로 HTML 생성 → _display()로 표시.
# 모델이 만든 답변과 채점값을 전달받으므로, 카드를 다시 표시해도 모델을 다시 학습하거나 답변을 생성하지 않습니다.

# 채점 지표의 화면 이름과 설명(train.METRICS와 같은 순서)
METRIC_LABELS = {
    "format_ok": ("JSON 형식", "answer·evidence·status 세 필드의 JSON 하나인가"),
    "answer_correct": ("정답 일치", "answer가 허용 표현과 같은가(보류 문항은 null)"),
    "source_correct": ("근거 ID", "evidence가 정답 문서 ID와 같은가"),
    "grounded_success": ("종합 성공", "형식·정답·근거·status가 모두 맞는가"),
    "context_policy_ok": ("응답 정책", "답이 있으면 답하고, 없으면 보류했는가"),
}
MODE_LABELS = {"base": "기본 모델", "lora": "LoRA 모델"}
MODE_COLORS = {"base": "#8b95a1", "lora": "#ff7710"}  # 회색 = 학습 전, 주황 = 학습 후

# 모든 카드에 공통으로 쓰는 스타일. Colab 다크 모드에서도 읽히도록 배경색을 직접 지정합니다.
STYLE = """
<style>
.lv{font-family:-apple-system,'Segoe UI','Noto Sans KR',sans-serif;color:#1f2328;background:#fff;
    border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;margin:8px 0;max-width:980px;line-height:1.5}
.lv h3{margin:0 0 10px;font-size:16px;color:#1f2328 !important;opacity:1 !important}
.lv th,.lv td,.lv b{color:#1f2328}
.lv .sub{color:#57606a;font-size:12.5px}
.lv table{border-collapse:collapse;width:100%;font-size:13px}
.lv th,.lv td{border-bottom:1px solid #eaeef2;padding:6px 8px;text-align:left;vertical-align:top}
.lv th{background:#f6f8fa;font-weight:600}
.lv pre{background:#f6f8fa;border-radius:6px;padding:8px 10px;margin:4px 0;white-space:pre-wrap;word-break:break-all;font-size:12.5px}
.lv .ok{color:#1a7f37;font-weight:700}.lv .no{color:#cf222e;font-weight:700}.lv .na{color:#8c959f}
.lv .badge{display:inline-block;border-radius:999px;padding:2px 9px;margin:2px 4px 2px 0;font-size:12px;font-weight:600}
.lv .b-ok{background:#dafbe1;color:#1a7f37}.lv .b-no{background:#ffebe9;color:#cf222e}.lv .b-na{background:#eaeef2;color:#57606a}
.lv .grid{display:grid;grid-template-columns:110px 1fr;gap:4px 10px;align-items:start}
.lv .label{color:#57606a;font-size:12.5px;padding-top:6px}
.lv .bar{height:16px;border-radius:4px;display:inline-block;vertical-align:middle}
.lv .track{background:#eaeef2;border-radius:4px;width:100%;height:16px;position:relative}
.lv .warn{background:#fff8c5;border-radius:6px;padding:6px 10px;font-size:12.5px;margin-top:8px}
.lv .best{background:#fff4e5}
.lv .up{color:#1a7f37;font-weight:700}.lv .down{color:#cf222e;font-weight:700}.lv .same{color:#8c959f}
</style>
"""


def _e(value) -> str:
    """HTML에 넣기 전에 특수문자를 안전하게 바꿉니다."""
    # 질문이나 답변에 <, >, & 등이 있어도 HTML 태그가 아니라 원래 글자로 보이게 합니다.
    # 예를 들어 모델이 <script>를 출력해도 화면에서 코드로 실행되지 않게 합니다.
    return html.escape("" if value is None else str(value))


def _mark(value) -> str:
    """True/False/None을 ✔/✘/–로 표시합니다."""
    if value is None:
        return '<span class="na">–</span>'
    return '<span class="ok">✔</span>' if value else '<span class="no">✘</span>'


def _badge(text, value) -> str:
    cls = "b-na" if value is None else ("b-ok" if value else "b-no")
    icon = "–" if value is None else ("✔" if value else "✘")
    return f'<span class="badge {cls}">{icon} {_e(text)}</span>'


def _pct(x) -> str:
    # 채점 비율 0.8은 화면에서 80%로 표시합니다. loss 값에는 이 변환을 적용하지 않습니다.
    return "–" if x is None else f"{100 * x:.0f}%"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _mode_label(mode) -> str:
    return MODE_LABELS.get(mode, mode or "")


def _display(html_text: str) -> None:
    """노트북 출력 칸에 HTML을 표시합니다(노트북 밖에서는 아무것도 하지 않음)."""
    # IPython의 HTML 표시 기능에 공통 스타일과 카드 내용을 넘깁니다.
    # *_html()은 문자열만 돌려주고, 실제 노트북 출력은 이 함수가 맡습니다.
    try:
        from IPython.display import HTML, display
    except ImportError:
        return
    display(HTML(STYLE + html_text))


# ── 1단계: 데이터·모델 정보 ─────────────────────────────────────────
def data_overview_html(docs: dict, splits: dict) -> str:
    """데이터 구성 표: 분할별 예시 수, 답할 수 있는 질문/보류 질문 수, 사용하는 문서 ID."""
    names = {"train": "학습(train)", "validation": "검증(validation)", "test": "최종 평가(test)"}
    uses = {"train": "LoRA 가중치 업데이트", "validation": "학습 전후 비교 · 설정 고르기", "test": "설정 확정 뒤 마지막 1회"}
    rows = []
    for split, items in splits.items():
        answerable = sum(r["answerable"] for r in items)
        doc_ids = sorted({d for r in items for d in r["context_ids"]})
        rows.append(f"<tr><td>{names[split]}</td><td>{len(items)}</td><td>{answerable}</td>"
                    f"<td>{len(items) - answerable}</td><td>{_e(', '.join(doc_ids))}</td><td>{uses[split]}</td></tr>")
    sample = next(iter(docs.values()))
    return (f'<div class="lv"><h3>데이터 구성 · 문서 {len(docs)}개</h3><table><tr><th>분할</th><th>예시</th>'
            f'<th>답 있음</th><th>답 보류</th><th>문서</th><th>쓰임</th></tr>{"".join(rows)}</table>'
            f'<p class="sub">문서 예시 [{_e(sample["id"])}] {_e(sample["title"])}: {_e(sample["text"])}</p></div>')


def model_info_html(info: dict) -> str:
    """불러온 모델·GPU 정보 카드."""
    used, total = info.get("used_gib"), info.get("total_gib")
    ratio = (used / total) if used and total else 0
    return (f'<div class="lv"><h3>기본 모델 준비 완료</h3><div class="grid">'
            f'<div class="label">모델</div><div>{_e(info.get("model"))}</div>'
            f'<div class="label">파라미터</div><div>{info.get("total_params", 0):,}개</div>'
            f'<div class="label">GPU</div><div>{_e(info.get("gpu"))}</div>'
            f'<div class="label">가중치 정밀도</div><div>{_e(info.get("model_dtype"))}</div>'
            f'<div class="label">GPU 메모리</div><div><div class="track"><span class="bar" '
            f'style="width:{100 * ratio:.0f}%;background:#0969da"></span></div>'
            f'<span class="sub">{used or 0:.1f} / {total or 0:.1f} GiB 사용</span></div></div></div>')


# ── 2·4단계: 질문 하나의 입력·출력·채점 ─────────────────────────────
def answer_card_html(r: dict) -> str:
    """질문 하나의 카드: 입력(질문·문서) → 모델 출력(원문) → 채점."""
    # r에는 이미 생성한 raw_output과 각 지표의 True/False가 들어 있습니다.
    # 기대 정답이 없는 자유 질문은 정답·근거를 맞혔다고 판단할 수 없어 JSON 형식만 표시합니다.
    title = f'{_mode_label(r.get("mode"))} 답변'
    if r.get("expected") is not None:
        checks = "".join(_badge(METRIC_LABELS[m][0], r.get(m)) for m in METRIC_LABELS)
        expected = f'<pre>{_e(_json(r["expected"]))}</pre>'
    else:
        checks = _badge("JSON 형식", r.get("format_ok")) + '<span class="sub">데이터셋에 없는 입력이라 정답 비교 없이 출력만 확인합니다.</span>'
        expected = '<span class="na">정답 없음(직접 만든 입력)</span>'
    warn = ""
    if r.get("dataset_split") == "train":
        warn = '<div class="warn">이 질문은 학습(train)에 들어 있던 문항입니다. LoRA 모델이 맞혀도 "외운 것"일 수 있어요.</div>'
    if r.get("hit_token_limit"):
        warn += '<div class="warn">생성 길이 한도에 닿아 답이 잘렸을 수 있습니다.</div>'
    return (f'<div class="lv"><h3>{_e(title)}{" · " + _e(r["id"]) if r.get("id") else ""}</h3><div class="grid">'
            f'<div class="label">입력 · 질문</div><div><pre>{_e(r.get("question"))}</pre></div>'
            f'<div class="label">입력 · 문서</div><div><pre>{_e(r.get("document") or "(문서 없음)")}</pre></div>'
            f'<div class="label">출력 · 원문</div><div><pre>{_e(r.get("raw_output"))}</pre></div>'
            f'<div class="label">기대 정답</div><div>{expected}</div>'
            f'<div class="label">채점</div><div>{checks}</div>'
            f'<div class="label">생성</div><div class="sub">{r.get("generated_tokens", 0)} 토큰 · '
            f'{r.get("generation_seconds", 0):.1f}초</div></div>{warn}</div>')


def results_table_html(results: list[dict], title: str) -> str:
    """여러 질문의 결과 표: 질문 / 모델 출력 / 지표별 ✔✘."""
    head = "".join(f"<th>{METRIC_LABELS[m][0]}</th>" for m in METRIC_LABELS)
    rows = []
    for r in results:
        cells = "".join(f"<td>{_mark(r.get(m))}</td>" for m in METRIC_LABELS)
        rows.append(f'<tr><td>{_e(r.get("id") or "-")}</td><td>{_e(r.get("question"))}</td>'
                    f'<td><pre>{_e(r.get("raw_output"))}</pre></td>{cells}</tr>')
    return (f'<div class="lv"><h3>{_e(title)}</h3><table><tr><th>ID</th><th>질문</th><th>모델 출력</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def metrics_html(summaries: dict, title: str = "점수 요약") -> str:
    """지표별 막대그래프. summaries = {"base": {...}, "lora": {...}} 또는 {"기본 모델": {...}}."""
    # 각 지표는 평가 문항 중 통과한 비율입니다. 예: 10문항 중 8개 통과 → 80% (8/10).
    # 생성된 답변의 채점 결과를 요약한 값이며, 학습 중 기록한 loss나 모델의 확신도와는 다릅니다.
    rows = []
    for metric, (name, desc) in METRIC_LABELS.items():
        bars = []
        for mode, s in summaries.items():
            if not s:
                continue
            value = s.get(metric, 0) or 0
            color = MODE_COLORS.get(mode, "#0969da")
            hits = round(value * s.get("n", 0))
            bars.append(f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                        f'<span class="sub" style="width:76px">{_e(_mode_label(mode))}</span>'
                        f'<div class="track"><span class="bar" style="width:{100 * value:.0f}%;background:{color}"></span></div>'
                        f'<span style="width:92px;text-align:right;font-size:12.5px">{_pct(value)} ({hits}/{s.get("n", 0)})</span></div>')
        rows.append(f'<tr><td style="width:150px"><b>{name}</b><br><span class="sub">{desc}</span></td><td>{"".join(bars)}</td></tr>')
    return f'<div class="lv"><h3>{_e(title)}</h3><table>{"".join(rows)}</table></div>'


# ── 3단계: 학습 설정·LoRA 정보·학습 결과 ─────────────────────────────
def settings_html(lr, epochs, rank, train_count, batch=1, accumulation=4) -> str:
    """학습 변수 3개와, 그로부터 계산되는 실제 학습 스텝 수."""
    # batch개씩 처리한 기울기를 accumulation번 모은 뒤 한 번 업데이트합니다.
    # 여기서 스텝은 질문 번호가 아니라 가중치 업데이트 횟수입니다.
    steps_per_epoch = math.ceil(train_count / (batch * accumulation))
    total_steps = math.ceil(steps_per_epoch * epochs)
    return (f'<div class="lv"><h3>이번 학습 설정</h3><table>'
            f'<tr><th>변수</th><th>값</th><th>의미</th></tr>'
            f'<tr><td>LEARNING_RATE</td><td>{lr:g}</td><td>한 번 업데이트할 때 움직이는 크기</td></tr>'
            f'<tr><td>EPOCHS</td><td>{epochs:g}</td><td>학습 데이터 {train_count}개를 몇 바퀴 볼지</td></tr>'
            f'<tr><td>LORA_RANK</td><td>{rank}</td><td>추가 행렬 A·B의 폭 (alpha = {2 * rank})</td></tr></table>'
            f'<p class="sub">한 스텝 = 예시 {batch * accumulation}개 → epoch당 {steps_per_epoch}스텝 × {epochs:g} = '
            f'<b>총 {total_steps}번 업데이트</b></p></div>')


def lora_info_html(info: dict) -> str:
    """LoRA로 학습하는 파라미터가 전체에서 얼마나 되는지 보여 줍니다."""
    trainable, total = info["trainable_params"], info["total_params"]
    ratio = trainable / total if total else 0
    return (f'<div class="lv"><h3>LoRA 어댑터를 붙였습니다</h3><div class="grid">'
            f'<div class="label">학습할 파라미터</div><div><b>{trainable:,}개</b> '
            f'<span class="sub">/ 전체 {total:,}개 = {100 * ratio:.3f}%</span>'
            f'<div class="track"><span class="bar" style="width:{max(100 * ratio, 0.6):.2f}%;background:#ff7710"></span></div>'
            f'<span class="sub">주황 막대만 학습합니다. 나머지(회색) 기본 가중치는 고정됩니다.</span></div>'
            f'<div class="label">rank / alpha</div><div>r = {info["rank"]}, α = {info["alpha"]} → 보정 크기 α/r = {info["alpha"] / info["rank"]:g}</div>'
            f'<div class="label">붙인 층</div><div class="sub">{_e(", ".join(info["target_modules"]))}</div></div></div>')


def loss_points(history: list[dict]) -> tuple[list, list]:
    """Trainer 기록에서 (스텝, 학습 loss)와 (스텝, 검증 loss) 목록을 뽑습니다."""
    # 같은 기록 목록에서도 loss는 학습 데이터, eval_loss는 검증 데이터로 계산된 값입니다.
    # 현재 실습은 학습 loss를 매 업데이트마다, 검증 loss를 각 epoch가 끝날 때 기록합니다.
    # 그래프를 그릴 때 다시 평가하지 않고, 기록에 남아 있는 점만 가져옵니다.
    train = [(h["step"], float(h["loss"])) for h in history if "loss" in h and "step" in h]
    evals = [(h["step"], float(h["eval_loss"])) for h in history if "eval_loss" in h and "step" in h]
    return train, evals


def training_html(info: dict) -> str:
    """학습 결과 요약 카드: 시간·스텝·처음/마지막 loss·검증 loss 변화."""
    train, evals = loss_points(info.get("log_history", []))
    first = train[0][1] if train else None
    last = train[-1][1] if train else None
    eval_text = " → ".join(f"{v:.3f}" for _, v in evals) or "–"
    trend = ""
    # 처음과 마지막 기록의 방향만 보고 주의 문구를 보여 줍니다.
    # 검증 loss가 올랐다는 사실만으로 과적합을 확정하지 않고 실제 답변도 함께 비교합니다.
    if len(evals) >= 2 and evals[-1][1] > evals[0][1] and last is not None and first is not None and last < first:
        trend = ('<div class="warn">학습 loss는 내려갔는데 검증 loss는 올라갔습니다. 학습 예시에 지나치게 맞춰지는 '
                 '<b>과적합 신호</b>일 수 있어요. 점이 적으므로 실제 답변과 함께 판단하세요.</div>')
    return (f'<div class="lv"><h3>LoRA 학습 완료</h3><div class="grid">'
            f'<div class="label">학습 시간</div><div>{info.get("train_seconds", 0):.0f}초 · {info.get("steps", 0)}스텝 '
            f'<span class="sub">(정밀도 {_e(info.get("train_precision"))})</span></div>'
            f'<div class="label">학습 loss</div><div>{"–" if first is None else f"{first:.3f}"} → '
            f'<b>{"–" if last is None else f"{last:.3f}"}</b> <span class="sub">(처음 → 마지막 스텝)</span></div>'
            f'<div class="label">검증 loss</div><div>{eval_text} <span class="sub">(epoch마다)</span></div>'
            f'<div class="label">어댑터 저장</div><div class="sub">{_e(info.get("out"))}/adapter</div></div>{trend}</div>')


def plot_loss(history: list[dict], title: str = "LoRA training loss"):
    """학습 loss(매 스텝)와 검증 loss(epoch마다) 그래프. Colab에 한글 글꼴이 없어 축 이름은 영어로 씁니다."""
    import matplotlib.pyplot as plt

    train, evals = loss_points(history)
    # 두 곡선 모두 x축은 업데이트 스텝입니다. 검증 점은 epoch 종료 시점의 스텝에 놓입니다.
    # 검증 점 사이의 선은 흐름을 보기 위해 연결한 것이며, 중간마다 검증했다는 뜻은 아닙니다.
    # loss는 낮을수록 학습 목표에 가까워지지만, 정답률이 몇 %인지 직접 알려 주는 값은 아닙니다.
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    if train:
        ax.plot(*zip(*train), marker="o", markersize=3, color="#ff7710", label="train loss (each step)")
    if evals:
        ax.plot(*zip(*evals), marker="s", markersize=7, linestyle="--", color="#0969da", label="validation loss (each epoch)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (lower = closer to answer)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ── 5단계: 학습 전후 비교 ───────────────────────────────────────────
def _key(r: dict):
    # 학습 전후 목록의 순서가 달라도 같은 ID끼리 연결합니다.
    # ID가 없는 직접 입력은 질문과 문서 ID 목록을 함께 비교 키로 씁니다.
    return r.get("id") or (r.get("question"), tuple(r.get("context_ids") or []))


def comparison_html(base_results: list[dict], lora_results: list[dict], title: str) -> str:
    """같은 입력에 대한 기본 모델과 LoRA 모델의 출력을 나란히 놓고, 종합 성공 기준으로 변화를 표시합니다."""
    lora_by_key = {_key(r): r for r in lora_results}
    rows, counts = [], {"up": 0, "down": 0, "same": 0}
    for b in base_results:
        l = lora_by_key.get(_key(b))
        # 짝이 없는 입력은 비교에서 제외합니다. 비교 전에는 같은 문항을 평가했는지 확인하세요.
        if l is None:
            continue
        # 개선·악화는 종합 성공(형식·정답·근거·status 모두 통과)의 실패↔성공 변화입니다.
        # 답변 문장이 달라졌거나 일부 지표만 좋아져도 종합 성공이 같으면 '= 그대로'로 집계됩니다.
        bs, ls = b.get("grounded_success"), l.get("grounded_success")
        if bs is None or ls is None:
            change, cls = "정답 없음", "same"
        elif ls and not bs:
            change, cls = "▲ 개선", "up"
        elif bs and not ls:
            change, cls = "▼ 악화", "down"
        else:
            change, cls = ("= 둘 다 성공" if ls else "= 둘 다 실패"), "same"
        if cls in counts and bs is not None:
            counts[cls] += 1
        expected = _json(b["expected"]) if b.get("expected") is not None else "-"
        rows.append(f'<tr><td>{_e(b.get("id") or "-")}<br><span class="sub">{_e(b.get("question"))}</span></td>'
                    f'<td><pre>{_e(expected)}</pre></td>'
                    f'<td><pre>{_e(b.get("raw_output"))}</pre>{_mark(bs)}</td>'
                    f'<td><pre>{_e(l.get("raw_output"))}</pre>{_mark(ls)}</td>'
                    f'<td class="{cls}" style="white-space:nowrap">{change}</td></tr>')
    summary = (f'<p><span class="up">▲ 개선 {counts["up"]}</span> · <span class="down">▼ 악화 {counts["down"]}</span> · '
               f'<span class="same">= 그대로 {counts["same"]}</span></p>')
    return (f'<div class="lv"><h3>{_e(title)} · 학습 전후 비교</h3>{summary}<table><tr><th>질문</th><th>기대 정답</th>'
            f'<th>기본 모델 출력</th><th>LoRA 모델 출력</th><th>변화</th></tr>{"".join(rows)}</table></div>')


# ── 6단계: 실험 기록과 순위 ─────────────────────────────────────────
def experiments_html(rows: list[dict], ranked: list[dict] | None = None) -> str:
    """실험 기록표. 검증 실험은 순위대로, 최종 test 평가는 아래에 따로 보여 줍니다."""
    # 순서를 받지 않으면 검증 종합 성공이 높은 실험부터, 동률이면 검증 loss가 낮은 실험부터 정렬합니다.
    # 순위는 기록된 실험끼리의 비교입니다. test 결과는 설정 선택에 쓰지 않도록 별도 표에 둡니다.
    if ranked is None:
        ranked = sorted([r for r in rows if r.get("split") == "validation" and r.get("lora")],
                        key=lambda r: (-r["lora"]["grounded_success"],
                                       r["eval_loss"] if r.get("eval_loss") is not None else math.inf))
    if not rows:
        return '<div class="lv"><h3>실험 기록</h3><p class="sub">아직 기록된 실험이 없습니다.</p></div>'

    def line(i, r, best=False):
        lora, base = r.get("lora") or {}, r.get("base") or {}
        value = lora.get("grounded_success", 0) or 0
        eval_loss = "–" if r.get("eval_loss") is None else f'{r["eval_loss"]:.3f}'
        return (f'<tr class="{"best" if best else ""}"><td>{i}</td><td>{_e(r.get("label") or "-")}</td>'
                f'<td>{r["learning_rate"]:g}</td><td>{r["epochs"]:g}</td><td>{r["lora_rank"]}</td>'
                f'<td><div class="track" style="width:120px;display:inline-block"><span class="bar" '
                f'style="width:{100 * value:.0f}%;background:#ff7710"></span></div> {_pct(value)}</td>'
                f'<td>{_pct(lora.get("answer_correct"))}</td><td>{_pct(lora.get("source_correct"))}</td>'
                f'<td>{_pct(lora.get("context_policy_ok"))}</td>'
                f'<td>{eval_loss}</td>'
                f'<td>{_pct(base.get("grounded_success")) if base else "–"}</td>'
                f'<td>{(r.get("train_seconds") or 0):.0f}초</td></tr>')

    head = ('<tr><th>순위</th><th>메모</th><th>학습률</th><th>epoch</th><th>rank</th><th>종합 성공</th><th>정답</th>'
            '<th>근거</th><th>응답 정책</th><th>검증 loss</th><th>기본 모델</th><th>학습 시간</th></tr>')
    # 왼쪽 순위 1, 2, 3은 정렬된 행의 번호이며, LoRA 설정인 rank와는 다른 숫자입니다.
    # 모든 비교값이 같아도 공동 순위를 계산하지 않고 행 순서대로 번호를 붙입니다.
    body = "".join(line(i, r, best=(i == 1)) for i, r in enumerate(ranked, 1))
    tests = [r for r in rows if r.get("split") == "test"]
    test_body = "".join(line("test", r) for r in tests)
    test_table = (f'<h3 style="margin-top:14px">최종 평가(test)</h3><table>{head}{test_body}</table>' if tests else "")
    best = ranked[0] if ranked else None
    tip = (f'<p class="sub">현재 1위: 학습률 {best["learning_rate"]:g} · epoch {best["epochs"]:g} · rank {best["lora_rank"]} '
           f'(종합 성공이 같으면 검증 loss가 낮은 쪽을 위로). 문항이 10개라 한 문제 차이(10%)는 우연일 수 있어요.</p>'
           if best else "")
    return (f'<div class="lv"><h3>실험 기록 · 검증(validation) 순위</h3>{tip}<table>{head}{body}</table>{test_table}</div>')


# ── 6단계 과제: 나만의 출력 학습 ────────────────────────────────────
def _similarity_bar(value) -> str:
    value = value or 0
    color = "#1a7f37" if value >= 0.999 else "#ff7710" if value >= 0.6 else "#8b95a1"
    return (f'<div class="track" style="width:110px;display:inline-block"><span class="bar" '
            f'style="width:{100 * value:.0f}%;background:{color}"></span></div> <span class="sub">{100 * value:.0f}%</span>')


def custom_task_html(task: dict) -> str:
    """학생이 쓴 문장과, 그 문장이 학습용·시험용으로 어떻게 나뉘었는지 보여 줍니다."""
    rows = "".join(
        f'<tr><td>{no}</td><td><b>{_e(v["출력"])}</b></td><td>{_e(v["질문_a"])}</td><td>{_e(v["질문_b"])}</td>'
        f'<td>{_e(v["질문_c"])}</td></tr>' for no, v in enumerate(task["items"], 1))
    unrelated = " · ".join(_e(r["question"]) for r in task["unrelated"])
    return (f'<div class="lv"><h3>나만의 출력 학습 데이터</h3>'
            f'<p class="sub">학습 예시 {len(task["train"])}개(질문 a·b) · 처음 보는 질문 {len(task["test"])}개(질문 c, 학습에 안 씀) · '
            f'관계없는 질문 {len(task["unrelated"])}개</p><table><tr><th>#</th><th>원하는 출력</th><th>질문 a (학습)</th>'
            f'<th>질문 b (학습)</th><th>질문 c (시험)</th></tr>{rows}</table>'
            f'<p class="sub">관계없는 질문: {unrelated}</p></div>')


def _custom_scores(results: list[dict]) -> dict:
    # 이미 채점된 exact·leaked를 질문 종류별로 모아 비율을 냅니다.
    # '관계없는 질문 보존'은 내 문장이 섞이지 않았다는 뜻이며, 그 답변의 사실 정확성을 채점한 값은 아닙니다.
    def ratio(rows, ok):
        return (sum(ok(r) for r in rows) / len(rows), len(rows)) if rows else (None, 0)
    return {
        "학습 질문 정확 일치": ratio([r for r in results if r["kind"] == "학습"], lambda r: r["exact"]),
        "처음 보는 질문 정확 일치": ratio([r for r in results if r["kind"] == "시험"], lambda r: r["exact"]),
        "관계없는 질문 보존": ratio([r for r in results if r["kind"] == "관계없음"], lambda r: not r["leaked"]),
    }


def custom_result_html(experiment: dict) -> str:
    """과제 실험 결과: 점수 막대(기본 모델 vs LoRA) + 질문별 출력 비교."""
    # 위 막대는 종류별 통과 비율이고, 아래 행의 '비슷한 정도'는 각 출력의 문자열 유사도입니다.
    # 유사도가 높아도 정확 일치에 실패할 수 있으므로 ✔ 표시와 함께 읽습니다.
    lora, base = experiment["results"], experiment.get("base_results") or []
    lora_scores, base_scores = _custom_scores(lora), _custom_scores(base)
    bars = []
    for name, (value, n) in lora_scores.items():
        lines = []
        for mode, (v, m) in (("base", base_scores[name]), ("lora", (value, n))):
            if v is None:
                continue
            lines.append(f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                         f'<span class="sub" style="width:76px">{MODE_LABELS[mode]}</span>'
                         f'<div class="track"><span class="bar" style="width:{100 * v:.0f}%;background:{MODE_COLORS[mode]}"></span></div>'
                         f'<span style="width:92px;text-align:right;font-size:12.5px">{_pct(v)} ({round(v * m)}/{m})</span></div>')
        bars.append(f'<tr><td style="width:170px"><b>{name}</b></td><td>{"".join(lines)}</td></tr>')
    base_by_q = {r["question"]: r for r in base}
    rows = []
    for r in lora:
        b = base_by_q.get(r["question"])
        if r["kind"] == "관계없음":
            verdict = '<span class="ok">✔ 보존</span>' if not r["leaked"] else '<span class="no">✘ 내 문장이 섞임</span>'
            target, score = '<span class="na">(관계없는 질문)</span>', verdict
        else:
            target = _e(r["target"])
            score = f'{_mark(r["exact"])} {_similarity_bar(r["similarity"])}'
        rows.append(f'<tr><td>{_e(r["kind"])}{" " + str(r["no"]) if r.get("no") else ""}</td><td>{_e(r["question"])}</td>'
                    f'<td>{target}</td><td><pre>{_e(b["raw_output"]) if b else "–"}</pre></td>'
                    f'<td><pre>{_e(r["raw_output"])}</pre></td><td style="white-space:nowrap">{score}</td></tr>')
    title = (f'과제 결과 · 학습률 {experiment["learning_rate"]:g} · epoch {experiment["epochs"]:g} · rank {experiment["lora_rank"]}'
             f' · {experiment.get("steps", 0)}스텝')
    return (f'<div class="lv"><h3>{_e(title)}</h3><table>{"".join(bars)}</table>'
            f'<table style="margin-top:12px"><tr><th>종류</th><th>질문</th><th>원하는 출력</th><th>기본 모델</th>'
            f'<th>LoRA 모델</th><th>정확 일치 · 비슷한 정도</th></tr>{"".join(rows)}</table>'
            f'<p class="sub">정확 일치는 앞뒤 공백·여러 칸 띄어쓰기만 무시하고 글자가 모두 같아야 ✔입니다. '
            f'관계없는 질문은 내 문장과 60% 이상 비슷하거나 내 문장을 포함하면 "섞임"으로 봅니다. '
            f'기본 모델 칸은 같은 문장으로 처음 실험할 때 한 번 기록한 답입니다.</p></div>')


def custom_experiments_html(rows: list[dict], ranked: list[dict] | None = None) -> str:
    """과제 실험 순위표(같은 문장으로 한 실험끼리)."""
    # 과제 순서는 train.rank_custom_experiments()에서 받은 ranked를 그대로 사용합니다.
    # ranked를 생략하면 입력 순서가 표시되므로, 순위 비교에는 정렬된 목록을 넘겨야 합니다.
    # 순위의 1, 2, 3은 행 번호입니다. '처음 보는 질문 정확'의 %는 질문 c의 정확 일치 비율입니다.
    if not rows:
        return '<div class="lv"><h3>과제 실험 기록</h3><p class="sub">아직 이 문장으로 한 실험이 없습니다.</p></div>'
    ranked = ranked if ranked is not None else rows
    lines = []
    for i, r in enumerate(ranked, 1):
        s = r.get("summary") or {}
        lines.append(f'<tr class="{"best" if i == 1 else ""}"><td>{i}</td><td>{_e(r.get("label") or "-")}</td>'
                     f'<td>{r["learning_rate"]:g}</td><td>{r["epochs"]:g}</td><td>{r["lora_rank"]}</td><td>{r.get("steps", "–")}</td>'
                     f'<td>{_similarity_bar(s.get("test_exact"))}</td><td>{_pct(s.get("unrelated_ok"))}</td>'
                     f'<td>{_pct(s.get("train_exact"))}</td><td>{_pct(s.get("test_similarity"))}</td>'
                     f'<td>{(r.get("train_seconds") or 0):.0f}초</td></tr>')
    best = ranked[0]
    tip = (f'<p class="sub">현재 1위: 학습률 {best["learning_rate"]:g} · epoch {best["epochs"]:g} · rank {best["lora_rank"]}. '
           f'순위 = ① 처음 보는 질문 정확 일치 ② 관계없는 질문 보존 ③ 학습 질문 정확 일치 ④ 스텝이 적을수록 ⑤ 시간이 짧을수록</p>')
    return (f'<div class="lv"><h3>과제 실험 순위 · 같은 문장으로 한 실험 {len(rows)}개</h3>{tip}<table>'
            f'<tr><th>순위</th><th>메모</th><th>학습률</th><th>epoch</th><th>rank</th><th>스텝</th><th>처음 보는 질문 정확</th>'
            f'<th>관계없는 질문 보존</th><th>학습 질문 정확</th><th>처음 보는 질문 비슷한 정도</th><th>학습 시간</th></tr>'
            f'{"".join(lines)}</table></div>')


def plain_answer_html(r: dict) -> str:
    """6-4 자유 질문 카드."""
    return (f'<div class="lv"><h3>LoRA 모델 답변 · 자유 질문</h3><div class="grid">'
            f'<div class="label">입력 · 질문</div><div><pre>{_e(r["question"])}</pre></div>'
            f'<div class="label">출력</div><div><pre>{_e(r["raw_output"])}</pre></div>'
            f'<div class="label">생성</div><div class="sub">{r.get("generated_tokens", 0)} 토큰 · '
            f'{r.get("generation_seconds", 0):.1f}초 · 문서 없이 질문만 넣었습니다</div></div></div>')


# ── 화면에 표시하는 함수들(노트북에서 호출) ──────────────────────────
# show_*()에는 앞서 얻은 결과를 넘깁니다. 표시한 카드를 변수에 저장하는 반환 함수는 아닙니다.
# 같은 결과로 show_answer_card()와 show_results_table()을 모두 호출해도 추가 추론은 없습니다.
def show_custom_task(task):
    # 학생이 만든 출력 문장과 학습용·시험용 질문 구성을 보여 줍니다.
    _display(custom_task_html(task))


def show_custom_result(experiment):
    """과제 결과 표 + loss 그래프."""
    _display(custom_result_html(experiment))
    if experiment.get("train", {}).get("log_history"):
        import matplotlib.pyplot as plt
        plot_loss(experiment["train"]["log_history"], "My-sentence LoRA loss (validation = question c)")
        plt.show()


def show_custom_experiments(rows, ranked=None):
    # 같은 문장으로 진행한 과제 실험의 설정과 점수를 나란히 보여 줍니다.
    _display(custom_experiments_html(rows, ranked))


def show_plain_answer(result):
    # 자유 질문·답변 원문·생성량·시간을 표시합니다. 기대 정답이 없어 점수표는 만들지 않습니다.
    _display(plain_answer_html(result))

def show_data_overview(docs, splits):
    # train·validation·test의 문항 수와 쓰임을 확인하는 표입니다.
    _display(data_overview_html(docs, splits))


def show_model_info(info):
    # 이미 불러온 모델의 이름·정밀도·메모리 정보를 보여 줍니다. 여기서 모델을 불러오지는 않습니다.
    _display(model_info_html(info))


def show_answer_card(result):
    _display(answer_card_html(result))


def show_results_table(results, title):
    _display(results_table_html(results, title))


def show_metrics(summaries, title="점수 요약"):
    # 저장된 채점 비율을 % 막대로 보여 줍니다. loss 그래프는 show_training()에서 표시합니다.
    _display(metrics_html(summaries, title))


def show_settings(lr, epochs, rank, train_count):
    _display(settings_html(lr, epochs, rank, train_count))


def show_lora_info(info):
    # 전체 파라미터 중 학습 가능한 비율과 어댑터가 붙은 층을 확인합니다.
    _display(lora_info_html(info))


def show_training(info, title="LoRA training loss"):
    """학습 요약 카드 + loss 그래프."""
    # 학습이 끝난 뒤 받은 기록을 그립니다. 이 함수를 다시 실행해도 추가 학습은 하지 않습니다.
    _display(training_html(info))
    if info.get("log_history"):
        import matplotlib.pyplot as plt
        plot_loss(info["log_history"], title)
        plt.show()


def show_comparison(base_results, lora_results, title):
    _display(comparison_html(base_results, lora_results, title))


def show_experiments(rows, ranked=None):
    _display(experiments_html(rows, ranked))


def show_gpu():
    """현재 GPU 메모리 사용량 막대."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info(0)
    except Exception:  # GPU가 없거나 torch가 없으면 건너뜀
        return
    gib = 1024 ** 3
    used = (total - free) / gib
    _display(f'<div class="lv"><h3>GPU 메모리</h3><div class="track"><span class="bar" '
             f'style="width:{100 * used / (total / gib):.0f}%;background:#0969da"></span></div>'
             f'<span class="sub">{used:.1f} / {total / gib:.1f} GiB 사용 중</span></div>')
