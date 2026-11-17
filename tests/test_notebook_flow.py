"""노트북 2~6단계 흐름을 가짜 모델로 확인합니다(GPU·모델 다운로드 없이 연결 오류만 잡는 테스트)."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ASSETS = Path(__file__).resolve().parents[1] / "workshop-assets"
sys.path.insert(0, str(ASSETS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ASSETS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load("workshop_train_flow", "train.py")
view = load("workshop_view", "lab_view.py")


class FakeIds:
    shape = (1, 20)


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeGenerated:
    def __getitem__(self, key):
        return [1, 2, 3]


class AnswerKeyTokenizer:
    """질문을 보고 정답 JSON(또는 일부러 틀린 답)을 돌려주는 가짜 토크나이저."""
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, splits, wrong_ids=()):
        self.answers = {r["question"]: r["expected"] for rows in splits.values() for r in rows}
        self.wrong_ids = set(wrong_ids)
        self.ids = {r["question"]: r["id"] for rows in splits.values() for r in rows}
        self.last_question = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_question = messages[1]["content"].rsplit("질문: ", 1)[1]
        return messages[1]["content"]

    def __call__(self, text, **kwargs):
        return FakeInputs(input_ids=FakeIds())

    def decode(self, tokens, **kwargs):
        q = self.last_question
        if q not in self.answers:
            return "자유 입력에 대한 설명문"
        if self.ids[q] in self.wrong_ids:
            return '{"answer":"모름","evidence":["[X] 틀림"],"status":"answered"}'
        return json.dumps(self.answers[q], ensure_ascii=False)


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_calls = 0

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.generate_calls += 1
        return FakeGenerated()


class NotebookFlowTests(unittest.TestCase):
    def setUp(self):
        self.docs, self.splits = train.load_data()
        self.tmp = tempfile.TemporaryDirectory()
        self._outputs = train.OUTPUTS
        train.OUTPUTS = Path(self.tmp.name) / "outputs"

    def tearDown(self):
        train.OUTPUTS = self._outputs
        self.tmp.cleanup()

    def test_guide_matches_validation_rows(self):
        guide = train.load_guide()
        self.assertEqual(len(guide), 5)
        rows = {r["id"]: r for r in self.splits["validation"]}
        for g in guide:
            row = rows[g["row_id"]]  # 가이드 입력은 모두 학습에 쓰지 않은 validation 문항
            self.assertEqual(g["question"], row["question"])
            self.assertEqual([g["document_id"]], row["context_ids"])
            self.assertEqual(g["document"], row["document"])
            self.assertEqual(g["expected"], row["expected"])

    def test_ask_scores_dataset_questions_and_shows_free_input(self):
        tok = AnswerKeyTokenizer(self.splits)
        r = train.ask(FakeModel(), tok, "14기 총무는 누구야?", "V03", mode="base")
        self.assertEqual(r["id"], "V03-Q1")
        self.assertTrue(r["grounded_success"])
        self.assertEqual(r["dataset_split"], "validation")
        free = train.ask(FakeModel(), tok, "아무 질문", "직접 입력", "내가 쓴 문서", mode="base")
        self.assertIsNone(free["expected"])
        self.assertFalse(free["format_ok"])
        self.assertIn("정답 없음", view.answer_card_html(free))
        trained = train.ask(FakeModel(), tok, "14기 대표는 누구야?", "T11", mode="lora")
        self.assertIn("학습(train)에 들어 있던 문항", view.answer_card_html(trained))

    def test_full_stepwise_flow_renders_and_records(self):
        base_tok = AnswerKeyTokenizer(self.splits, wrong_ids={"V03-Q1", "V01-Q1", "V01-U1"})
        lora_tok = AnswerKeyTokenizer(self.splits, wrong_ids={"V02-Q1"})
        base_guide = train.ask_guide(FakeModel(), base_tok, mode="base")
        lora_guide = train.ask_guide(FakeModel(), lora_tok, mode="lora")
        base_val = train.evaluate(FakeModel(), base_tok, mode="base")
        lora_val = train.evaluate(FakeModel(), lora_tok, mode="lora")
        self.assertEqual(len(base_val), len(self.splits["validation"]))
        summary = train.summarize(base_val + lora_val)
        self.assertEqual(list(summary), ["base", "lora"])
        self.assertAlmostEqual(summary["lora"]["grounded_success"], 0.9)

        compare = view.comparison_html(base_guide, lora_guide, "가이드")
        self.assertIn("▲ 개선 3", compare)
        self.assertIn("▼ 악화 1", compare)
        self.assertIn("LoRA 모델", view.metrics_html(summary))

        history = [{"loss": 0.4, "step": 1}, {"loss": 0.05, "step": 9}, {"eval_loss": 0.12, "step": 9},
                   {"loss": 0.01, "step": 18}, {"eval_loss": 0.19, "step": 18}]
        train_info = {"log_history": history, "train_seconds": 120.0, "steps": 18, "train_precision": "fp32",
                      "train_loss": 0.01, "eval_loss": 0.19, "out": "x"}
        self.assertIn("과적합 신호", view.training_html(train_info))
        lora_info = {"rank": 8, "alpha": 16, "trainable_params": 5_000_000, "total_params": 2_000_000_000,
                     "target_modules": train.LORA_TARGET_MODULES}
        self.assertIn("0.250%", view.lora_info_html(lora_info))
        self.assertIn("총 18번 업데이트", view.settings_html(2e-4, 2, 8, len(self.splits["train"])))

        saved = train.save_session(base_val, lora_val, train_info, lora_info, base_guide=base_guide, lora_guide=lora_guide)
        for name in ("comparison.csv", "summary.json", "config.json", "guide_answers.json", "loss_history.json"):
            self.assertTrue((saved / name).exists(), name)
        rows = train.load_experiments()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "3~5단계 기본 설정")

    def test_experiment_ranking_prefers_success_then_lower_eval_loss(self):
        def row(label, success, eval_loss, split="validation"):
            return {"label": label, "split": split, "learning_rate": 2e-4, "epochs": 2, "lora_rank": 8,
                    "eval_loss": eval_loss, "train_seconds": 60, "lora": {"grounded_success": success}, "base": None}
        rows = [row("a", 0.7, 0.10), row("b", 0.9, 0.30), row("c", 0.9, 0.20), row("t", 1.0, 0.01, "test")]
        ranked = train.rank_experiments(rows)
        self.assertEqual([r["label"] for r in ranked], ["c", "b", "a"])  # test는 순위에서 제외
        html = view.experiments_html(rows, ranked)
        self.assertIn("최종 평가(test)", html)
        self.assertIn('class="best"', html)

    def test_notebook_cells_compile_and_call_existing_functions(self):
        nb = json.loads((ASSETS / "lora_workshop.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(code):
            compile(re.sub(r"^(\s*)[!%].*$", r"\1pass", src, flags=re.M), f"cell{i}", "exec")
        text = "\n".join(code)
        for name in set(re.findall(r"\blab\.(\w+)", text)):
            self.assertTrue(hasattr(train, name), f"train.{name} 없음")
        for name in set(re.findall(r"\bview\.(\w+)", text)):
            self.assertTrue(hasattr(view, name), f"lab_view.{name} 없음")
        headings = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        for step in range(1, 7):
            self.assertRegex(headings, rf"##\s*{step}단계\s*·")  # 웹(LessonDetails.jsx)이 이 제목으로 단계를 나눕니다
        # 실습 파일은 1-2 셀에서 ZIP을 직접 올리는 방식 하나만 씁니다.
        self.assertEqual(sum("files.upload()" in src for src in code), 1)
        self.assertFalse(any("wget" in src for src in code))
        self.assertTrue(all("#@title" in src for src in code), "웹 코드 블록 제목에 #@title을 씁니다")

    def test_required_evaluation_cells_generate_once_and_reuse_guide_results(self):
        """통합 셀을 실제 실행하되 모델 로딩·출력만 대체합니다. GPU 학습 검증은 아닙니다."""
        nb = json.loads((ASSETS / "lora_workshop.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        guide_ids = [g["row_id"] for g in train.load_guide()]
        validation_ids = [row["id"] for row in self.splits["validation"]]
        self.assertEqual(len(guide_ids), 5)
        self.assertEqual(len(validation_ids), 10)
        model = FakeModel()
        tokenizer = AnswerKeyTokenizer(self.splits)
        env = {"lab": train, "view": view}

        with patch.object(train, "load_base_model", return_value=(model, tokenizer, {"model": "fake-cpu-model"})) as loader:
            for prefix, mode in (("2-1", "base"), ("4-1", "lora")):
                with self.subTest(cell=prefix, mode=mode):
                    cells = [src for src in code if re.match(rf"#@title {prefix}(?:\.|\s)", src)]
                    self.assertEqual(len(cells), 1, "각 단계의 필수 셀은 하나여야 합니다")
                    self.assertNotIn("#@param", cells[0])
                    if mode == "lora":
                        # GPU 학습 대신 완료 상태만 모사합니다. 4-1은 이 모델을 그대로 써야 합니다.
                        env["train_info"] = {"fake_training": True}
                    generated_before = model.generate_calls
                    with (patch.object(train, "evaluate", wraps=train.evaluate) as evaluate,
                          patch.object(view, "_display") as display,
                          patch.object(view, "show_answer_card", wraps=view.show_answer_card) as cards,
                          patch.object(view, "show_results_table", wraps=view.show_results_table) as tables,
                          patch.object(view, "show_metrics", wraps=view.show_metrics) as metrics):
                        exec(compile(cells[0], f"notebook-{prefix}", "exec"), env)

                        loader.assert_called_once_with()  # 4-1에서 기본 모델을 다시 불러오지 않음
                        self.assertIs(env["model"], model)
                        self.assertIs(env["tokenizer"], tokenizer)
                        evaluate.assert_called_once_with(model, tokenizer, split="validation", mode=mode)
                        self.assertEqual(model.generate_calls - generated_before, 10)
                        validation = env[f"{mode}_val"]
                        guide = env[f"{mode}_guide"]
                        self.assertEqual([row["id"] for row in validation], validation_ids)
                        self.assertTrue(all(row["mode"] == mode for row in validation))
                        self.assertTrue(all(row["dataset_split"] == "validation" for row in validation))
                        self.assertEqual([row["id"] for row in guide], guide_ids)
                        by_id = {row["id"]: row for row in validation}
                        for row in guide:
                            self.assertIs(row, by_id[row["id"]], "가이드가 검증 결과를 복사하거나 다시 생성했습니다")
                            self.assertEqual(json.loads(row["raw_output"]), tokenizer.answers[row["question"]])
                        self.assertEqual(cards.call_count, 5)
                        for call, row in zip(cards.call_args_list, guide):
                            self.assertIs(call.args[0], row)
                        self.assertTrue(any(call.args[0] is guide for call in tables.call_args_list))
                        self.assertTrue(any(call.args[0] is validation for call in tables.call_args_list))
                        metrics.assert_called_once()
                        self.assertEqual(metrics.call_args.args[0], train.summarize(validation))
                        self.assertTrue(display.called)  # 실제 HTML 렌더링 경로도 실행됨
                    if mode == "base":
                        base_snapshot = json.dumps(env["base_val"], sort_keys=True)
                    else:
                        self.assertEqual(json.dumps(env["base_val"], sort_keys=True), base_snapshot)

    def test_skipped_cells_explain_which_cell_to_run_first(self):
        nb = json.loads((ASSETS / "lora_workshop.ipynb").read_text(encoding="utf-8"))
        code = {"".join(c["source"]).splitlines()[0]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
        cases = [("#@title 2-1", {}, "1-4"),                                    # 세션 재시작 직후
                 ("#@title 2-2", {"lab": object(), "view": object()}, "2-1"),  # 기본 모델을 안 불러옴
                 ("#@title 4-1", {"lab": object(), "view": object(), "model": object(),
                                   "tokenizer": object()}, "3-3"),             # 학습을 안 함
                 ("#@title 5-1", {"lab": object(), "base_guide": []}, "4-1")]
        for head, env, expected in cases:
            src = next(v for k, v in code.items() if k.startswith(head))
            with self.subTest(cell=head), self.assertRaisesRegex(RuntimeError, f"먼저 {expected}"):
                exec(compile(re.sub(r"^(\s*)[!%].*$", r"\1pass", src, flags=re.M), head, "exec"), dict(env))


EXAMPLE_ITEMS = [
    {"출력": "멋사 중부대 14기 대표는 백서진입니다.", "질문_a": "14기 대표가 누구야?",
     "질문_b": "멋사 중부대 대표 알려줘", "질문_c": "이번 기수 대표는 누구니?"},
    {"출력": "정규 세미나는 화요일 오후 7시에 열려요.", "질문_a": "세미나 언제야?",
     "질문_b": "정규 세미나 시간 알려줘", "질문_c": "세미나는 무슨 요일 몇 시에 해?"},
    {"출력": "총무는 윤주혁입니다.", "질문_a": "총무 누구야?", "질문_b": "14기 총무 알려줘", "질문_c": "돈 관리는 누가 해?"},
]


class PlainTokenizer:
    """과제용 가짜 토크나이저: 질문별로 정해 둔 답을 돌려줍니다."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def __init__(self, answers):
        self.answers = answers
        self.last = None

    def apply_chat_template(self, messages, **kwargs):
        self.last = messages[-1]["content"]
        return self.last

    def __call__(self, text, **kwargs):
        return FakeInputs(input_ids=FakeIds())

    def decode(self, tokens, **kwargs):
        return self.answers.get(self.last, "잘 모르겠어요.")


class CustomTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._outputs = train.OUTPUTS
        train.OUTPUTS = Path(self.tmp.name) / "outputs"

    def tearDown(self):
        train.OUTPUTS = self._outputs
        self.tmp.cleanup()

    def test_task_splits_questions_into_train_and_unseen_test(self):
        task = train.make_custom_task(EXAMPLE_ITEMS, ["파이썬은 어떤 언어야?"])
        self.assertEqual(len(task["train"]), 6)   # 질문 a·b × 3
        self.assertEqual([r["question"] for r in task["test"]], [i["질문_c"] for i in EXAMPLE_ITEMS])
        self.assertFalse({r["question"] for r in task["test"]} & {r["question"] for r in task["train"]})
        self.assertIn("질문 c (시험)", view.custom_task_html(task))

    def test_friendly_errors_for_student_input(self):
        blank = [dict(EXAMPLE_ITEMS[0]), {"출력": "____", "질문_a": "a", "질문_b": "b", "질문_c": "c"}]
        with self.assertRaisesRegex(ValueError, "2번 문장의 '출력' 빈칸"):
            train.make_custom_task(blank, ["x"])
        dup = [dict(EXAMPLE_ITEMS[0], 질문_c="14기 대표가 누구야?")]
        with self.assertRaisesRegex(ValueError, "같은 질문"):
            train.make_custom_task(dup, ["x"])
        with self.assertRaisesRegex(ValueError, "80자 이하"):
            train.make_custom_task([dict(EXAMPLE_ITEMS[0], 출력="가" * 81)], ["x"])
        with self.assertRaisesRegex(ValueError, "칸이 없습니다"):
            train.make_custom_task([{"출력": "a", "질문_a": "b"}], ["x"])

    def test_scoring_exact_similarity_and_leak(self):
        task = train.make_custom_task(EXAMPLE_ITEMS, ["파이썬은 어떤 언어야?", "1 더하기 1은?"])
        answers = {r["question"]: r["target"] for r in task["train"]}           # 학습 질문은 모두 정확
        answers["이번 기수 대표는 누구니?"] = "멋사 중부대 14기 대표는  백서진입니다. "  # 띄어쓰기만 다름 → 정확
        answers["세미나는 무슨 요일 몇 시에 해?"] = "정규 세미나는 화요일에 열려요."       # 비슷하지만 다름
        answers["파이썬은 어떤 언어야?"] = "총무는 윤주혁입니다."                     # 관계없는 질문에 내 문장 → 섞임
        tok = PlainTokenizer(answers)
        results = train.custom_answers(FakeModel(), tok, task)
        s = train.custom_summary(results)
        self.assertEqual(s["train_exact"], 1.0)
        self.assertAlmostEqual(s["test_exact"], 1 / 3)
        self.assertEqual(s["unrelated_ok"], 0.5)
        near = next(r for r in results if r["question"] == "세미나는 무슨 요일 몇 시에 해?")
        self.assertFalse(near["exact"])
        self.assertGreater(near["similarity"], 0.6)
        experiment = {"results": results, "base_results": train.custom_answers(FakeModel(), PlainTokenizer({}), task,
                                                                                kinds=("시험", "관계없음"), mode="base"),
                      "learning_rate": 5e-4, "epochs": 10, "lora_rank": 8, "steps": 60}
        html = view.custom_result_html(experiment)
        self.assertIn("✘ 내 문장이 섞임", html)
        self.assertIn("처음 보는 질문 정확 일치", html)

    def test_ranking_prefers_unseen_exact_then_preserved_then_less_training(self):
        def row(label, test, unrelated, steps):
            return {"label": label, "learning_rate": 2e-4, "epochs": 2, "lora_rank": 8, "steps": steps,
                    "train_seconds": steps, "summary": {"test_exact": test, "unrelated_ok": unrelated, "train_exact": 1}}
        ranked = train.rank_custom_experiments([row("과적합", 1, 0.5, 120), row("부족", 1 / 3, 1, 12),
                                                row("최적", 1, 1, 60), row("느림", 1, 1, 120)])
        self.assertEqual([r["label"] for r in ranked], ["최적", "느림", "과적합", "부족"])
        self.assertIn('class="best"', view.custom_experiments_html(ranked, ranked))

    def test_notebook_example_cell_asks_students_to_fill_blanks(self):
        nb = json.loads((ASSETS / "lora_workshop.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code" and c["source"][0].startswith("#@title 6-1"))
        self.assertIn('"____"', cell)
        self.assertNotIn("#@param", cell)  # 입력 칸이 아니라 코드 빈칸을 직접 채우는 방식
        with self.assertRaisesRegex(ValueError, "빈칸"):  # 그대로 실행하면 빈칸을 채우라는 안내
            exec(cell, {"lab": train, "view": view})


if __name__ == "__main__":
    unittest.main()
