"""CPU-only checks for the LoRA comparison protocol, without model execution."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "workshop-assets" / "train.py"
spec = importlib.util.spec_from_file_location("workshop_train", SCRIPT)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class InputIds:
    shape = (1, 12)


class Inputs(dict):
    def to(self, device):
        return self


class Generated:
    def __getitem__(self, key):
        return [101, 102]


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, output):
        self.output = output
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages)
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **kwargs):
        return Inputs(input_ids=InputIds())

    def decode(self, tokens, **kwargs):
        return self.output


class Model:
    device = "test"

    def eval(self):
        pass

    def generate(self, **kwargs):
        return Generated()


class Torch:
    inference_mode = staticmethod(contextlib.nullcontext)


class LoRAComparisonTests(unittest.TestCase):
    def setUp(self):
        _, self.splits = train.load_data()
        self.row = next(r for r in self.splits["validation"] if r["answerable"])

    def test_summary_contains_only_base_and_lora(self):
        score = train.score_output(json.dumps(self.row["expected"]), self.row, self.row["context_ids"])
        results = [{"mode": mode, **score} for mode in ("base", "lora")]
        with tempfile.TemporaryDirectory() as folder:
            train.save_results(Path(folder), results)
            summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(summary), {"base", "lora"})
        self.assertEqual(summary["base"]["n"], 1)
        self.assertEqual(summary["lora"]["grounded_success"], 1)

    def test_both_phases_use_the_same_preprovided_document(self):
        tokenizer = Tokenizer(json.dumps(self.row["expected"], ensure_ascii=False))
        results = []
        with tempfile.TemporaryDirectory() as folder:
            for mode in ("base", "lora"):
                train.evaluate_model(Model(), tokenizer, Torch, [self.row], mode, results, Path(folder))
            saved = json.loads((Path(folder) / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(tokenizer.prompts[0], tokenizer.prompts[1])
        self.assertIn(self.row["document"], tokenizer.prompts[0][1]["content"])
        self.assertIn(self.row["question"], tokenizer.prompts[0][1]["content"])
        self.assertEqual([r["mode"] for r in saved], ["base", "lora"])
        for result in saved:
            self.assertEqual(result["context_ids"], self.row["context_ids"])
            self.assertNotIn("retrieved_ids", result)
            self.assertTrue(result["grounded_success"])
            self.assertEqual(result["raw_output"], tokenizer.output)

    def test_information_missing_from_provided_document_requires_abstention(self):
        row = next(r for r in self.splits["validation"] if not r["answerable"])
        score = train.score_output(json.dumps(train.UNKNOWN), row, row["context_ids"])
        self.assertTrue(score["context_policy_ok"])
        self.assertTrue(score["grounded_success"])
        self.assertFalse(score["context_supports_answer"])

    def test_answer_without_provided_evidence_is_not_grounded(self):
        score = train.score_output(json.dumps(self.row["expected"]), self.row, [])
        self.assertTrue(score["answer_correct"])
        self.assertFalse(score["grounded_success"])
        self.assertFalse(score["context_policy_ok"])

    def test_multimodal_processor_is_unwrapped_to_text_tokenizer(self):
        class TextTokenizer(Tokenizer):
            chat_template = None

        class Processor:
            # 실제 Processor처럼 eos_token을 노출하고, 첫 위치 인자를 이미지로 해석합니다.
            eos_token = "<|im_end|>"
            chat_template = "{{ messages }}"

            def __init__(self):
                self.tokenizer = TextTokenizer("{}")

            def __call__(self, images=None, text=None, **kwargs):
                raise ValueError("Incorrect image source")

        processor = Processor()
        tokenizer = train.text_tokenizer(processor)
        self.assertIs(tokenizer, processor.tokenizer)
        self.assertEqual(tokenizer.chat_template, "{{ messages }}")
        plain = Tokenizer("{}")
        self.assertIs(train.text_tokenizer(plain), plain)

    def test_training_precision_follows_loaded_dtype(self):
        class FakeTorch:
            bfloat16, float16, float32 = "bf16", "fp16", "fp32"

        self.assertEqual(train.training_precision("bf16", FakeTorch), {"bf16": True, "fp16": False})
        self.assertEqual(train.training_precision("fp16", FakeTorch), {"bf16": False, "fp16": True})
        # T4에서 Unsloth가 Qwen3.5를 float32로 올린 경우: 혼합 정밀도를 켜지 않습니다.
        self.assertEqual(train.training_precision("fp32", FakeTorch), {"bf16": False, "fp16": False})

    def test_cli_options_override_file_defaults_one_at_a_time(self):
        parser = train.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual((defaults.lr, defaults.epochs, defaults.rank),
                         (train.LEARNING_RATE, train.EPOCHS, train.LORA_RANK))
        changed = parser.parse_args(["--lr", "1e-4"])
        self.assertEqual((changed.lr, changed.epochs, changed.rank), (1e-4, train.EPOCHS, train.LORA_RANK))
        self.assertEqual(parser.parse_args(["--rank", "16", "--final-test"]).rank, 16)

    def test_invalid_settings_stop_before_training(self):
        train.validate_settings(2e-4, 2, 8)
        for bad in [(0, 2, 8), (2e-4, 0, 8), (2e-4, 2, 0), (2e-4, 2, 256)]:
            with self.assertRaises(ValueError):
                train.validate_settings(*bad)

    def test_oversized_input_stops_instead_of_truncating(self):
        class LongTokenizer(Tokenizer):
            def __call__(self, text, **kwargs):
                ids = InputIds()
                ids.shape = (1, train.MAX_SEQUENCE_LENGTH)
                return Inputs(input_ids=ids)

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "평가 입력이 너무 깁니다"):
                train.evaluate_model(Model(), LongTokenizer("{}"), Torch, [self.row], "base", [], Path(folder))


if __name__ == "__main__":
    unittest.main()
