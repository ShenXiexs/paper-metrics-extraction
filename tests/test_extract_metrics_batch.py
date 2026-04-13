from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "extract_metrics_batch.py"
SPEC = importlib.util.spec_from_file_location("extract_metrics_batch", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["extract_metrics_batch"] = MODULE
SPEC.loader.exec_module(MODULE)


class ExtractMetricsBatchTests(unittest.TestCase):
    def test_parse_filename_metadata_standard(self) -> None:
        metadata = MODULE.parse_filename_metadata(
            "Ali 等 - 2020 - A Virtual Conversational Agent for Teens with Autism Spectrum Disorder.pdf"
        )
        self.assertEqual(metadata["authors"], "Ali 等")
        self.assertEqual(metadata["year"], "2020")
        self.assertEqual(
            metadata["title"],
            "A Virtual Conversational Agent for Teens with Autism Spectrum Disorder",
        )

    def test_parse_filename_metadata_irregular(self) -> None:
        metadata = MODULE.parse_filename_metadata(
            "Gershan 等 - A Pilot Analysis Investigating the Use of AI in Malingering.pdf"
        )
        self.assertEqual(metadata["authors"], "Gershan 等")
        self.assertEqual(metadata["year"], "")
        self.assertEqual(
            metadata["title"],
            "A Pilot Analysis Investigating the Use of AI in Malingering",
        )

    def test_scan_pdf_paths_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "1").mkdir()
            (root / "2").mkdir()
            (root / "1" / "paper.pdf").write_text("x", encoding="utf-8")
            (root / "2" / "paper.PDF").write_text("x", encoding="utf-8")
            found = MODULE.scan_pdf_paths(root)
            self.assertEqual([path.parent.name for path in found], ["1", "2"])

    def test_select_context_pages_prefers_keyword_pages(self) -> None:
        pages = [
            "Introduction only.",
            "Background section.",
            "Results: accuracy 0.84 and auc 0.90 on test set with cross-validation.",
            "Discussion only.",
        ]
        context, page_numbers = MODULE.select_context_pages(pages, char_budget=5000)
        self.assertIn("[Page 3]", context)
        self.assertIn(3, page_numbers)

    def test_build_full_context_keeps_page_order(self) -> None:
        pages = [
            "Page one content.",
            "Page two content.",
            "Page three content.",
        ]
        context, page_numbers = MODULE.build_full_context(pages, char_budget=5000)
        self.assertIn("[Page 1]", context)
        self.assertIn("[Page 2]", context)
        self.assertIn("[Page 3]", context)
        self.assertEqual(page_numbers, [1, 2, 3])

    def test_infer_evaluation_methods_from_text(self) -> None:
        text = "We used 10-fold cross-validation on the training data and evaluated on a held-out test set."
        self.assertEqual(MODULE.infer_evaluation_methods_from_text(text), ["independent_test_set"])

    def test_normalize_metric_category(self) -> None:
        self.assertEqual(MODULE.normalize_metric_category("roc auc"), "AUC")
        self.assertEqual(MODULE.normalize_metric_category("recall"), "Sensitivity")
        self.assertEqual(MODULE.normalize_metric_category("PR-AUC"), "PR-AUC")
        self.assertEqual(MODULE.normalize_metric_category("Dice score"), "Dice")
        self.assertEqual(
            MODULE.normalize_metric_category("Matthews correlation coefficient", "Matthews correlation coefficient"),
            "Other:Matthews correlation coefficient",
        )

    def test_normalize_metric_items_filters_non_predictive_statistics(self) -> None:
        metrics = MODULE.normalize_metric_items(
            [
                {
                    "category": "Other:Mann–Whitney U (p value)",
                    "raw_name": "Mann–Whitney U (p value)",
                    "values": ["0.004"],
                    "contexts": ["significant reduction in depressive symptoms"],
                    "evidence_snippet": "Mann–Whitney U test p value = 0.004.",
                    "page_numbers": [5],
                }
            ]
        )
        self.assertEqual(metrics, [])

    def test_normalize_metric_items_keeps_predictive_other_metric(self) -> None:
        metrics = MODULE.normalize_metric_items(
            [
                {
                    "category": "Other:balanced accuracy",
                    "raw_name": "balanced accuracy",
                    "values": ["0.81"],
                    "contexts": ["classification performance on held-out test set"],
                    "evidence_snippet": "Balanced accuracy was 0.81 on the held-out test set.",
                    "page_numbers": [7],
                }
            ]
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["category"], "Accuracy")

    def test_build_pdf_direct_input(self) -> None:
        payload = MODULE.build_pdf_direct_input("file-123", "Extract metrics.")
        self.assertEqual(payload[0]["role"], "user")
        self.assertEqual(payload[0]["content"][0], {"type": "input_file", "file_id": "file-123"})
        self.assertEqual(payload[0]["content"][1], {"type": "input_text", "text": "Extract metrics."})

    def test_normalize_evaluation_methods(self) -> None:
        methods = MODULE.normalize_evaluation_methods(
            ["10-fold cross-validation", "held-out test set", "external dataset validation"]
        )
        self.assertEqual(methods, ["external_validation"])

    def test_build_text_mode_prompt_cn(self) -> None:
        prompt = MODULE.build_text_mode_prompt(
            paper_id="1",
            pdf_path=Path("Paper/1/test.pdf"),
            filename_meta={"title": "标题", "authors": "作者", "year": "2024"},
            front_matter_meta={"title": "前页标题", "authors": "前页作者", "year": "2024"},
            context_text="结果部分显示 Accuracy 0.84。",
            prompt_language="cn",
        )
        self.assertIn("请从以下论文文本中提取结构化定量评估信息", prompt)
        self.assertIn("只保留主要结果、最优结果、最终结果", prompt)

    def test_get_system_prompt_en(self) -> None:
        prompt = MODULE.get_system_prompt("en")
        self.assertIn("Core extraction rules", prompt)
        self.assertIn("PR-AUC", prompt)

    def test_render_final_line_without_metrics(self) -> None:
        line = MODULE.render_final_line(
            authors="Ali 等",
            year="2020",
            metrics=[],
            methods=["cross_validation"],
            fallback_key="fallback",
        )
        self.assertEqual(line, "[Ali_2020] | 指标: [无定量指标] | 数值: [无] | 方法: [交叉验证]")

    def test_build_paper_record_preserves_method_hints_without_metrics(self) -> None:
        record = MODULE.build_paper_record(
            paper_id="1",
            pdf_path=Path("Paper/1/test.pdf"),
            filename_meta={"title": "Test", "authors": "Ali 等", "year": "2020"},
            front_meta={"title": "", "authors": "", "year": ""},
            llm_payload={
                "title": "Test",
                "authors": "Ali 等",
                "year": "2020",
                "has_quantitative_metrics": False,
                "metrics": [],
                "evaluation_methods": [],
            },
            extracted_page_count=12,
            sent_page_count=8,
            sent_char_count=42000,
            truncated=True,
            method_hints=["independent_test_set"],
        )
        self.assertEqual(record["evaluation_methods"], ["independent_test_set"])
        self.assertIn("独立测试集", record["final_line"])
        self.assertEqual(record["extracted_page_count"], 12)
        self.assertEqual(record["sent_page_count"], 8)
        self.assertEqual(record["sent_char_count"], 42000)
        self.assertTrue(record["truncated"])

    def test_load_completed_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            records_path.write_text(
                "\n".join(
                    [
                        json.dumps({"paper_id": "1"}, ensure_ascii=False),
                        json.dumps({"paper_id": "2"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.load_completed_paper_ids(records_path), {"1", "2"})

    def test_persist_run_config_serializes_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results" / "test_run"
            config = MODULE.ProviderConfig(
                provider="custom",
                base_url="https://example.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-test",
                input_mode="text",
                prompt_language="en",
            )
            args = Namespace(
                paper_root="Paper",
                run_name="test_run",
                limit=10,
                paper_id=["3,1", "2"],
                char_budget=24000,
                input_mode="text",
                prompt_language="en",
                keep_uploaded_files=False,
                resume=False,
            )
            MODULE.persist_run_config(output_dir, config, args, scheduled_count=3)
            payload = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["paper_ids"], ["1", "2", "3"])

    def test_materialize_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records_path = root / "records.jsonl"
            summary_path = root / "summary.csv"
            lines_path = root / "lines.txt"
            payload = {
                "paper_id": "1",
                "input_mode": "text",
                "extracted_page_count": 12,
                "sent_page_count": 8,
                "sent_char_count": 42000,
                "truncated": True,
                "title": "Test Paper",
                "authors": "Ali 等",
                "year": "2020",
                "pdf_path": "Paper/1/test.pdf",
                "has_quantitative_metrics": True,
                "metrics": [
                    {
                        "category": "Accuracy",
                        "raw_name": "accuracy",
                        "values": ["0.84"],
                        "contexts": ["held-out test set"],
                        "evidence_snippet": "Accuracy was 0.84.",
                        "page_numbers": [5],
                    }
                ],
                "evaluation_methods": ["independent_test_set"],
                "evidence": [{"snippet": "Accuracy was 0.84.", "page_numbers": [5]}],
                "final_line": "[Ali_2020] | 指标: [Accuracy] | 数值: [Accuracy=0.84 (held-out test set)] | 方法: [独立测试集]",
                "confidence": 0.9,
                "error": None,
            }
            records_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            MODULE.materialize_outputs(records_path, summary_path, lines_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn("Test Paper", summary_text)
            self.assertTrue(summary_text.splitlines()[0].startswith("folder_id,result_line,"))
            self.assertTrue(summary_text.splitlines()[1].startswith("1,"))
            self.assertIn("42000", summary_text)
            self.assertIn("[Ali_2020]", lines_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
