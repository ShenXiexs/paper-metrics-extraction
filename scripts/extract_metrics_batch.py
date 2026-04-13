#!/usr/bin/env python3
"""Batch-extract quantitative mental-health prediction metrics from PDFs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from pypdf import PdfReader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
LOGGER = logging.getLogger("extract_metrics_batch")

ALLOWED_METRIC_CATEGORIES = [
    "Accuracy",
    "AUC",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1",
]
ALLOWED_METHODS = [
    "cross_validation",
    "independent_test_set",
    "external_validation",
    "not_reported",
]
METHOD_LABELS = {
    "cross_validation": "交叉验证",
    "independent_test_set": "独立测试集",
    "external_validation": "外部验证",
    "not_reported": "未说明",
}
KEYWORD_PATTERNS = {
    r"\baccuracy\b": 3,
    r"\bauc\b": 3,
    r"\broc[- ]?auc\b": 3,
    r"\bpr[- ]?auc\b": 3,
    r"\broc\b": 2,
    r"\bsensitivity\b": 3,
    r"\bspecificity\b": 3,
    r"\bprecision\b": 3,
    r"\bf1\b": 3,
    r"\bdice\b": 3,
    r"\biou\b": 3,
    r"\bmae\b": 3,
    r"\brmse\b": 3,
    r"\brecall\b": 2,
    r"\bperformance\b": 1,
    r"\bclassifier\b": 1,
    r"\bpredict(?:ion|ive|or)?\b": 2,
    r"\bscreen(?:ing)?\b": 2,
    r"\bresults?\b": 1,
    r"\bexperiments?\b": 1,
    r"\btable\b": 1,
    r"\bfigure\b": 1,
    r"\btest set\b": 2,
    r"\btrain/test\b": 2,
    r"\btrain/validation/test\b": 2,
    r"\bhold[- ]?out\b": 2,
    r"\bcross[- ]?validation\b": 2,
    r"\bk[- ]?fold\b": 2,
    r"\bexternal validation\b": 3,
    r"\bvalidation\b": 1,
}
SYSTEM_PROMPTS = {
    "en": """Role: You are a rigorous academic information extraction assistant specializing in quantitative results extraction from research papers.

Task: Extract structured quantitative evaluation information from the given paper. All information must be strictly based on the original text. Do NOT infer, guess, or fabricate any information.

Core extraction rules:
1. Focus on quantitative evaluation results reported in Results / Experiments / Tables / Figures. Ignore background or methodology descriptions that only define metrics.
2. Extract only main results, best performance, final reported results, or the result explicitly emphasized by the authors.
3. If multiple models are reported, keep only the best-performing or most emphasized result set.
4. Ensure strict one-to-one mapping between each metric and each reported value.
5. If a metric is mentioned but no concrete value is reported, set its value to "Not reported".
6. Keep model-performance metrics reported for mental-health-related systems, including direct disorder prediction / screening / classification tasks and component-level classification or detection tasks that are part of the system (for example, cognitive distortion classification, risk classification, intent classification within a mental-health chatbot pipeline).
7. Do NOT extract intervention effectiveness statistics, hypothesis-test outputs, questionnaire score changes, usability scores, p-values, effect sizes, mean differences, or baseline/follow-up scale scores unless they are explicitly used as model-performance metrics.
8. If the paper does not report any relevant model-performance metrics, set has_quantitative_metrics to false and return an empty metrics array.
9. Return a single JSON object only. Do not use Markdown. Do not add commentary.
10. For each extracted metric item, normalize its category to one of these high-level metric groups: Accuracy, AUC, Sensitivity, Specificity, Precision, F1, or Other:<raw_name>. A paper may contain multiple metric items and multiple categories. Map ROC-AUC / AUROC into AUC. Put PR-AUC, Dice, IoU, MAE, RMSE, MCC, and other predictive metrics into Other:<raw_name> instead of creating new top-level groups.

For each metric item, normalize its category to one of:
- Accuracy
- AUC
- Sensitivity
- Specificity
- Precision
- F1
- Other:<raw_name>

Normalize evaluation methods to zero or more of:
- cross_validation
- independent_test_set
- external_validation
- not_reported

Expected JSON schema:
{
  "title": "string",
  "authors": "string or array of strings",
  "year": "string or integer",
  "has_quantitative_metrics": true,
  "metrics": [
    {
      "category": "Accuracy",
      "raw_name": "accuracy",
      "values": ["0.84"],
      "contexts": ["main result"],
      "evidence_snippet": "The model achieved 84.0% accuracy on the test set.",
      "page_numbers": [7]
    }
  ],
  "evaluation_methods": ["independent_test_set"],
  "evidence": [
    {
      "snippet": "The model achieved 84.0% accuracy on the test set.",
      "page_numbers": [7]
    }
  ],
  "confidence": 0.86
}
""",
    "cn": """角色：你是一名严谨的学术信息抽取助手，专门负责从论文中提取定量评估结果。

任务：从输入论文中结构化提取定量评估信息。所有信息都必须严格基于论文原文，不得推测、猜测或编造。

核心提取规则：
1. 重点从 Results / Experiments / Tables / Figures 中提取定量结果；忽略背景和方法部分里仅用于说明概念的指标描述。
2. 只提取主要结果、最优结果、最终结果，或作者明确强调的结果。
3. 若有多个模型，只保留论文中表现最优或作者最强调的一组结果。
4. 必须保证“指标-数值”严格一一对应。
5. 若提到了某个指标但没有给出具体数值，则该指标的数值写为“未报告数值”。
6. 允许提取心理健康相关系统中的模型性能指标，包括直接的精神疾病预测/筛查/分类任务，也包括系统内部与心理健康相关的组件级分类或检测任务，例如认知扭曲分类、风险分类、聊天机器人流程中的意图分类等。
7. 不要提取干预效果统计、假设检验结果、问卷分数变化、可用性分数、p 值、effect size、mean difference、基线/随访量表分数，除非这些量被明确作为模型性能指标使用。
8. 若论文没有报告相关的模型性能指标，则将 has_quantitative_metrics 设为 false，并返回空 metrics 数组。
9. 必须只返回一个 JSON 对象，不要使用 Markdown，不要添加解释性文字。
10. 对每一个提取出的指标项，都要把其类别归到这些高层类别之一：Accuracy、AUC、Sensitivity、Specificity、Precision、F1 或 Other:<raw_name>。同一篇论文可以同时包含多个指标项和多个类别。其中 ROC-AUC / AUROC 统一并入 AUC；PR-AUC、Dice、IoU、MAE、RMSE、MCC 等其他预测性能指标统一写为 Other:<raw_name>，不要新增新的一级类别。

对每一个指标项，类别标准化为以下之一：
- Accuracy
- AUC
- Sensitivity
- Specificity
- Precision
- F1
- Other:<raw_name>

评估方法标准化为以下零个或多个：
- cross_validation
- independent_test_set
- external_validation
- not_reported

返回 JSON 结构：
{
  "title": "string",
  "authors": "string 或 string 数组",
  "year": "string 或 integer",
  "has_quantitative_metrics": true,
  "metrics": [
    {
      "category": "Accuracy",
      "raw_name": "accuracy",
      "values": ["0.84"],
      "contexts": ["main result"],
      "evidence_snippet": "The model achieved 84.0% accuracy on the test set.",
      "page_numbers": [7]
    }
  ],
  "evaluation_methods": ["independent_test_set"],
  "evidence": [
    {
      "snippet": "The model achieved 84.0% accuracy on the test set.",
      "page_numbers": [7]
    }
  ],
  "confidence": 0.86
}
""",
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    input_mode: str = "text"
    prompt_language: str = "en"
    temperature: float = 0.0
    timeout: float = 180.0
    max_retries: int = 3
    concurrency: int = 4
    keep_uploaded_files: bool = False


@dataclass
class MetricItem:
    category: str
    raw_name: str
    values: list[str]
    contexts: list[str]
    evidence_snippet: str
    page_numbers: list[int]


@dataclass
class PaperRecord:
    paper_id: str
    pdf_path: str
    title: str
    authors: str
    year: str
    extracted_page_count: int
    sent_page_count: int
    sent_char_count: int
    truncated: bool
    has_quantitative_metrics: bool
    metrics: list[dict[str, Any]]
    evaluation_methods: list[str]
    evaluation_method_source: str
    evidence: list[dict[str, Any]]
    final_line: str
    confidence: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract quantitative predictive metrics from Paper PDFs.",
    )
    parser.add_argument("--paper-root", default="Paper", help="Root directory containing numbered paper folders.")
    parser.add_argument("--provider", required=True, help="Provider label, e.g. openai or deepseek.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", required=True, help="Model name to call.")
    parser.add_argument("--api-key-env", required=True, help="Environment variable holding the API key.")
    parser.add_argument("--run-name", required=True, help="Output directory name under results/.")
    parser.add_argument(
        "--input-mode",
        choices=["text", "text_full", "text_full_chunked", "pdf_direct"],
        default="text",
        help="Use selective local PDF-to-text extraction, fuller local PDF-to-text extraction, full-paper chunked extraction, or upload the PDF directly to the API.",
    )
    parser.add_argument(
        "--prompt-language",
        choices=["en", "cn"],
        default="en",
        help="Prompt language used for the extraction instructions.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip paper_ids already present in records.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N papers after filtering.")
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Specific paper id(s) to process. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--char-budget",
        type=int,
        default=24000,
        help="Maximum number of characters sent to the model for one paper.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum API retries per paper.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent papers to process.")
    parser.add_argument(
        "--keep-uploaded-files",
        action="store_true",
        help="When using pdf_direct mode, keep uploaded Files API objects instead of deleting them after each call.",
    )
    return parser.parse_args()


def numeric_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def trim_text(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def scan_pdf_paths(paper_root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for folder in sorted((path for path in paper_root.iterdir() if path.is_dir()), key=lambda p: numeric_sort_key(p.name)):
        matched = sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
            key=lambda p: p.name.lower(),
        )
        if not matched:
            LOGGER.warning("No PDF found in %s", folder)
            continue
        if len(matched) > 1:
            LOGGER.warning("Multiple PDFs found in %s; using %s", folder, matched[0].name)
        pdfs.append(matched[0])
    return pdfs


def parse_requested_paper_ids(values: Iterable[str]) -> set[str]:
    requested: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                requested.add(item)
    return requested


def parse_filename_metadata(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    parts = [part.strip() for part in stem.split(" - ") if part.strip()]
    authors = ""
    year = ""
    title = ""

    if len(parts) >= 3 and re.fullmatch(r"(19|20)\d{2}", parts[1]):
        authors = parts[0]
        year = parts[1]
        title = " - ".join(parts[2:])
    elif len(parts) >= 2:
        authors = parts[0]
        title = " - ".join(parts[1:])
        year_match = re.search(r"\b(19|20)\d{2}\b", stem)
        if year_match:
            year = year_match.group(0)
    else:
        title = stem
        year_match = re.search(r"\b(19|20)\d{2}\b", stem)
        if year_match:
            year = year_match.group(0)

    return {
        "title": normalize_whitespace(title),
        "authors": normalize_whitespace(authors),
        "year": year,
    }


def extract_front_matter_candidates(first_pages: list[str]) -> dict[str, str]:
    lines: list[str] = []
    for page in first_pages:
        for line in page.splitlines():
            cleaned = normalize_whitespace(line)
            if cleaned:
                lines.append(cleaned)

    title = ""
    authors = ""
    year = ""

    title_candidates: list[str] = []
    for line in lines[:25]:
        if len(line) < 20 or len(line) > 250:
            continue
        if "abstract" in line.lower() or "keywords" in line.lower():
            break
        if re.search(r"@\w|\bdoi\b|http[s]?://", line, re.IGNORECASE):
            continue
        if re.search(r"\b(received|accepted|published|citation|editor)\b", line, re.IGNORECASE):
            continue
        title_candidates.append(line)
        if len(title_candidates) == 2:
            break
    if title_candidates:
        title = normalize_whitespace(" ".join(title_candidates))

    for line in lines[:40]:
        if len(line) > 250 or len(line) < 6:
            continue
        if re.search(r"@\w|\b(university|department|school|college|institute|hospital)\b", line, re.IGNORECASE):
            continue
        if re.search(r"\babstract\b", line, re.IGNORECASE):
            break
        if re.search(r"(,| and | et al\.?| 等| 和 )", line, re.IGNORECASE):
            authors = line
            break

    for line in lines[:80]:
        year_match = re.search(r"\b(19|20)\d{2}\b", line)
        if year_match:
            year = year_match.group(0)
            break

    return {
        "title": normalize_whitespace(title),
        "authors": normalize_whitespace(authors),
        "year": year,
    }


def read_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive against malformed PDFs
            LOGGER.warning("Page extraction failed for %s: %s", pdf_path, exc)
            text = ""
        pages.append(normalize_whitespace(text))
    return pages


def score_page(text: str) -> int:
    score = 0
    for pattern, weight in KEYWORD_PATTERNS.items():
        score += len(re.findall(pattern, text, re.IGNORECASE)) * weight
    return score


def select_context_pages(pages: list[str], char_budget: int) -> tuple[str, list[int]]:
    if not pages:
        return "", []

    selected: set[int] = set(range(min(2, len(pages))))
    scored = sorted(
        ((score_page(text), index) for index, text in enumerate(pages)),
        key=lambda item: (-item[0], item[1]),
    )
    top_hits = [index for score, index in scored if score > 0][:6]
    if not top_hits:
        top_hits = list(range(min(4, len(pages))))
    for index in top_hits:
        for neighbor in (index - 1, index, index + 1):
            if 0 <= neighbor < len(pages):
                selected.add(neighbor)

    pieces: list[str] = []
    kept_pages: list[int] = []
    total_chars = 0
    for index in sorted(selected):
        page_text = normalize_whitespace(pages[index])
        if not page_text:
            continue
        header = f"[Page {index + 1}]\n"
        block = header + page_text + "\n"
        if total_chars + len(block) > char_budget:
            remaining = char_budget - total_chars - len(header) - 1
            if remaining <= 0:
                break
            block = header + trim_text(page_text, remaining) + "\n"
        pieces.append(block)
        kept_pages.append(index + 1)
        total_chars += len(block)
        if total_chars >= char_budget:
            break
    return "\n".join(pieces).strip(), kept_pages


def build_full_context(pages: list[str], char_budget: int) -> tuple[str, list[int]]:
    if not pages:
        return "", []

    pieces: list[str] = []
    kept_pages: list[int] = []
    total_chars = 0
    for index, page in enumerate(pages):
        page_text = normalize_whitespace(page)
        if not page_text:
            continue
        header = f"[Page {index + 1}]\n"
        block = header + page_text + "\n"
        if total_chars + len(block) > char_budget:
            remaining = char_budget - total_chars - len(header) - 1
            if remaining <= 0:
                break
            block = header + trim_text(page_text, remaining) + "\n"
            pieces.append(block)
            kept_pages.append(index + 1)
            break
        pieces.append(block)
        kept_pages.append(index + 1)
        total_chars += len(block)
    return "\n".join(pieces).strip(), kept_pages


def build_full_context_chunks(pages: list[str], char_budget: int) -> tuple[list[tuple[str, list[int]]], bool]:
    if not pages:
        return [], False

    chunks: list[tuple[str, list[int]]] = []
    current_pieces: list[str] = []
    current_pages: list[int] = []
    current_chars = 0
    truncated = False

    for index, page in enumerate(pages):
        page_text = normalize_whitespace(page)
        if not page_text:
            continue
        header = f"[Page {index + 1}]\n"
        block = header + page_text + "\n"

        if len(block) > char_budget:
            if current_pieces:
                chunks.append(("".join(current_pieces).strip(), current_pages.copy()))
                current_pieces = []
                current_pages = []
                current_chars = 0
            remaining = char_budget - len(header) - 1
            if remaining <= 0:
                continue
            chunks.append(((header + trim_text(page_text, remaining) + "\n").strip(), [index + 1]))
            truncated = True
            continue

        if current_chars + len(block) > char_budget and current_pieces:
            chunks.append(("".join(current_pieces).strip(), current_pages.copy()))
            current_pieces = []
            current_pages = []
            current_chars = 0

        current_pieces.append(block)
        current_pages.append(index + 1)
        current_chars += len(block)

    if current_pieces:
        chunks.append(("".join(current_pieces).strip(), current_pages.copy()))

    return chunks, truncated


def get_system_prompt(prompt_language: str) -> str:
    return SYSTEM_PROMPTS.get(prompt_language, SYSTEM_PROMPTS["en"])


def build_text_mode_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_matter_meta: dict[str, str],
    context_text: str,
    prompt_language: str = "en",
) -> str:
    if prompt_language == "cn":
        return f"""请从以下论文文本中提取结构化定量评估信息。

论文信息：
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}

请完成以下任务：
1. 判断论文是否明确报告了与精神疾病或精神健康相关的模型定量评估指标，包括直接的预测、分类、筛查、识别、检测任务，也包括系统内部与心理健康相关的组件级分类或检测任务。
2. 若有，请提取指标及其对应数值，并保证指标和数值一一对应。
3. 只保留主要结果、最优结果、最终结果，或作者明确强调的结果。
4. 评估方法可以多选：cross_validation、independent_test_set、external_validation；如果文中没有明确或可判断的方法信息，则写 not_reported。
5. 同时返回 title、authors、year，优先依据论文正文，其次参考文件名候选。
6. 指标高层类别只允许：Accuracy、AUC、Sensitivity、Specificity、Precision、F1、Other:<raw_name>。其中 ROC-AUC / AUROC 归入 AUC，其他预测性指标归入 Other。

附加说明：
- 只根据原文提取，不得补全缺失信息。
- 若论文没有报告任何定量指标，请返回 has_quantitative_metrics=false 和空 metrics。
- 若指标被提到但没有具体数值，则 values 中写“未报告数值”。
- 不要提取 p 值、效应量、量表前后测分数、均值差、可用性分数等非预测性能结果。
- 以下文本可能是从整篇论文中筛选出的关键页面片段，而不是全文。

论文文本：
{context_text}
"""

    return f"""Extract structured quantitative evaluation information from the paper text below.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}

Please do the following:
1. Decide whether the paper explicitly reports quantitative model-evaluation metrics for mental disorder or mental-health-related systems, including direct prediction/classification/screening tasks and component-level classification or detection tasks used inside the system.
2. If yes, extract the metrics and their corresponding values with strict one-to-one mapping.
3. Keep only the main results, best performance, final reported results, or the result emphasized by the authors.
4. Evaluation methods may include one or more of: cross_validation, independent_test_set, external_validation. Use not_reported only when the paper does not provide enough information.
5. Also return title, authors, and year, preferring explicit paper evidence over filename candidates.
6. Only use these high-level metric groups: Accuracy, AUC, Sensitivity, Specificity, Precision, F1, Other:<raw_name>. Map ROC-AUC / AUROC into AUC. Put other predictive metrics into Other.

Important:
- Extract strictly from the paper text. Do not fill in missing information.
- If the paper does not report any quantitative metrics, return has_quantitative_metrics=false and an empty metrics array.
- If a metric is mentioned but no concrete value is given, use "Not reported" in values.
- Do not extract p-values, effect sizes, questionnaire score changes, mean differences, or usability scores as metrics.
- The text below may be a selected evidence-focused subset rather than the full paper.

Paper text:
{context_text}
"""


def build_text_full_mode_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_matter_meta: dict[str, str],
    context_text: str,
    prompt_language: str = "en",
) -> str:
    if prompt_language == "cn":
        return f"""请从以下论文全文文本中提取结构化定量评估信息。

论文信息：
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}

请完成以下任务：
1. 判断论文是否明确报告了与精神疾病或精神健康相关的模型定量评估指标，包括直接的预测、分类、筛查、识别、检测任务，也包括系统内部与心理健康相关的组件级分类或检测任务。
2. 若有，请提取指标及其对应数值，并保证指标和数值一一对应。
3. 只保留主要结果、最优结果、最终结果，或作者明确强调的结果。
4. 评估方法可以多选：cross_validation、independent_test_set、external_validation；如果文中没有明确或可判断的方法信息，则写 not_reported。
5. 同时返回 title、authors、year，优先依据论文正文，其次参考文件名候选。
6. 指标高层类别只允许：Accuracy、AUC、Sensitivity、Specificity、Precision、F1、Other:<raw_name>。其中 ROC-AUC / AUROC 归入 AUC，其他预测性指标归入 Other。

附加说明：
- 以下文本按页顺序来自论文全文；若因长度限制被截断，也应优先基于已提供的全文顺序文本提取。
- 只根据原文提取，不得补全缺失信息。
- 若论文没有报告任何定量指标，请返回 has_quantitative_metrics=false 和空 metrics。
- 若指标被提到但没有具体数值，则 values 中写“未报告数值”。
- 不要提取 p 值、效应量、量表前后测分数、均值差、可用性分数等非预测性能结果。

论文文本：
{context_text}
"""

    return f"""Extract structured quantitative evaluation information from the full paper text below.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}

Please do the following:
1. Decide whether the paper explicitly reports quantitative model-evaluation metrics for mental disorder or mental-health-related systems, including direct prediction/classification/screening tasks and component-level classification or detection tasks used inside the system.
2. If yes, extract the metrics and their corresponding values with strict one-to-one mapping.
3. Keep only the main results, best performance, final reported results, or the result emphasized by the authors.
4. Evaluation methods may include one or more of: cross_validation, independent_test_set, external_validation. Use not_reported only when the paper does not provide enough information.
5. Also return title, authors, and year, preferring explicit paper evidence over filename candidates.
6. Only use these high-level metric groups: Accuracy, AUC, Sensitivity, Specificity, Precision, F1, Other:<raw_name>. Map ROC-AUC / AUROC into AUC. Put other predictive metrics into Other.

Important:
- The text below is the full paper text in page order, unless truncated due to length limits.
- Extract strictly from the paper text. Do not fill in missing information.
- If the paper does not report any quantitative metrics, return has_quantitative_metrics=false and an empty metrics array.
- If a metric is mentioned but no concrete value is given, use "Not reported" in values.
- Do not extract p-values, effect sizes, questionnaire score changes, mean differences, or usability scores as metrics.

Paper text:
{context_text}
"""


def build_text_full_chunk_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_matter_meta: dict[str, str],
    context_text: str,
    chunk_index: int,
    chunk_count: int,
    prompt_language: str = "en",
) -> str:
    if prompt_language == "cn":
        return f"""请从以下论文全文分块文本中提取结构化定量评估信息。

论文信息：
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}
- chunk_index: {chunk_index}
- chunk_count: {chunk_count}

请完成以下任务：
1. 这只是整篇论文按页顺序切分后的第 {chunk_index}/{chunk_count} 块。只提取本块中明确出现的定量评估结果，不要假设其他块中的内容。
2. 判断本块是否明确报告了与精神疾病或精神健康相关的模型定量评估指标，包括直接的预测、分类、筛查、识别、检测任务，也包括系统内部与心理健康相关的组件级分类或检测任务。
3. 若有，请提取指标及其对应数值，并保证指标和数值一一对应。
4. 只保留本块中主要结果、最优结果、最终结果，或作者明确强调的结果。
5. 评估方法可以多选：cross_validation、independent_test_set、external_validation；如果文中没有明确或可判断的方法信息，则写 not_reported。
6. 指标高层类别只允许：Accuracy、AUC、Sensitivity、Specificity、Precision、F1、Other:<raw_name>。其中 ROC-AUC / AUROC 归入 AUC，其他预测性指标归入 Other。

附加说明：
- 若本块没有任何相关定量指标，请返回 has_quantitative_metrics=false 和空 metrics。
- 若指标被提到但没有具体数值，则 values 中写“未报告数值”。
- 不要提取 p 值、效应量、量表前后测分数、均值差、可用性分数等非预测性能结果。
- title、authors、year 可以使用本块中的明确证据；若本块未出现，可留空，后续会结合首页和文件名补全。

论文文本：
{context_text}
"""

    return f"""Extract structured quantitative evaluation information from the chunked full-paper text below.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}
- chunk_index: {chunk_index}
- chunk_count: {chunk_count}

Please do the following:
1. This is chunk {chunk_index}/{chunk_count} from the full paper in page order. Extract only information explicitly present in this chunk. Do not assume content from other chunks.
2. Decide whether this chunk explicitly reports quantitative model-evaluation metrics for mental disorder or mental-health-related systems, including direct prediction/classification/screening tasks and component-level classification or detection tasks used inside the system.
3. If yes, extract the metrics and their corresponding values with strict one-to-one mapping.
4. Keep only the main results, best performance, final reported results, or the result emphasized by the authors within this chunk.
5. Evaluation methods may include one or more of: cross_validation, independent_test_set, external_validation. Use not_reported only when the chunk does not provide enough information.
6. Only use these high-level metric groups: Accuracy, AUC, Sensitivity, Specificity, Precision, F1, Other:<raw_name>. Map ROC-AUC / AUROC into AUC. Put other predictive metrics into Other.

Important:
- If this chunk contains no relevant quantitative metrics, return has_quantitative_metrics=false and an empty metrics array.
- If a metric is mentioned but no concrete value is given, use "Not reported" in values.
- Do not extract p-values, effect sizes, questionnaire score changes, mean differences, or usability scores as metrics.
- title, authors, and year may be left empty if they are not explicitly visible in this chunk; they will be reconciled later with front-matter and filename candidates.

Paper text:
{context_text}
"""


def build_pdf_direct_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    prompt_language: str = "en",
) -> str:
    if prompt_language == "cn":
        return f"""请从所附论文 PDF 中提取结构化定量评估信息。

论文信息：
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}

请完成以下任务：
1. 判断论文是否明确报告了与精神疾病或精神健康相关的模型定量评估指标，包括直接的预测、分类、筛查、识别、检测任务，也包括系统内部与心理健康相关的组件级分类或检测任务。
2. 若有，请提取指标及其对应数值，并保证指标和数值一一对应。
3. 只保留主要结果、最优结果、最终结果，或作者明确强调的结果。
4. 评估方法可以多选：cross_validation、independent_test_set、external_validation；如果文中没有明确或可判断的方法信息，则写 not_reported。
5. 同时返回 title、authors、year，优先依据论文正文，其次参考文件名候选。
6. 指标高层类别只允许：Accuracy、AUC、Sensitivity、Specificity、Precision、F1、Other:<raw_name>。其中 ROC-AUC / AUROC 归入 AUC，其他预测性指标归入 Other。

附加说明：
- 所附 PDF 是主信息源，文件名候选仅作辅助参考。
- 只根据原文提取，不得补全缺失信息。
- 若论文没有报告任何定量指标，请返回 has_quantitative_metrics=false 和空 metrics。
- 若指标被提到但没有具体数值，则 values 中写“未报告数值”。
- 不要提取 p 值、效应量、量表前后测分数、均值差、可用性分数等非预测性能结果。
"""

    return f"""Extract structured quantitative evaluation information from the attached paper PDF.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}

Please do the following:
1. Decide whether the paper explicitly reports quantitative model-evaluation metrics for mental disorder or mental-health-related systems, including direct prediction/classification/screening tasks and component-level classification or detection tasks used inside the system.
2. If yes, extract the metrics and their corresponding values with strict one-to-one mapping.
3. Keep only the main results, best performance, final reported results, or the result emphasized by the authors.
4. Evaluation methods may include one or more of: cross_validation, independent_test_set, external_validation. Use not_reported only when the paper does not provide enough information.
5. Also return title, authors, and year, preferring explicit paper evidence over the filename candidates.
6. Only use these high-level metric groups: Accuracy, AUC, Sensitivity, Specificity, Precision, F1, Other:<raw_name>. Map ROC-AUC / AUROC into AUC. Put other predictive metrics into Other.

Important:
- The attached PDF is the primary source. Filename candidates are only fallback hints.
- Extract strictly from the paper. Do not fill in missing information.
- If the paper does not report any quantitative metrics, return has_quantitative_metrics=false and an empty metrics array.
- If a metric is mentioned but no concrete value is given, use "Not reported" in values.
- Do not extract p-values, effect sizes, questionnaire score changes, mean differences, or usability scores as metrics.
"""


def parse_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("Empty model response.")
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    if start == -1:
        raise ValueError("Model response does not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start : index + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break
    raise ValueError("Unable to parse a JSON object from model response.")


class LLMExtractor:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def _client(self) -> OpenAI:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")
        return OpenAI(api_key=api_key, base_url=self.config.base_url, timeout=self.config.timeout)

    def extract(self, user_prompt: str, pdf_path: Path | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if self.config.input_mode == "pdf_direct":
                    if pdf_path is None:
                        raise ValueError("pdf_path is required when input_mode=pdf_direct")
                    return self._extract_once_pdf_direct(user_prompt, pdf_path)
                return self._extract_once_text(user_prompt)
            except Exception as exc:  # pragma: no cover - retry path depends on live API
                last_exc = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_seconds = min(2 ** (attempt - 1), 16)
                LOGGER.warning(
                    "API call failed on attempt %s/%s for model %s: %s",
                    attempt,
                    self.config.max_retries,
                    self.config.model,
                    exc,
                )
                time.sleep(sleep_seconds)
        assert last_exc is not None
        raise last_exc

    def _extract_once_text(self, user_prompt: str) -> dict[str, Any]:
        client = self._client()
        common_kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": get_system_prompt(self.config.prompt_language)},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
        }
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                **common_kwargs,
            )
        except Exception:
            response = client.chat.completions.create(**common_kwargs)

        content = response.choices[0].message.content or ""
        return parse_json_object(content)

    def _extract_once_pdf_direct(self, user_prompt: str, pdf_path: Path) -> dict[str, Any]:
        client = self._client()
        uploaded_file_id: str | None = None
        try:
            with pdf_path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="user_data")
            uploaded_file_id = uploaded.id
            response = client.responses.create(
                model=self.config.model,
                instructions=get_system_prompt(self.config.prompt_language),
                input=build_pdf_direct_input(uploaded_file_id, user_prompt),
                temperature=self.config.temperature,
            )
            return parse_json_object(extract_response_text(response))
        finally:
            if uploaded_file_id and not self.config.keep_uploaded_files:
                try:
                    client.files.delete(uploaded_file_id)
                except Exception as exc:  # pragma: no cover - network cleanup path
                    LOGGER.warning("Failed to delete uploaded file %s: %s", uploaded_file_id, exc)


def build_pdf_direct_input(file_id: str, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": user_prompt},
            ],
        }
    ]


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if text:
                    pieces.append(text)
    return "\n".join(pieces).strip()


def normalize_year(value: Any) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def normalize_authors(value: Any) -> str:
    if isinstance(value, list):
        return normalize_whitespace(", ".join(str(item) for item in value if item))
    return normalize_whitespace(str(value or ""))


def normalize_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(confidence, 1.0))


def normalize_metric_category(category: Any, raw_name: Any = "") -> str:
    source = normalize_whitespace(str(category or raw_name or ""))
    lowered = source.lower()
    if any(token in lowered for token in ["pr-auc", "pr auc", "average precision"]):
        return f"Other:{normalize_whitespace(str(raw_name or category or 'PR-AUC'))}"
    if any(token in lowered for token in ["dice", "iou", "intersection over union", "mae", "mean absolute error", "rmse", "root mean squared error"]):
        return f"Other:{normalize_whitespace(str(raw_name or category or source or 'Other'))}"
    mapping = {
        "accuracy": "Accuracy",
        "balanced accuracy": "Accuracy",
        "acc": "Accuracy",
        "auc": "AUC",
        "roc auc": "AUC",
        "auroc": "AUC",
        "roc-auc": "AUC",
        "roc auc score": "AUC",
        "sensitivity": "Sensitivity",
        "recall": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "positive predictive value": "Precision",
        "ppv": "Precision",
        "f1": "F1",
        "f1 score": "F1",
        "f-1": "F1",
    }
    if lowered in mapping:
        return mapping[lowered]
    for key, target in mapping.items():
        if key in lowered:
            return target
    raw_display = normalize_whitespace(str(raw_name or category or "Other"))
    return f"Other:{raw_display}"


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        cleaned = normalize_whitespace(str(item))
        if cleaned:
            normalized.append(cleaned)
    return normalized


def normalize_page_numbers(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    page_numbers: list[int] = []
    for item in values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0:
            page_numbers.append(number)
    return sorted(dict.fromkeys(page_numbers))


def is_prediction_performance_metric(metric: dict[str, Any]) -> bool:
    category = normalize_whitespace(str(metric.get("category", "")))
    if category in {
        "Accuracy",
        "AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    }:
        return True

    text = " ".join(
        [
            normalize_whitespace(str(metric.get("raw_name", ""))),
            " ".join(normalize_string_list(metric.get("contexts"))),
            normalize_whitespace(str(metric.get("evidence_snippet", ""))),
        ]
    ).lower()

    non_predictive_patterns = [
        r"\brecall task\b",
        r"\bp[\s-]?value\b",
        r"\beffect size\b",
        r"\bmann",
        r"\bwilcoxon\b",
        r"\bt[- ]?test\b",
        r"\banova\b",
        r"\bmean difference\b",
        r"\bphq[- ]?9\b",
        r"\bgad[- ]?7\b",
        r"\bscore\b",
        r"\bbaseline\b",
        r"\bfollow[- ]?up\b",
        r"\bpost[- ]?test\b",
        r"\bpre[- ]?test\b",
        r"\bcohen'?s d\b",
        r"\bconfidence interval\b",
        r"\bodds ratio\b",
        r"\bhazard ratio\b",
        r"\bregression\b",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in non_predictive_patterns):
        return False

    predictive_other_patterns = [
        r"\bpr[- ]?auc\b",
        r"\baverage precision\b",
        r"\bdice\b",
        r"\biou\b",
        r"\bintersection over union\b",
        r"\bmae\b",
        r"\bmean absolute error\b",
        r"\brmse\b",
        r"\broot mean squared error\b",
        r"\bbalanced accuracy\b",
        r"\bmcc\b",
        r"\bconcordance index\b",
        r"\bc-index\b",
        r"\bclassification performance\b",
        r"\bprediction performance\b",
        r"\bscreening performance\b",
        r"\bdiagnostic performance\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in predictive_other_patterns)


def normalize_metric_items(metrics: Any) -> list[dict[str, Any]]:
    if not isinstance(metrics, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in metrics:
        if isinstance(item, dict):
            raw_name = normalize_whitespace(str(item.get("raw_name", "") or item.get("name", "")))
            category = normalize_metric_category(item.get("category"), raw_name)
            metric = MetricItem(
                category=category,
                raw_name=raw_name or category,
                values=normalize_string_list(item.get("values")),
                contexts=normalize_string_list(item.get("contexts") or item.get("context")),
                evidence_snippet=normalize_whitespace(str(item.get("evidence_snippet", "") or item.get("evidence", ""))),
                page_numbers=normalize_page_numbers(item.get("page_numbers") or item.get("pages")),
            )
        else:
            category = normalize_metric_category(item)
            metric = MetricItem(
                category=category,
                raw_name=normalize_whitespace(str(item)),
                values=[],
                contexts=[],
                evidence_snippet="",
                page_numbers=[],
            )
        normalized.append(asdict(metric))
    return [metric for metric in normalized if is_prediction_performance_metric(metric)]


HEURISTIC_METRIC_PATTERNS = {
    "Accuracy": [r"\bbalanced accuracy\b", r"\baccuracy\b", r"\bacc\b"],
    "AUC": [r"\broc[- ]?auc\b", r"\bauroc\b", r"\bauc\b"],
    "Sensitivity": [r"\bsensitivity\b"],
    "Specificity": [r"\bspecificity\b"],
    "Precision": [r"\bprecision\b", r"\bpositive predictive value\b", r"\bppv\b"],
    "F1": [r"\bf1(?:[- ]?score)?\b", r"\bf-1\b"],
}
NUMERIC_VALUE_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}(?:\.\d+)?%?|\.\d+)(?![\w.])")


def split_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def pick_numeric_value_near_match(sentence: str, alias_match: re.Match[str]) -> str | None:
    after_window = sentence[alias_match.end() : min(len(sentence), alias_match.end() + 120)]
    connector_patterns = [
        "=",
        ":",
        " was ",
        " were ",
        " is ",
        " are ",
        " reached ",
        " reach ",
        " achieved ",
        " achieve ",
        " yielded ",
        " obtained ",
        " on the ",
        " on ",
        " for the ",
        " for ",
        " in the ",
        " at ",
        " to ",
    ]
    for match in NUMERIC_VALUE_PATTERN.finditer(after_window):
        bridge = f" {after_window[: match.start()].lower()} "
        if bridge.strip() and not any(token in bridge for token in connector_patterns):
            continue
        value = match.group(0)
        if value.isdigit() and int(value) > 100:
            continue
        return value
    return None


def extract_context_labels(text: str) -> list[str]:
    lowered = text.lower()
    contexts: list[str] = []
    if any(token in lowered for token in ["external validation", "external dataset", "external cohort", "external hospital"]):
        contexts.append("external validation")
    if any(token in lowered for token in ["test set", "held-out", "holdout", "train/test", "independent test"]):
        contexts.append("test set")
    if any(token in lowered for token in ["cross-validation", "cross validation", "k-fold", "k fold", "leave-one-out"]):
        contexts.append("cross-validation")
    if any(token in lowered for token in ["classification performance", "classification task", "classifier"]):
        contexts.append("classification task")
    if any(token in lowered for token in ["screening performance", "screening task"]):
        contexts.append("screening task")
    return contexts


def heuristic_extract_metrics_from_pages(pages: list[str]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for page_number, page_text in enumerate(pages, start=1):
        for sentence in split_sentences(page_text):
            for category, patterns in HEURISTIC_METRIC_PATTERNS.items():
                matched_value: str | None = None
                for pattern in patterns:
                    alias_match = re.search(pattern, sentence, re.IGNORECASE)
                    if not alias_match:
                        continue
                    matched_value = pick_numeric_value_near_match(sentence, alias_match)
                    if matched_value:
                        break
                if not matched_value:
                    continue

                item = aggregated.setdefault(
                    category,
                    {
                        "category": category,
                        "raw_name": category.lower(),
                        "values": [],
                        "contexts": [],
                        "evidence_snippet": trim_text(sentence, 320),
                        "page_numbers": [],
                    },
                )
                if matched_value not in item["values"]:
                    item["values"].append(matched_value)
                for context in extract_context_labels(sentence):
                    if context not in item["contexts"]:
                        item["contexts"].append(context)
                if page_number not in item["page_numbers"]:
                    item["page_numbers"].append(page_number)
                if not item["evidence_snippet"]:
                    item["evidence_snippet"] = trim_text(sentence, 320)
    return list(aggregated.values())


def normalize_evaluation_methods(methods: Any) -> list[str]:
    raw_methods = normalize_string_list(methods)
    detected: list[str] = []
    for value in raw_methods:
        lowered = value.lower()
        if any(token in lowered for token in ["cross-validation", "cross validation", "k-fold", "k fold", "leave-one-out", "loo"]):
            detected.append("cross_validation")
        elif any(token in lowered for token in ["external validation", "external test", "external dataset", "external cohort", "external hospital"]):
            detected.append("external_validation")
        elif any(token in lowered for token in ["test set", "held-out", "holdout", "train/test", "independent test", "split"]):
            detected.append("independent_test_set")
        elif lowered in ALLOWED_METHODS:
            detected.append(lowered)

    deduped = [item for item in dict.fromkeys(detected) if item in ALLOWED_METHODS and item != "not_reported"]
    if not deduped:
        return ["not_reported"]
    ordered = [item for item in ["cross_validation", "independent_test_set", "external_validation"] if item in deduped]
    return ordered or deduped


def infer_evaluation_methods_from_text(text: str) -> list[str]:
    text = normalize_whitespace(text).lower()
    if not text:
        return ["not_reported"]

    candidates: list[str] = []
    if any(token in text for token in ["external validation", "external dataset", "external cohort", "external hospital", "external test set"]):
        candidates.append("external_validation")
    if any(token in text for token in ["train/test split", "train/validation/test", "test set", "held-out test", "holdout set", "independent test set"]):
        candidates.append("independent_test_set")
    if any(token in text for token in ["cross-validation", "cross validation", "k-fold", "k fold", "leave-one-out", "loo cross"]):
        candidates.append("cross_validation")
    return normalize_evaluation_methods(candidates)


def infer_evaluation_method_with_source(text: str, metrics: list[dict[str, Any]] | None = None) -> tuple[list[str], str]:
    explicit_methods = infer_evaluation_methods_from_text(text)
    if explicit_methods != ["not_reported"]:
        return explicit_methods, "explicit_text"

    if not metrics:
        return ["not_reported"], "not_reported"

    lowered = normalize_whitespace(text).lower()
    heuristic_independent_signals = [
        "training data",
        "train data",
        "train set",
        "test data",
        "test set",
        "evaluation set",
        "evaluation data",
        "experimental test",
        "experimental tests",
        "benchmark",
        "dataset",
        "datasets",
        "fine-tuning stage",
        "fine tuning stage",
        "inference stage",
        "classification model",
        "classification performance",
        "model structure",
        "we train models",
        "trained model",
    ]
    if any(signal in lowered for signal in heuristic_independent_signals):
        return ["independent_test_set"], "heuristic_text"

    return ["not_reported"], "not_reported"


def normalize_evidence(evidence: Any, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            snippet = normalize_whitespace(str(item.get("snippet", "") or item.get("evidence", "")))
            pages = normalize_page_numbers(item.get("page_numbers") or item.get("pages"))
            if snippet or pages:
                normalized.append({"snippet": snippet, "page_numbers": pages})
    if normalized:
        return normalized

    for metric in metrics:
        snippet = normalize_whitespace(metric.get("evidence_snippet", ""))
        pages = normalize_page_numbers(metric.get("page_numbers"))
        if snippet or pages:
            normalized.append({"snippet": snippet, "page_numbers": pages})
    return normalized


def deduplicate_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for metric in metrics:
        key = (
            normalize_whitespace(str(metric.get("category", ""))),
            tuple(normalize_string_list(metric.get("values"))),
            tuple(normalize_string_list(metric.get("contexts"))),
            normalize_whitespace(str(metric.get("evidence_snippet", ""))),
            tuple(normalize_page_numbers(metric.get("page_numbers"))),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(metric)
    return deduped


def deduplicate_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for item in evidence:
        snippet = normalize_whitespace(str(item.get("snippet", "")))
        pages = tuple(normalize_page_numbers(item.get("page_numbers")))
        key = (snippet, pages)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"snippet": snippet, "page_numbers": list(pages)})
    return deduped


def merge_chunk_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged_metrics: list[dict[str, Any]] = []
    merged_evidence: list[dict[str, Any]] = []
    merged_methods: list[str] = []
    title = ""
    authors = ""
    year = ""
    confidences: list[float] = []

    for payload in payloads:
        if not title:
            title = normalize_whitespace(str(payload.get("title", "")))
        if not authors:
            authors = normalize_authors(payload.get("authors"))
        if not year:
            year = normalize_year(payload.get("year"))
        chunk_metrics = normalize_metric_items(payload.get("metrics"))
        if chunk_metrics:
            merged_metrics.extend(chunk_metrics)
        merged_evidence.extend(normalize_evidence(payload.get("evidence"), chunk_metrics))
        merged_methods.extend(normalize_string_list(payload.get("evaluation_methods")))
        confidence = normalize_confidence(payload.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)

    merged_metrics = deduplicate_metrics(merged_metrics)
    merged_evidence = deduplicate_evidence(merged_evidence)
    merged_methods = normalize_evaluation_methods(merged_methods)

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "has_quantitative_metrics": bool(merged_metrics),
        "metrics": merged_metrics,
        "evaluation_methods": merged_methods,
        "evidence": merged_evidence,
        "confidence": max(confidences) if confidences else None,
    }


def pick_first_non_empty(*values: str) -> str:
    for value in values:
        cleaned = normalize_whitespace(value)
        if cleaned:
            return cleaned
    return ""


def build_citation_key(authors: str, year: str, fallback: str) -> str:
    normalized_authors = normalize_whitespace(authors)
    if normalized_authors:
        first_author = re.split(r",|;| and | & | et al\.?| 等| 和 ", normalized_authors, maxsplit=1, flags=re.IGNORECASE)[0]
        first_author = re.sub(r"\s+", "", first_author)
    else:
        first_author = re.sub(r"\s+", "", fallback)[:24]
    normalized_year = normalize_year(year) or "UnknownYear"
    return f"{first_author}_{normalized_year}" if first_author else f"Unknown_{normalized_year}"


def render_metric_values(metrics: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for metric in metrics:
        category = metric.get("category", "Other")
        values = normalize_string_list(metric.get("values"))
        contexts = normalize_string_list(metric.get("contexts"))
        display_value = " / ".join(values) if values else "未报告数值"
        if contexts:
            display_value = f"{display_value} ({' / '.join(contexts)})"
        parts.append(f"{category}={display_value}")
    return ", ".join(parts)


def render_final_line(authors: str, year: str, metrics: list[dict[str, Any]], methods: list[str], fallback_key: str) -> str:
    citation_key = build_citation_key(authors, year, fallback_key)
    method_labels = ", ".join(METHOD_LABELS.get(method, method) for method in methods if method != "not_reported")
    method_labels = method_labels or "未说明"
    if not metrics:
        return f"[{citation_key}] | 指标: [无定量指标] | 数值: [无] | 方法: [{method_labels}]"

    categories = ", ".join(dict.fromkeys(metric["category"] for metric in metrics))
    values = render_metric_values(metrics)
    return f"[{citation_key}] | 指标: [{categories}] | 数值: [{values}] | 方法: [{method_labels}]"


def build_paper_record(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_meta: dict[str, str],
    llm_payload: dict[str, Any],
    extracted_page_count: int,
    sent_page_count: int,
    sent_char_count: int,
    truncated: bool,
    method_hints: list[str] | None = None,
    fallback_metrics: list[dict[str, Any]] | None = None,
    method_source: str = "not_reported",
) -> dict[str, Any]:
    metrics = normalize_metric_items(llm_payload.get("metrics"))
    if not metrics and fallback_metrics:
        metrics = normalize_metric_items(fallback_metrics)
    has_quantitative_metrics = bool(metrics)

    methods = normalize_evaluation_methods(llm_payload.get("evaluation_methods"))
    if methods == ["not_reported"] and method_hints:
        methods = method_hints
    if methods != ["not_reported"] and method_source == "not_reported":
        method_source = "explicit_or_fallback"

    title = pick_first_non_empty(str(llm_payload.get("title", "")), filename_meta.get("title", ""), front_meta.get("title", ""), pdf_path.stem)
    authors = pick_first_non_empty(normalize_authors(llm_payload.get("authors")), filename_meta.get("authors", ""), front_meta.get("authors", ""), "Unknown")
    year = pick_first_non_empty(normalize_year(llm_payload.get("year")), filename_meta.get("year", ""), front_meta.get("year", ""), "Unknown")
    evidence = normalize_evidence(llm_payload.get("evidence"), metrics)
    confidence = normalize_confidence(llm_payload.get("confidence"))
    final_line = render_final_line(authors, year, metrics, methods, pdf_path.stem)

    record = PaperRecord(
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        title=title,
        authors=authors,
        year=year,
        extracted_page_count=extracted_page_count,
        sent_page_count=sent_page_count,
        sent_char_count=sent_char_count,
        truncated=truncated,
        has_quantitative_metrics=has_quantitative_metrics,
        metrics=metrics,
        evaluation_methods=methods,
        evaluation_method_source=method_source,
        evidence=evidence,
        final_line=final_line,
        confidence=confidence,
        error=None,
    )
    return asdict(record)


def process_one_paper(pdf_path: Path, extractor: LLMExtractor, char_budget: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    paper_id = pdf_path.parent.name
    filename_meta = parse_filename_metadata(pdf_path.name)
    try:
        front_meta = {"title": "", "authors": "", "year": ""}
        selected_pages: list[int] = []
        method_hints = ["not_reported"]
        method_source = "not_reported"
        fallback_metrics: list[dict[str, Any]] = []
        extracted_page_count = 0
        sent_char_count = 0
        truncated = False
        full_text = ""

        if extractor.config.input_mode in {"text", "text_full", "text_full_chunked"}:
            pages = read_pdf_pages(pdf_path)
            if not any(page.strip() for page in pages):
                raise RuntimeError("No extractable text found in PDF.")

            extracted_page_count = len(pages)
            full_text = " ".join(pages)
            front_meta = extract_front_matter_candidates(pages[:2])
            fallback_metrics = heuristic_extract_metrics_from_pages(pages)
            method_hints, method_source = infer_evaluation_method_with_source(full_text, fallback_metrics)
            if extractor.config.input_mode == "text_full_chunked":
                context_chunks, truncated = build_full_context_chunks(pages, char_budget)
                if not context_chunks:
                    raise RuntimeError("No context chunks built from PDF text.")
                llm_payloads: list[dict[str, Any]] = []
                selected_pages = []
                sent_char_count = 0
                for chunk_index, (context_text, chunk_pages) in enumerate(context_chunks, start=1):
                    selected_pages.extend(chunk_pages)
                    sent_char_count += len(context_text)
                    llm_payloads.append(
                        extractor.extract(
                            build_text_full_chunk_prompt(
                                paper_id=paper_id,
                                pdf_path=pdf_path,
                                filename_meta=filename_meta,
                                front_matter_meta=front_meta,
                                context_text=context_text,
                                chunk_index=chunk_index,
                                chunk_count=len(context_chunks),
                                prompt_language=extractor.config.prompt_language,
                            )
                        )
                    )
                llm_payload = merge_chunk_payloads(llm_payloads)
                selected_pages = sorted(dict.fromkeys(selected_pages))
            elif extractor.config.input_mode == "text_full":
                context_text, selected_pages = build_full_context(pages, char_budget)
                if not context_text:
                    raise RuntimeError("No context selected from PDF text.")
                sent_char_count = len(context_text)
                truncated = sent_char_count >= char_budget or len(selected_pages) < extracted_page_count
                llm_payload = extractor.extract(
                    build_text_full_mode_prompt(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        filename_meta=filename_meta,
                        front_matter_meta=front_meta,
                        context_text=context_text,
                        prompt_language=extractor.config.prompt_language,
                    )
                )
            else:
                context_text, selected_pages = select_context_pages(pages, char_budget)
                if not context_text:
                    raise RuntimeError("No context selected from PDF text.")
                sent_char_count = len(context_text)
                truncated = sent_char_count >= char_budget or len(selected_pages) < extracted_page_count
                llm_payload = extractor.extract(
                    build_text_mode_prompt(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        filename_meta=filename_meta,
                        front_matter_meta=front_meta,
                        context_text=context_text,
                        prompt_language=extractor.config.prompt_language,
                    )
                )
        else:
            llm_payload = extractor.extract(
                build_pdf_direct_prompt(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    filename_meta=filename_meta,
                    prompt_language=extractor.config.prompt_language,
                ),
                pdf_path=pdf_path,
            )
        if full_text:
            llm_metrics_present = bool(normalize_metric_items(llm_payload.get("metrics")))
            if llm_metrics_present:
                method_hints, method_source = infer_evaluation_method_with_source(full_text, [{"category": "metric"}])
        record = build_paper_record(
            paper_id=paper_id,
            pdf_path=pdf_path,
            filename_meta=filename_meta,
            front_meta=front_meta,
            llm_payload=llm_payload,
            extracted_page_count=extracted_page_count,
            sent_page_count=len(selected_pages),
            sent_char_count=sent_char_count,
            truncated=truncated,
            method_hints=method_hints,
            fallback_metrics=fallback_metrics,
            method_source=method_source,
        )
        record["input_mode"] = extractor.config.input_mode
        record["context_page_numbers"] = selected_pages
        LOGGER.info(
            "paper_id=%s | mode=%s | extracted_pages=%s | sent_pages=%s | sent_chars=%s | truncated=%s",
            paper_id,
            extractor.config.input_mode,
            extracted_page_count,
            len(selected_pages),
            sent_char_count,
            truncated,
        )
        return record, None
    except Exception as exc:
        error_payload = {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "title": filename_meta.get("title", ""),
            "authors": filename_meta.get("authors", ""),
            "year": filename_meta.get("year", ""),
            "error": str(exc),
        }
        return None, error_payload


def load_completed_paper_ids(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    completed: set[str] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(payload.get("paper_id", "")).strip()
            if paper_id:
                completed.add(paper_id)
    return completed


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    ensure_parent_dir(path)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", [])
    methods = record.get("evaluation_methods", [])
    evidence = record.get("evidence", [])
    evidence_snippets = " || ".join(
        normalize_whitespace(str(item.get("snippet", "")))
        for item in evidence
        if normalize_whitespace(str(item.get("snippet", "")))
    )
    evidence_page_numbers = sorted(
        {
            page
            for item in evidence
            for page in normalize_page_numbers(item.get("page_numbers"))
        }
    )
    context_page_numbers = normalize_page_numbers(record.get("context_page_numbers"))
    return {
        "folder_id": record.get("paper_id", ""),
        "result_line": record.get("final_line", ""),
        "input_mode": record.get("input_mode", ""),
        "extracted_page_count": record.get("extracted_page_count", ""),
        "sent_page_count": record.get("sent_page_count", ""),
        "sent_char_count": record.get("sent_char_count", ""),
        "truncated": record.get("truncated", ""),
        "title": record.get("title", ""),
        "authors": record.get("authors", ""),
        "year": record.get("year", ""),
        "pdf_path": record.get("pdf_path", ""),
        "has_quantitative_metrics": record.get("has_quantitative_metrics", False),
        "metric_categories": ", ".join(dict.fromkeys(metric.get("category", "") for metric in metrics if metric.get("category"))),
        "metric_values": render_metric_values(metrics),
        "evaluation_methods": ", ".join(METHOD_LABELS.get(method, method) for method in methods),
        "evaluation_method_source": record.get("evaluation_method_source", ""),
        "evidence_snippets": evidence_snippets,
        "evidence_page_numbers": ", ".join(str(page) for page in evidence_page_numbers),
        "context_page_numbers": ", ".join(str(page) for page in context_page_numbers),
        "confidence": record.get("confidence", ""),
    }


def materialize_outputs(records_path: Path, summary_csv_path: Path, lines_txt_path: Path) -> None:
    records = read_jsonl(records_path)
    records.sort(key=lambda item: numeric_sort_key(str(item.get("paper_id", ""))))

    ensure_parent_dir(summary_csv_path)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "folder_id",
                "result_line",
                "input_mode",
                "extracted_page_count",
                "sent_page_count",
                "sent_char_count",
                "truncated",
                "title",
                "authors",
                "year",
                "pdf_path",
                "has_quantitative_metrics",
                "metric_categories",
                "metric_values",
                "evaluation_methods",
                "evaluation_method_source",
                "evidence_snippets",
                "evidence_page_numbers",
                "context_page_numbers",
                "confidence",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record_for_csv(record))

    ensure_parent_dir(lines_txt_path)
    with lines_txt_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(str(record.get("final_line", "")).strip() + "\n")


def persist_run_config(output_dir: Path, config: ProviderConfig, args: argparse.Namespace, scheduled_count: int) -> None:
    config_path = output_dir / "run_config.json"
    requested_ids = sorted(parse_requested_paper_ids(args.paper_id), key=numeric_sort_key)
    payload = {
        "provider": asdict(config),
        "paper_root": str(Path(args.paper_root).resolve()),
        "run_name": args.run_name,
        "limit": args.limit,
        "paper_ids": requested_ids,
        "char_budget": args.char_budget,
        "input_mode": args.input_mode,
        "prompt_language": args.prompt_language,
        "keep_uploaded_files": args.keep_uploaded_files,
        "resume": args.resume,
        "scheduled_count": scheduled_count,
    }
    ensure_parent_dir(config_path)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    paper_root = Path(args.paper_root).resolve()
    if not paper_root.exists():
        raise SystemExit(f"Paper root does not exist: {paper_root}")

    output_dir = Path("results") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    lines_txt_path = output_dir / "lines.txt"

    requested_ids = parse_requested_paper_ids(args.paper_id)
    scanned_pdfs = scan_pdf_paths(paper_root)
    if requested_ids:
        scanned_pdfs = [pdf for pdf in scanned_pdfs if pdf.parent.name in requested_ids]
    if args.limit is not None:
        scanned_pdfs = scanned_pdfs[: args.limit]

    if args.resume:
        completed = load_completed_paper_ids(records_path)
        pending_pdfs = [pdf for pdf in scanned_pdfs if pdf.parent.name not in completed]
    else:
        pending_pdfs = scanned_pdfs

    config = ProviderConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        input_mode=args.input_mode,
        prompt_language=args.prompt_language,
        max_retries=max(1, args.max_retries),
        concurrency=max(1, args.concurrency),
        keep_uploaded_files=args.keep_uploaded_files,
    )
    if args.input_mode == "pdf_direct" and "api.openai.com" not in args.base_url:
        LOGGER.warning(
            "input_mode=pdf_direct depends on Files + Responses API compatibility and is only validated against official OpenAI."
        )
    persist_run_config(output_dir, config, args, scheduled_count=len(pending_pdfs))

    LOGGER.info("Scanned %s PDFs; %s pending for this run.", len(scanned_pdfs), len(pending_pdfs))
    if not pending_pdfs:
        LOGGER.info("Nothing to process. Rebuilding summary outputs from existing records.")
        materialize_outputs(records_path, summary_csv_path, lines_txt_path)
        return 0

    extractor = LLMExtractor(config)
    write_lock = threading.Lock()
    total = len(pending_pdfs)

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_map = {
            executor.submit(process_one_paper, pdf_path, extractor, args.char_budget): pdf_path
            for pdf_path in pending_pdfs
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            pdf_path = future_map[future]
            try:
                record, error = future.result()
                if record is not None:
                    append_jsonl(records_path, record, write_lock)
                    LOGGER.info("[%s/%s] OK %s", index, total, pdf_path.parent.name)
                else:
                    append_jsonl(errors_path, error or {}, write_lock)
                    LOGGER.error("[%s/%s] ERROR %s", index, total, pdf_path.parent.name)
            except Exception as exc:  # pragma: no cover - defensive around future handling
                append_jsonl(
                    errors_path,
                    {"paper_id": pdf_path.parent.name, "pdf_path": str(pdf_path), "error": str(exc)},
                    write_lock,
                )
                LOGGER.exception("[%s/%s] FUTURE ERROR %s", index, total, pdf_path.parent.name)

    materialize_outputs(records_path, summary_csv_path, lines_txt_path)
    LOGGER.info("Finished. Results written to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
