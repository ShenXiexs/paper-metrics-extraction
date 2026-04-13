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
    "cross_validation": "Cross-validation",
    "independent_test_set": "Independent test set",
    "external_validation": "External validation",
    "not_reported": "未说明",
}
KEYWORD_PATTERNS = {
    r"\baccuracy\b": 3,
    r"\bauc\b": 3,
    r"\broc\b": 2,
    r"\bsensitivity\b": 3,
    r"\bspecificity\b": 3,
    r"\bprecision\b": 3,
    r"\bf1\b": 3,
    r"\brecall\b": 2,
    r"\bperformance\b": 1,
    r"\bclassifier\b": 1,
    r"\bpredict(?:ion|ive|or)?\b": 2,
    r"\bscreen(?:ing)?\b": 2,
    r"\btest set\b": 2,
    r"\btrain/test\b": 2,
    r"\bhold[- ]?out\b": 2,
    r"\bcross[- ]?validation\b": 2,
    r"\bk[- ]?fold\b": 2,
    r"\bexternal validation\b": 3,
    r"\bvalidation\b": 1,
}
SYSTEM_PROMPT = """You are an expert research extraction assistant.

Your job is to extract only explicitly reported quantitative predictive performance results for mental disorder / mental health prediction, recognition, classification, detection, or screening tasks.

Return a single JSON object only. Do not wrap it in Markdown. Do not add commentary.

Hard rules:
1. Only extract metrics that are explicitly reported in the paper text.
2. Do not infer missing numbers from qualitative claims such as "improved", "effective", or "promising".
3. If the paper reports multiple models, datasets, or splits, keep all of them. Do not average and do not keep only the best one.
4. Normalize metric categories to one of:
   - Accuracy
   - AUC
   - Sensitivity
   - Specificity
   - Precision
   - F1
   - Other:<raw_name>
5. Normalize evaluation methods to any subset of:
   - cross_validation
   - independent_test_set
   - external_validation
   - not_reported
6. If the paper does not report any quantitative metric relevant to prediction / classification / screening performance, set has_quantitative_metrics to false and return an empty metrics array.
7. Keep short evidence snippets and page numbers when possible.

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
      "values": ["0.84", "84.0%"],
      "contexts": ["SVM on held-out test set"],
      "evidence_snippet": "Accuracy reached 84.0% on the held-out test set.",
      "page_numbers": [7]
    }
  ],
  "evaluation_methods": ["cross_validation", "independent_test_set"],
  "evidence": [
    {
      "snippet": "Accuracy reached 84.0% on the held-out test set.",
      "page_numbers": [7]
    }
  ],
  "confidence": 0.86
}
"""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    input_mode: str = "text"
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
    has_quantitative_metrics: bool
    metrics: list[dict[str, Any]]
    evaluation_methods: list[str]
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
        choices=["text", "pdf_direct"],
        default="text",
        help="Use local PDF-to-text extraction or upload the PDF directly to the API.",
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


def build_text_mode_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_matter_meta: dict[str, str],
    context_text: str,
) -> str:
    return f"""Extract structured information from the paper below.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}
- front_matter_title_candidate: {front_matter_meta.get("title", "")}
- front_matter_authors_candidate: {front_matter_meta.get("authors", "")}
- front_matter_year_candidate: {front_matter_meta.get("year", "")}

Extraction target:
1. Decide whether the paper explicitly reports quantitative predictive / classification / screening performance metrics for mental disorder or mental health related prediction tasks.
2. If yes, extract every reported metric and its concrete values.
3. Extract evaluation method labels from only: cross_validation, independent_test_set, external_validation. If not clear, use not_reported.
4. Also return title, authors, and year. Prefer explicit paper evidence over filename guesses.

Important:
- The paper may be about interventions or chatbots and may contain no predictive metrics. In that case set has_quantitative_metrics to false.
- Keep values exactly as reported when possible.
- Keep evidence snippets short.

Paper text:
{context_text}
"""


def build_pdf_direct_prompt(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
) -> str:
    return f"""Extract structured information from the attached paper PDF.

Paper info:
- paper_id: {paper_id}
- pdf_path: {pdf_path}
- filename_title_candidate: {filename_meta.get("title", "")}
- filename_authors_candidate: {filename_meta.get("authors", "")}
- filename_year_candidate: {filename_meta.get("year", "")}

Extraction target:
1. Decide whether the paper explicitly reports quantitative predictive / classification / screening performance metrics for mental disorder or mental health related prediction tasks.
2. If yes, extract every reported metric and its concrete values.
3. Extract evaluation method labels from only: cross_validation, independent_test_set, external_validation. If not clear, use not_reported.
4. Also return title, authors, and year. Prefer explicit paper evidence over filename guesses.

Important:
- The attached PDF is the primary source. Use filename candidates only as fallback metadata hints.
- The paper may be about interventions or chatbots and may contain no predictive metrics. In that case set has_quantitative_metrics to false.
- Keep values exactly as reported when possible.
- Keep evidence snippets short.
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
                {"role": "system", "content": SYSTEM_PROMPT},
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
                instructions=SYSTEM_PROMPT,
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
    mapping = {
        "accuracy": "Accuracy",
        "acc": "Accuracy",
        "auc": "AUC",
        "roc auc": "AUC",
        "auroc": "AUC",
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
    return normalized


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
    return deduped or ["not_reported"]


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
        display_value = " / ".join(values) if values else "未提取"
        if contexts:
            display_value = f"{display_value} ({' / '.join(contexts)})"
        parts.append(f"{category}={display_value}")
    return "; ".join(parts)


def render_final_line(authors: str, year: str, metrics: list[dict[str, Any]], methods: list[str], fallback_key: str) -> str:
    citation_key = build_citation_key(authors, year, fallback_key)
    if not metrics:
        return f"[{citation_key}] | 指标: 无定量指标 | 数值: 无 | 方法: 未说明"

    categories = "; ".join(dict.fromkeys(metric["category"] for metric in metrics))
    values = render_metric_values(metrics)
    method_labels = "; ".join(METHOD_LABELS.get(method, method) for method in methods if method != "not_reported")
    method_labels = method_labels or "未说明"
    return f"[{citation_key}] | 指标: [{categories}] | 数值: [{values}] | 方法: [{method_labels}]"


def build_paper_record(
    paper_id: str,
    pdf_path: Path,
    filename_meta: dict[str, str],
    front_meta: dict[str, str],
    llm_payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = normalize_metric_items(llm_payload.get("metrics"))
    has_quantitative_metrics = bool(llm_payload.get("has_quantitative_metrics")) and bool(metrics)
    if not has_quantitative_metrics:
        metrics = []

    methods = normalize_evaluation_methods(llm_payload.get("evaluation_methods"))
    if not has_quantitative_metrics:
        methods = ["not_reported"]

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
        has_quantitative_metrics=has_quantitative_metrics,
        metrics=metrics,
        evaluation_methods=methods,
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

        if extractor.config.input_mode == "text":
            pages = read_pdf_pages(pdf_path)
            if not any(page.strip() for page in pages):
                raise RuntimeError("No extractable text found in PDF.")

            front_meta = extract_front_matter_candidates(pages[:2])
            context_text, selected_pages = select_context_pages(pages, char_budget)
            if not context_text:
                raise RuntimeError("No context selected from PDF text.")

            llm_payload = extractor.extract(
                build_text_mode_prompt(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    filename_meta=filename_meta,
                    front_matter_meta=front_meta,
                    context_text=context_text,
                )
            )
        else:
            llm_payload = extractor.extract(
                build_pdf_direct_prompt(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    filename_meta=filename_meta,
                ),
                pdf_path=pdf_path,
            )
        record = build_paper_record(
            paper_id=paper_id,
            pdf_path=pdf_path,
            filename_meta=filename_meta,
            front_meta=front_meta,
            llm_payload=llm_payload,
        )
        record["input_mode"] = extractor.config.input_mode
        record["context_page_numbers"] = selected_pages
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
    return {
        "paper_id": record.get("paper_id", ""),
        "input_mode": record.get("input_mode", ""),
        "title": record.get("title", ""),
        "authors": record.get("authors", ""),
        "year": record.get("year", ""),
        "pdf_path": record.get("pdf_path", ""),
        "has_quantitative_metrics": record.get("has_quantitative_metrics", False),
        "metric_categories": "; ".join(dict.fromkeys(metric.get("category", "") for metric in metrics if metric.get("category"))),
        "metric_values": render_metric_values(metrics),
        "evaluation_methods": "; ".join(METHOD_LABELS.get(method, method) for method in methods),
        "confidence": record.get("confidence", ""),
        "final_line": record.get("final_line", ""),
    }


def materialize_outputs(records_path: Path, summary_csv_path: Path, lines_txt_path: Path) -> None:
    records = read_jsonl(records_path)
    records.sort(key=lambda item: numeric_sort_key(str(item.get("paper_id", ""))))

    ensure_parent_dir(summary_csv_path)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "input_mode",
                "title",
                "authors",
                "year",
                "pdf_path",
                "has_quantitative_metrics",
                "metric_categories",
                "metric_values",
                "evaluation_methods",
                "confidence",
                "final_line",
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
    payload = {
        "provider": asdict(config),
        "paper_root": str(Path(args.paper_root).resolve()),
        "run_name": args.run_name,
        "limit": args.limit,
        "paper_ids": parse_requested_paper_ids(args.paper_id),
        "char_budget": args.char_budget,
        "input_mode": args.input_mode,
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
        max_retries=max(1, args.max_retries),
        concurrency=max(1, args.concurrency),
        keep_uploaded_files=args.keep_uploaded_files,
    )
    if args.input_mode == "pdf_direct" and args.provider.lower() != "openai":
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
