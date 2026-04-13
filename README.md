# Paper LLM Extract

Batch extraction toolkit for identifying and structuring quantitative predictive metrics from a local PDF paper collection.

This repository is designed for a workflow where the paper corpus stays local.

## What It Does

The main script scans a local `Paper/` directory and supports two extraction modes:

- `text`: read the PDF locally with `pypdf`, select evidence-focused text, and send only that text to the API
- `pdf_direct`: upload the local PDF itself through the Files API and pass it to the Responses API as an `input_file`

In both modes, the script extracts:

- whether the paper reports quantitative predictive performance metrics
- which metrics are reported
- the reported values
- the evaluation method
- basic paper metadata such as title, authors, and year

The current implementation is model-agnostic at the script level. You can switch between providers such as ChatGPT and DeepSeek by changing CLI arguments only.

## Repository Layout

```text
.
├── README.md
├── .gitignore
├── scripts/
│   └── extract_metrics_batch.py
└── tests/
    └── test_extract_metrics_batch.py
```

Local-only files are intentionally excluded from version control:

- `Paper/` for the PDF corpus
- `results/` for extraction outputs
- `docs/paper_metrics_extraction_guide.md` for internal development guidance

## Requirements

- Python 3.10+
- `openai`
- `pypdf`

Example install:

```bash
pip install openai pypdf
```

## Quick Start

Place your local paper folders under:

```text
Paper/<paper_id>/<paper_file>.pdf
```

Then run the extractor with an OpenAI-compatible API.

### OpenAI Example

Text mode:

```bash
export OPENAI_API_KEY=your_key

python3 scripts/extract_metrics_batch.py \
  --paper-root Paper \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1 \
  --api-key-env OPENAI_API_KEY \
  --input-mode text \
  --run-name chatgpt_gpt41_v1 \
  --limit 10
```

Direct PDF mode with official OpenAI:

```bash
export OPENAI_API_KEY=your_key

python3 scripts/extract_metrics_batch.py \
  --paper-root Paper \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-5 \
  --api-key-env OPENAI_API_KEY \
  --input-mode pdf_direct \
  --run-name chatgpt_gpt5_pdfdirect_v1 \
  --limit 10
```

### DeepSeek Example

```bash
export DEEPSEEK_API_KEY=your_key

python3 scripts/extract_metrics_batch.py \
  --paper-root Paper \
  --provider deepseek \
  --base-url https://api.deepseek.com/v1 \
  --model deepseek-chat \
  --api-key-env DEEPSEEK_API_KEY \
  --input-mode text \
  --run-name deepseek_chat_v1 \
  --limit 10
```

Remove `--limit` for a full run. Add `--resume` to continue an interrupted run without reprocessing completed papers.

## Outputs

Each run writes to `results/<run_name>/`:

- `records.jsonl`: full structured extraction records
- `summary.csv`: tabular summary for manual review
- `lines.txt`: one final delivery line per paper
- `errors.jsonl`: per-paper failures that did not stop the batch
- `run_config.json`: run-time configuration snapshot

## CLI Parameters

The script supports:

- `--paper-root`
- `--provider`
- `--base-url`
- `--model`
- `--api-key-env`
- `--run-name`
- `--input-mode`
- `--resume`
- `--limit`
- `--paper-id`
- `--char-budget`
- `--max-retries`
- `--concurrency`
- `--keep-uploaded-files`

See full help with:

```bash
python3 scripts/extract_metrics_batch.py --help
```

## Development Notes

- The extractor uses `pypdf` for local text extraction.
- It sends only a selected subset of pages to the LLM rather than the full PDF text.
- In `pdf_direct` mode, it uploads the local PDF to the OpenAI Files API and deletes the uploaded file after each request by default.
- It does not hard-code provider-specific logic.
- It is built to keep running even if individual PDFs fail.

## Testing

Run unit tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Publication Guidance

If you push this repository to GitHub, make sure your API keys stay in environment variables and never in tracked files. The included `.gitignore` already excludes the local paper corpus, generated results, and local planning notes.
