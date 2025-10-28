# Compliance Report Export

Automates the daily export of ChatGPT Enterprise Compliance API usage data into partitioned Parquet datasets that can be consumed by downstream analytics tooling.

## Features
- Pulls conversations, users, and projects for a workspace and aggregates user-level metrics.
- Filters to active users and counts only messages within the requested UTC day.
- Produces JSON map columns for model, GPT, tool, and project usage as defined by the compliance reporting spec.
- Writes separate user and GPT usage datasets under partitioned `date=YYYY-MM-DD` directories for easy ingestion into data warehouses.
- Respects Compliance API pagination by batching up to 100 active users per request when fetching conversations, preventing `since_timestamp`/`after` conflicts and keeping within per-endpoint rate limits.

## Requirements
- Python 3.9+
- Compliance API access with the `read` scope.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Copy `.env.sample` to `.env` and fill in your workspace details:

```bash
cp .env.sample .env
```

Variables:
- `CAPI_API_KEY` – Compliance API key (`sk-proj-...`)
- `WORKSPACE_ID` – Target workspace UUID
- `CAPI_BASE_URL` – Base URL (defaults to `https://api.chatgpt.com/v1`)
- `OUTPUT_DIR` – Directory for Parquet output (defaults to current directory)

## Usage
Run the script for a specific UTC date:

```bash
python usage_report.py --date 2025-10-21
```

If `--date` is omitted, the script defaults to yesterday’s UTC day.

## Output
The script writes two Parquet files per day, partitioned by UTC date:

```
OUTPUT_DIR/
  users/
    date=YYYY-MM-DD/
      part-00000.parquet
  gpts/
    date=YYYY-MM-DD/
      part-00000.parquet
```

The user dataset contains one row per workspace user with metrics aligned to the compliance reporting schema. The GPT dataset contains one row per GPT configuration with daily usage and unique messager counts.
