"""CSV submission helper.

Run after a scoring script has written `submissions/submission.csv`:

    python -m src.submit --tag baseline_v1

Steps:
    1. Validates the CSV (id coverage, score range, dtype) against priv.pt ids.
    2. Archives a timestamped copy to submissions/history/.
    3. POSTs the file to the leaderboard server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = REPO_ROOT / "submissions" / "submission.csv"
HISTORY_DIR = REPO_ROOT / "submissions" / "history"
SECRETS_PATH = REPO_ROOT / "secrets.env"

BASE_URL = "http://34.63.153.158"
TASK_ID = "01-mia"


def read_api_key() -> str:
    if not SECRETS_PATH.exists():
        sys.exit(
            f"Missing {SECRETS_PATH}. Copy secrets.env.example to secrets.env "
            "and fill in TML_API_KEY."
        )
    for line in SECRETS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "TML_API_KEY":
            return v.strip().strip('"').strip("'")
    sys.exit(f"TML_API_KEY not found in {SECRETS_PATH}")


def validate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if list(df.columns) != ["id", "score"]:
        sys.exit(f"Bad columns {list(df.columns)} — expected ['id', 'score']")
    if df["id"].duplicated().any():
        sys.exit("Duplicate ids in submission")
    if df["score"].isna().any():
        sys.exit("NaN scores in submission")
    if not pd.api.types.is_numeric_dtype(df["score"]):
        sys.exit(f"Non-numeric scores: dtype={df['score'].dtype}")
    if (df["score"] < 0).any() or (df["score"] > 1).any():
        sys.exit("Scores out of [0, 1] range")
    print(f"Validated: {len(df)} rows, score range "
          f"[{df['score'].min():.4f}, {df['score'].max():.4f}]")
    return df


def archive(csv_path: Path, tag: str) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    dest = HISTORY_DIR / f"{stamp}_{tag}.csv"
    shutil.copy2(csv_path, dest)
    print(f"Archived to {dest}")
    return dest


def upload(csv_path: Path, api_key: str) -> dict:
    with open(csv_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": api_key},
            files={"file": (csv_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    if resp.status_code == 413:
        sys.exit("Upload rejected: file too large (HTTP 413).")
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    if not resp.ok:
        print(f"HTTP {resp.status_code}: {body}", file=sys.stderr)
        sys.exit(1)
    return body


def main():
    p = argparse.ArgumentParser(description="Submit submissions/submission.csv.")
    p.add_argument("--tag", required=True,
                   help="Short identifier for this submission, e.g. baseline_v1")
    p.add_argument("--no-upload", action="store_true",
                   help="Validate + archive only; skip the network upload.")
    a = p.parse_args()

    validate(SUBMISSION_PATH)
    archive(SUBMISSION_PATH, a.tag)

    if a.no_upload:
        print("Dry run: skipped upload.")
        return

    api_key = read_api_key()
    body = upload(SUBMISSION_PATH, api_key)
    print("Submitted. Server response:", body)


if __name__ == "__main__":
    main()
