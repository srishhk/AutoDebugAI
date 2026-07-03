import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd 

RAW_DIR= Path("data/raw")
PROCESSED_DIR= Path("data/processed")
REPO_NAME= "psf/requests"

# words that mark a commit as a "police report" (bug-fix commit)
FIX_KEYWORDS = re.compile(r"\b(fix|fixes|fixed|bug|bugfix|hotfix|patch|resolve|resolves|resolved)\b", re.IGNORECASE)

# how long after a PR's merge we still consider a fix "connected" to it
WINDOW_DAYS = 90


def load_json(name):
    filename = REPO_NAME.replace("/", "_") + f"_{name}.json"
    with open(RAW_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def parse_date(s):
    # dates were saved as strings like "2026-07-01 21:58:05+00:00"
    return datetime.fromisoformat(s)


def find_fix_commits(commits):
    """Step 1: filter commits down to the 'police reports'."""
    fixes = []
    for c in commits:
        if FIX_KEYWORDS.search(c["message"]):
            fixes.append({
                "date": parse_date(c["date"]),
                "files": set(c["files"]),
                "message": c["message"].splitlines()[0],  # first line only
            })
    return fixes


def label_prs(prs, fix_commits):
    """Steps 2 & 3: match each PR against fix commits, assign label."""
    rows = []
    for pr in prs:
        merged_at = parse_date(pr["merged_at"])
        pr_files = set(pr["files"])
        window_end = merged_at + timedelta(days=WINDOW_DAYS)

        label = 0
        matched_fix = None
        for fix in fix_commits:
            # fix must come AFTER the merge, within the window,
            # and share at least one file with the PR
            if merged_at < fix["date"] <= window_end and pr_files & fix["files"]:
                label = 1
                matched_fix = fix["message"]
                break

        rows.append({
            "number": pr["number"],
            "title": pr["title"],
            "author": pr["author"],
            "created_at": pr["created_at"],
            "merged_at": pr["merged_at"],
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "changed_files": pr["changed_files"],
            "commits": pr["commits"],
            "comments": pr["comments"],
            "review_comments": pr["review_comments"],
            "num_files": len(pr_files),
            "label": label,
            "matched_fix": matched_fix,  # for eyeballing, not for training!
        })
    return pd.DataFrame(rows)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prs = load_json("prs")
    commits = load_json("commits")
    print(f"Loaded {len(prs)} PRs and {len(commits)} commits")

    fix_commits = find_fix_commits(commits)
    print(f"Found {len(fix_commits)} fix commits ('police reports')")

    df = label_prs(prs, fix_commits)
    buggy = df["label"].sum()
    print(f"Labeled: {buggy} buggy ({buggy / len(df):.1%}), {len(df) - buggy} safe")

    out = PROCESSED_DIR / "labeled_prs.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()