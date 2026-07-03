import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from github import Github, Auth

load_dotenv()

RAW_DIR = Path("data/raw")
REPO_NAME = "psf/requests"


def get_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in .env")
    return Github(auth=Auth.Token(token))


# collecting merged pull requests and saving them to a JSON file
def collect_prs(repo_name: str = REPO_NAME, max_prs: int = 500):
    g = get_client()
    repo = g.get_repo(repo_name)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    prs_data = []
    # closed PRs only — we need to know their outcome
    pulls = repo.get_pulls(state="closed", sort="created", direction="desc")

    merged_count = 0
    for pr in pulls:
        if merged_count >= max_prs:
            break

        if pr.merged_at is None:
            continue  # only merged PRs can introduce bugs

        record = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,  # description of the PR
            "author": pr.user.login if pr.user else None,
            "created_at": str(pr.created_at),
            "merged_at": str(pr.merged_at),
            "is_merged": True,
            "additions": pr.additions,  # lines added
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "commits": pr.commits,
            "comments": pr.comments,
            "review_comments": pr.review_comments,
            "files": [f.filename for f in pr.get_files()],
        }
        prs_data.append(record)
        merged_count += 1 

        if merged_count % 25 == 0:
            print(f"Collected {merged_count} merged PRs... "
                  f"(rate limit left: {g.get_rate_limit().resources.core.remaining})")
            # save progress as we go, so a crash doesn't lose everything
            save(prs_data, repo_name)

        time.sleep(0.3)

    save(prs_data, repo_name)
    print(f"Done. {len(prs_data)} merged PRs saved.")


def save(data, repo_name):
    filename = repo_name.replace("/", "_") + "_prs.json"
    with open(RAW_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    collect_prs()