import json
import time
from pathlib import Path

from collect import get_client, RAW_DIR, REPO_NAME


def collect_commits(repo_name: str = REPO_NAME, max_commits: int = 2000):
    g = get_client()
    repo = g.get_repo(repo_name)

    commits_data = []
    for i, commit in enumerate(repo.get_commits()):
        if i >= max_commits:
            break

        commits_data.append({
            "sha": commit.sha,
            "message": commit.commit.message,
            "date": str(commit.commit.author.date),
            "files": [f.filename for f in commit.files],
        })

        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1} commits...")
            save(commits_data, repo_name)

        time.sleep(0.2)

    save(commits_data, repo_name)
    print(f"Done. {len(commits_data)} commits saved.")


def save(data, repo_name):
    filename = repo_name.replace("/", "_") + "_commits.json"
    with open(RAW_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    collect_commits()