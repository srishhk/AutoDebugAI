from src.data.collect import get_client, REPO_NAME


def fetch_pr_features(pr_number: int, repo_name: str = REPO_NAME) -> dict:
    """Fetch one PR from GitHub and return the feature dict for score_pr."""
    g = get_client()
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    return {
        "additions": pr.additions,
        "deletions": pr.deletions,
        "changed_files": pr.changed_files,
        "commits": pr.commits,
        "comments": pr.comments,
        "review_comments": pr.review_comments,
        "num_files": pr.changed_files,
        "author_past_prs": 0,
        "author_past_bug_rate": 0.0,
        "is_first_pr": 1,
        "title": pr.title or "",
    }