def parse_hf_repo_id(hf_repo_id: str) -> tuple[str, str]:
    """Parse the repo_id and the revision from a hf_repo_id in the format:
    `org/repo-name@revision`.

    Returns a tuple (repo_id, revision), with revision = 'main' if it
    was not provided.
    """
    if "@" in hf_repo_id:
        hf_repo_id, revision = hf_repo_id.split("@", 1)
    else:
        revision = "main"

    return hf_repo_id, revision
