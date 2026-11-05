#!/usr/bin/env python3
"""Bump the DuckDB version pinned by this extension.

Updates:
  - duckdb submodule              -> checked out at <new_version>
  - .github/workflows/MainDistributionPipeline.yml:
      * duckdb-stable-build job: duckdb_version
      * code-quality-check job:  duckdb_version

The extension-ci-tools submodule and refs (uses / ci_tools_version) are
pinned by hand and left untouched.

Designed to run identically on a developer laptop and inside the
bump-duckdb workflow_dispatch GitHub Action.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "MainDistributionPipeline.yml"

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")

logger = logging.getLogger("bump-duckdb")


def die(msg: str, code: int = 1) -> "None":
    logger.error(msg)
    sys.exit(code)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True,
        capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_out(args: list[str], cwd: Path) -> str:
    return run(["git", *args], cwd=cwd, capture=True).stdout.strip()


def ensure_clean_tree() -> None:
    status = git_out(["status", "--porcelain"], REPO_ROOT)
    if status:
        die(
            "working tree is not clean — commit or stash changes first:\n"
            f"{status}"
        )


def detect_current_version() -> str:
    text = WORKFLOW_PATH.read_text()
    m = re.search(
        r"^\s+duckdb_version:\s*(v\d+\.\d+\.\d+)\s*$", text, re.MULTILINE
    )
    if not m:
        die("could not detect current DuckDB version from workflow file")
    return m.group(1)  # type: ignore[union-attr]


def remote_ref_exists(submodule: Path, ref: str) -> bool:
    """Return True if `ref` exists upstream as either a tag or a branch.

    DuckDB publishes versioned releases as tags; extension-ci-tools publishes
    them as branches with the same name (e.g. `v1.5.2`). Accept either."""
    remote_url = git_out(["config", "--get", "remote.origin.url"], submodule)
    result = run(
        [
            "git", "ls-remote", remote_url,
            f"refs/tags/{ref}", f"refs/heads/{ref}",
        ],
        capture=True,
        check=False,
    )
    return bool(result.stdout.strip())


def update_submodule(path: str, new_version: str) -> None:
    submodule = REPO_ROOT / path
    if not submodule.exists():
        die(f"submodule path does not exist: {path}")

    logger.info(f"checking out {new_version} in {path}")
    # Try the bare ref first (works for tags and local branches); fall back to
    # the remote-tracking branch in detached-HEAD mode for repos like
    # extension-ci-tools that publish versions as branches.
    result = run(
        ["git", "checkout", new_version],
        cwd=submodule, check=False, capture=True,
    )
    if result.returncode != 0:
        run(
            ["git", "-c", "advice.detachedHead=false",
             "checkout", f"origin/{new_version}"],
            cwd=submodule,
        )
    run(["git", "add", path], cwd=REPO_ROOT)


def _replace_in_job(text: str, job: str, line_re: str, old: str, new: str) -> str:
    """Replace `old` with `new` on the first line within `job`'s block that
    matches `line_re`. The block is bounded by the next top-level job header
    (two-space indent + name + colon) or end of file."""
    pattern = re.compile(
        r"(  " + re.escape(job) + r":\n(?:(?!^  \w[\w-]*:).*\n)*?" + line_re + r")"
        + re.escape(old),
        re.MULTILINE,
    )
    text, n = pattern.subn(r"\g<1>" + new, text, count=1)
    if n != 1:
        die(f"failed to update {job}: pattern {line_re!r} matched {n} times")
    return text


def update_workflow(old: str, new: str) -> None:
    text = WORKFLOW_PATH.read_text()
    original = text

    duckdb_line = r"      duckdb_version: "

    for job in ("duckdb-stable-build", "code-quality-check"):
        text = _replace_in_job(text, job, duckdb_line, old, new)

    if text == original:
        die("workflow file would be unchanged — already on this version?")

    WORKFLOW_PATH.write_text(text)
    logger.info(f"updated {WORKFLOW_PATH.relative_to(REPO_ROOT)}")
    run(["git", "add", str(WORKFLOW_PATH.relative_to(REPO_ROOT))], cwd=REPO_ROOT)


def write_github_output(pairs: dict[str, str]) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a") as f:
        for k, v in pairs.items():
            f.write(f"{k}={v}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new_version", help="new DuckDB version tag, e.g. v1.5.2")
    ap.add_argument(
        "--branch",
        help="branch name to create (default: feature/update-duckdb-to-<version>)",
    )
    ap.add_argument(
        "--no-commit",
        action="store_true",
        help="apply changes and stage them, but do not commit, branch, or push",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="push the branch to origin after committing",
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable DEBUG-level logging",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    new_version: str = args.new_version
    if not VERSION_RE.match(new_version):
        die(f"version must match vMAJOR.MINOR.PATCH, got: {new_version}")

    if not WORKFLOW_PATH.exists():
        die(f"workflow file not found: {WORKFLOW_PATH}")

    ensure_clean_tree()

    old_version = detect_current_version()
    if old_version == new_version:
        die(f"already on {new_version}; nothing to do")
    logger.info(f"bumping {old_version} -> {new_version}")

    branch = args.branch or f"feature/update-duckdb-to-{new_version}"

    # Verify the tag exists upstream before mutating anything (branch,
    # submodule HEADs, workflow file). This way a missing-tag failure leaves
    # the working tree exactly as we found it, so a rerun once upstream
    # publishes the tag is a clean redo.
    for path in ("duckdb",):
        submodule = REPO_ROOT / path
        logger.info(f"fetching refs for {path}")
        run(["git", "fetch", "--tags", "origin"], cwd=submodule)
        run(["git", "fetch", "origin", new_version], cwd=submodule, check=False)
        if not remote_ref_exists(submodule, new_version):
            die(
                f"ref {new_version} is not available on the {path} remote yet "
                "(checked tags and branches) — wait for upstream to publish "
                "it, then rerun"
            )

    if not args.no_commit:
        run(["git", "checkout", "-b", branch], cwd=REPO_ROOT)
        logger.info(f"created branch {branch}")

    update_submodule("duckdb", new_version)
    update_workflow(old_version, new_version)

    logger.info("staged diff:")
    run(["git", "--no-pager", "diff", "--cached", "--stat"], cwd=REPO_ROOT)

    if args.no_commit:
        logger.info("--no-commit set; leaving changes staged")
    else:
        commit_msg = f"build: bump to DuckDB {new_version}"
        run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT)
        logger.info(f"committed: {commit_msg}")

        if args.push:
            run(["git", "push", "-u", "origin", branch], cwd=REPO_ROOT)
            logger.info(f"pushed {branch} to origin")

    write_github_output({
        "old_version": old_version,
        "new_version": new_version,
        "branch": branch,
        "commit_message": f"build: bump to DuckDB {new_version}",
    })
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
