"""Repository Checkout and Sandbox Environment Manager for SWE-bench Instances.

Manages caching, cloning, and isolating target repositories at exact base commits via
Git worktrees, enabling authentic multi-turn agent exploration and test execution.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class RepoManager:
    """Clones, caches, and checks out SWE-bench repositories at exact commits using Git worktrees."""

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "swe_bench_repos"

    @classmethod
    def get_repo_cache_path(cls, repo_name: str, cache_dir: Optional[Path] = None) -> Path:
        base = cache_dir or cls.DEFAULT_CACHE_DIR
        safe_name = repo_name.replace("/", "__")
        return base / safe_name

    @classmethod
    def clone_or_fetch(cls, repo_name: str, cache_dir: Optional[Path] = None) -> Path:
        """Clone repository if not cached, or fetch latest commits."""
        repo_dir = cls.get_repo_cache_path(repo_name, cache_dir)
        if repo_dir.exists() and (repo_dir / ".git").exists():
            return repo_dir

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(repo_name):
            clone_url = str(Path(repo_name).resolve())
        elif "://" in repo_name or repo_name.startswith("git@"):
            clone_url = repo_name
        else:
            clone_url = f"https://github.com/{repo_name}.git"

        # Shallow clone initially or standard clone for small repos
        cmd = ["git", "clone", "--filter=blob:none", clone_url, str(repo_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Fallback to standard clone
            cmd_fallback = ["git", "clone", clone_url, str(repo_dir)]
            proc = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to clone repository {repo_name}: {proc.stderr}")

        return repo_dir

    @classmethod
    def create_instance_sandbox(
        cls,
        repo_name: str,
        base_commit: str,
        target_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """Prepare an isolated working directory checked out to base_commit via git worktree."""
        cached_repo = cls.clone_or_fetch(repo_name, cache_dir)

        dest_dir = target_dir or Path(tempfile.mkdtemp(prefix=f"swe_{repo_name.replace('/', '_')}_"))
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)

        # 1. Fetch base_commit so blobs are available in cache
        subprocess.run(
            ["git", "fetch", "origin", base_commit],
            cwd=str(cached_repo),
            capture_output=True,
            text=True,
        )

        # 2. Add detached worktree at base_commit
        wt_cmd = ["git", "worktree", "add", "--detach", str(dest_dir), base_commit]
        proc = subprocess.run(wt_cmd, cwd=str(cached_repo), capture_output=True, text=True)
        if proc.returncode != 0:
            # Fallback to local clone/copy if worktree fails
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(cached_repo), str(dest_dir), dirs_exist_ok=True)
            subprocess.run(["git", "checkout", base_commit], cwd=str(dest_dir), capture_output=True)

        return dest_dir

    @classmethod
    def remove_sandbox(
        cls,
        sandbox_dir: Path,
        repo_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Safely remove worktree and ephemeral sandbox directory."""
        if repo_name:
            cached_repo = cls.get_repo_cache_path(repo_name, cache_dir)
            if cached_repo.exists():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(sandbox_dir)],
                    cwd=str(cached_repo),
                    capture_output=True,
                )
                subprocess.run(["git", "worktree", "prune"], cwd=str(cached_repo), capture_output=True)

        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
