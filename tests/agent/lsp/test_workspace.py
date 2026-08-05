"""Tests for workspace + project-root resolution."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.lsp.workspace import (
    clear_cache,
    find_git_worktree,
    is_inside_workspace,
    nearest_root,
    normalize_path,
    resolve_workspace_for_file,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()




def test_find_git_worktree_finds_dotgit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_git_worktree(str(sub)) == str(repo)


def test_find_git_worktree_ignores_system_temp_root(tmp_path: Path, monkeypatch):
    system_temp = tmp_path / "system-temp"
    nested = system_temp / "pytest-case"
    nested.mkdir(parents=True)
    (system_temp / ".git").mkdir()
    monkeypatch.setattr(
        "agent.lsp.workspace.tempfile.gettempdir", lambda: str(system_temp)
    )

    assert find_git_worktree(str(nested)) is None


def test_find_git_worktree_ignores_temp_root_symlink_alias(
    tmp_path: Path, monkeypatch
):
    system_temp = tmp_path / "system-temp"
    nested = system_temp / "pytest-case"
    nested.mkdir(parents=True)
    (system_temp / ".git").mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(system_temp, target_is_directory=True)
    monkeypatch.setattr(
        "agent.lsp.workspace.tempfile.gettempdir", lambda: str(system_temp)
    )

    assert find_git_worktree(str(alias / "pytest-case")) is None


def test_find_git_worktree_preserves_nested_temp_repo(tmp_path: Path, monkeypatch):
    system_temp = tmp_path / "system-temp"
    repo = system_temp / "project"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (system_temp / ".git").mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setattr(
        "agent.lsp.workspace.tempfile.gettempdir", lambda: str(system_temp)
    )

    assert find_git_worktree(str(nested)) == str(repo)








def test_nearest_root_finds_first_marker(tmp_path: Path):
    root = tmp_path / "p"
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    found = nearest_root(str(deep / "mod.py"), ["pyproject.toml"])
    assert found == str(root)






def test_resolve_workspace_for_file_uses_cwd_first(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    file_path = repo / "x.py"
    file_path.write_text("")
    # cwd is inside the repo
    monkeypatch.chdir(str(repo))
    root, gated = resolve_workspace_for_file(str(file_path))
    assert root == str(repo)
    assert gated is True






def test_normalize_path_expands_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/home/user")
    p = normalize_path("~/x.py")
    assert p == os.path.abspath("/home/user/x.py")
