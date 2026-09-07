"""unit: scripts/corpus_versioning 语料本地版本化(011-5)——SHA 清单、独立仓库快照/校验/恢复、CLI。"""
import json
from pathlib import Path

import pytest

import scripts.corpus_versioning as cv


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测纯文件/git 逻辑, 无需重依赖(避免 ~18s 导入成本)。"""
    yield


def _make_corpus(root: Path) -> None:
    """在 root 下构造最小语料: data/ 两篇 md + benchmark 一个标注。"""
    (root / "data" / "Agent").mkdir(parents=True)
    (root / "data" / "Python").mkdir(parents=True)
    (root / "data" / "Agent" / "a.md").write_text("# 标题A\n正文A", encoding="utf-8")
    (root / "data" / "Python" / "b.md").write_text("# 标题B\n正文B", encoding="utf-8")
    (root / "benchmark").mkdir(parents=True)
    (root / "benchmark" / "private_v6.json").write_text("[]", encoding="utf-8")


class TestBuildManifest:
    def test_manifest_covers_data_and_benchmark(self, tmp_path):
        _make_corpus(tmp_path)
        manifest = cv.build_manifest(tmp_path)
        assert "data/Agent/a.md" in manifest["files"]
        assert "data/Python/b.md" in manifest["files"]
        assert "benchmark/private_v6.json" in manifest["files"]
        assert len(manifest["files"]) == 3

    def test_manifest_sha_is_stable(self, tmp_path):
        _make_corpus(tmp_path)
        first = cv.build_manifest(tmp_path)
        second = cv.build_manifest(tmp_path)
        assert first["files"] == second["files"]

    def test_missing_benchmark_json_is_skipped(self, tmp_path):
        _make_corpus(tmp_path)
        (tmp_path / "benchmark" / "private_v6.json").unlink()
        manifest = cv.build_manifest(tmp_path)
        assert "benchmark/private_v6.json" not in manifest["files"]


class TestSnapshotVerifyRestore:
    def test_snapshot_creates_repo_and_manifest(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        commit = cv.snapshot_corpus(tmp_path, repo, "first")
        assert commit
        assert (repo / ".git").exists()
        assert (repo / ".corpus-manifest.json").exists()
        assert (repo / "data" / "Agent" / "a.md").read_text(encoding="utf-8") == "# 标题A\n正文A"

    def test_verify_passes_after_snapshot(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.snapshot_corpus(tmp_path, repo, "first")
        ok, mismatches = cv.verify_corpus(tmp_path, repo)
        assert ok
        assert mismatches == []

    def test_verify_detects_modification(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.snapshot_corpus(tmp_path, repo, "first")
        (tmp_path / "data" / "Agent" / "a.md").write_text("# 改过\n", encoding="utf-8")
        ok, mismatches = cv.verify_corpus(tmp_path, repo)
        assert not ok
        assert any("data/Agent/a.md" in item for item in mismatches)

    def test_verify_detects_missing_file(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.snapshot_corpus(tmp_path, repo, "first")
        (tmp_path / "data" / "Python" / "b.md").unlink()
        ok, mismatches = cv.verify_corpus(tmp_path, repo)
        assert not ok
        assert any("data/Python/b.md" in item for item in mismatches)

    def test_restore_recovers_files(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.snapshot_corpus(tmp_path, repo, "first")
        # 破坏语料后恢复
        (tmp_path / "data" / "Agent" / "a.md").write_text("损坏", encoding="utf-8")
        (tmp_path / "data" / "Python" / "b.md").unlink()
        count = cv.restore_corpus(repo, tmp_path)
        assert count == 3
        assert (tmp_path / "data" / "Agent" / "a.md").read_text(encoding="utf-8") == "# 标题A\n正文A"
        assert (tmp_path / "data" / "Python" / "b.md").exists()

    def test_restore_without_manifest_raises(self, tmp_path):
        repo = tmp_path / "empty-repo"
        repo.mkdir()
        with pytest.raises(RuntimeError):
            cv.restore_corpus(repo, tmp_path)


class TestCli:
    def test_snapshot_verify_restore_flow(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        monkeypatch.setattr(cv, "_PROJECT_ROOT", tmp_path)
        repo = tmp_path / "corpus-repo"
        assert cv.main(["snapshot", "--repo", str(repo), "-m", "cli test"]) == 0
        assert cv.main(["verify", "--repo", str(repo)]) == 0
        assert cv.main(["manifest", "--repo", str(repo)]) == 0
        assert cv.main(["restore", "--repo", str(repo)]) == 0

    def test_verify_fails_after_tamper(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        monkeypatch.setattr(cv, "_PROJECT_ROOT", tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.main(["snapshot", "--repo", str(repo)])
        (tmp_path / "data" / "Agent" / "a.md").write_text("tampered", encoding="utf-8")
        assert cv.main(["verify", "--repo", str(repo)]) == 1

    def test_repo_option_before_subcommand(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        monkeypatch.setattr(cv, "_PROJECT_ROOT", tmp_path)
        repo = tmp_path / "corpus-repo"
        assert cv.main(["--repo", str(repo), "snapshot", "-m", "before"]) == 0
        assert cv.main(["--repo", str(repo), "verify"]) == 0

    def test_unknown_command_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            cv.main(["bogus"])


class TestEdgeCases:
    def test_git_failure_raises(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(RuntimeError):
            cv._git(repo, "not-a-real-git-command")

    def test_ensure_repo_is_idempotent(self, tmp_path):
        repo = tmp_path / "repo"
        cv._ensure_repo(repo)
        cv._ensure_repo(repo)  # 二次调用不报错
        assert (repo / ".git").exists()

    def test_verify_without_manifest_fails(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        ok, mismatches = cv.verify_corpus(tmp_path, repo)
        assert not ok
        assert mismatches == ["独立仓库无 manifest, 尚未快照"]

    def test_restore_skips_missing_repo_file(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        cv.snapshot_corpus(tmp_path, repo, "first")
        # 从仓库删除一个文件, restore 应跳过它而不报错
        (repo / "data" / "Python" / "b.md").unlink()
        count = cv.restore_corpus(repo, tmp_path)
        assert count == 2

    def test_snapshot_second_commit_keeps_history(self, tmp_path):
        _make_corpus(tmp_path)
        repo = tmp_path / "corpus-repo"
        first = cv.snapshot_corpus(tmp_path, repo, "first")
        (tmp_path / "data" / "Agent" / "a.md").write_text("# 标题A\n正文A2", encoding="utf-8")
        second = cv.snapshot_corpus(tmp_path, repo, "second")
        assert first != second
        log = cv._git(repo, "log", "--oneline")
        assert "second" in log
        assert "first" in log
