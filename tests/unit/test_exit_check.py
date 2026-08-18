"""unit: scripts/exit_check 会话退出检查——full/partial/info 三档编排、回归计数、清洁/git 检查、退出码。"""
import subprocess
from pathlib import Path

import pytest

import scripts.exit_check as ec


class TestRunChecks:
    def test_all_pass_exit_zero(self, capsys):
        checks = [("c1", lambda: (True, "")), ("c2", lambda: (True, "detail"))]
        summary = ec.run_checks(checks)
        assert summary.exit_code == 0
        assert summary.results == [("c1", True, ""), ("c2", True, "detail")]
        assert "[PASS] c1" in capsys.readouterr().out

    def test_fail_exit_one(self):
        checks = [("ok", lambda: (True, "")), ("bad", lambda: (False, "boom"))]
        assert ec.run_checks(checks).exit_code == 1

    def test_warn_does_not_affect_exit(self, capsys):
        assert ec.run_checks([("warn", lambda: (None, "注意"))]).exit_code == 0
        assert "[warn] warn" in capsys.readouterr().out

    def test_check_exception_counts_as_fail(self, capsys):
        def boom():
            raise RuntimeError("内部炸了")

        assert ec.run_checks([("c", boom)]).exit_code == 1
        assert "检查自身异常" in capsys.readouterr().out


class TestBuildChecks:
    def test_contains_all_gates(self):
        names = [name for name, _ in ec.build_checks()]
        expected = {
            "依赖一致",
            "测试全绿",
            "治理无回归",
            "治理无回归-存量",
            "进度快照",
            "启动可用",
            "清洁检查-编译残留",
            "清洁检查-源码残留",
            "git 检查",
        }
        assert expected.issubset(names)

    def test_report_after_verify_all(self):
        names = [name for name, _ in ec.build_checks()]
        assert names.index("治理无回归-存量") < names.index("进度快照")


class TestCountRegressed:
    def test_counts_regressed_items(self, tmp_path):
        features = tmp_path / "features.yaml"
        features.write_text(
            "items:\n"
            "- id: F1\n  state: passed\n"
            "- id: F2\n  state: regressed\n"
            "- id: F3\n  state: regressed\n",
            encoding="utf-8",
        )
        assert ec._count_regressed(features) == 2

    def test_no_regressed_zero(self, tmp_path):
        features = tmp_path / "features.yaml"
        features.write_text("items:\n- id: F1\n  state: passed\n", encoding="utf-8")
        assert ec._count_regressed(features) == 0

    def test_corrupt_yaml_zero(self, tmp_path):
        features = tmp_path / "features.yaml"
        features.write_text("{{{{{{", encoding="utf-8")
        assert ec._count_regressed(features) == 0

    def test_regressed_check_fail(self, tmp_path, monkeypatch):
        features = tmp_path / "features.yaml"
        features.write_text("items:\n- id: F2\n  state: regressed\n", encoding="utf-8")
        monkeypatch.setattr(ec, "FEATURES_PATH", features)
        passed, detail = ec._check_regressed_count()
        assert passed is False
        assert "regressed" in detail


class TestCleanliness:
    def test_tracked_bytecode_fail(self, monkeypatch):
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, "scripts/a.pyc\nscripts/b.py\n"))
        passed, detail = ec._check_tracked_bytecode()
        assert passed is False
        assert "a.pyc" in detail

    def test_no_tracked_bytecode_pass(self, monkeypatch):
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, "scripts/a.py\nscripts/b.py\n"))
        passed, _ = ec._check_tracked_bytecode()
        assert passed is True

    def test_uppercase_todo_hits(self, tmp_path):
        source = tmp_path / "module.py"
        source.write_text("x = 1  # TODO: finish\n", encoding="utf-8")
        hits = ec._collect_todo_hits([source])
        assert any("module.py:1" in hit for hit in hits)

    def test_lowercase_placeholder_not_hit(self, tmp_path):
        source = tmp_path / "module.py"
        source.write_text("from eval.xxx import foo\n", encoding="utf-8")
        assert ec._collect_todo_hits([source]) == []

    def test_source_todos_warn(self, monkeypatch):
        monkeypatch.setattr(ec, "_collect_todo_hits", lambda files: ["scripts/foo.py:3"])
        passed, detail = ec._check_source_todos()
        assert passed is None
        assert "TODO" in detail

    def test_source_todos_clean_pass(self, monkeypatch):
        monkeypatch.setattr(ec, "_collect_todo_hits", lambda files: [])
        passed, _ = ec._check_source_todos()
        assert passed is True

    def test_build_source_files_excludes_tool_itself(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ec, "_SOURCE_DIRS", ("scripts",))
        monkeypatch.setattr(ec, "_SOURCE_FILES", ())
        files = ec._build_source_files()
        resolved = [str(path.resolve()) for path in files]
        assert str(Path(ec.__file__).resolve()) not in resolved


class TestGitCheck:
    def test_parse_porcelain_modified_untracked_deleted(self):
        output = " M retrieval/retriever.py\n?? scripts/exit_check.py\nD  eval/monitor/__init__.py\n"
        paths = ec._parse_porcelain(output)
        assert "retrieval/retriever.py" in paths
        assert "scripts/exit_check.py" in paths
        assert "eval/monitor/__init__.py" in paths

    def test_parse_porcelain_rename_takes_new(self):
        assert ec._parse_porcelain("R  old.py -> new.py\n") == ["new.py"]

    def test_is_self_produced(self):
        assert ec._is_self_produced("harness/features.yaml")
        assert ec._is_self_produced("harness/features.md")
        assert ec._is_self_produced("PROGRESS.md")
        assert not ec._is_self_produced("retrieval/retriever.py")

    def test_self_produced_only_pass(self, monkeypatch):
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, " M harness/features.yaml\n M PROGRESS.md\n"))
        passed, _ = ec._check_git_dirty()
        assert passed is True

    def test_other_change_warns(self, monkeypatch):
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, " M retrieval/retriever.py\n"))
        passed, detail = ec._check_git_dirty()
        assert passed is None
        assert "retriever.py" in detail


class TestRun:
    def test_nonzero_exit_output_decoded(self, monkeypatch):
        def fake_run(cmd, timeout, **kwargs):
            class Process:
                returncode = 1
                stdout = "错误: 坏了"
                stderr = ""

            return Process()

        monkeypatch.setattr(subprocess, "run", fake_run)
        code, output = ec._run(["echo"], 10)
        assert code == 1
        assert "坏了" in output

    def test_timeout_counts_as_failure(self, monkeypatch):
        def fake_run(cmd, timeout, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="partial")

        monkeypatch.setattr(subprocess, "run", fake_run)
        code, output = ec._run(["echo"], 10)
        assert code == 1
        assert "timeout" in output

    def test_bytes_timeout_output_decoded(self, monkeypatch):
        def fake_run(cmd, timeout, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=b"\xff\xfe")

        monkeypatch.setattr(subprocess, "run", fake_run)
        code, output = ec._run(["echo"], 10)
        assert code == 1
        assert "timeout" in output

    def test_oserror_counts_as_failure(self, monkeypatch):
        def fake_run(cmd, timeout, **kwargs):
            raise OSError("not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        code, output = ec._run(["nope"], 10)
        assert code == 1
        assert "无法启动" in output

    def test_non_gbk_char_no_crash(self, monkeypatch):
        def fake_run(cmd, timeout, **kwargs):
            class Process:
                returncode = 1
                stdout = "¥ 符号"
                stderr = ""

            return Process()

        monkeypatch.setattr(subprocess, "run", fake_run)
        code, output = ec._run(["echo"], 10)
        assert "¥" in output


class TestModes:
    def test_partial_always_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, ""))
        assert ec.main(["--mode", "partial"]) == 0
        assert "partial" in capsys.readouterr().out

    def test_info_always_zero(self, monkeypatch, tmp_path, capsys):
        features = tmp_path / "features.yaml"
        features.write_text("items:\n- id: F1\n  state: passed\n", encoding="utf-8")
        monkeypatch.setattr(ec, "FEATURES_PATH", features)
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, ""))
        assert ec.main(["--mode", "info"]) == 0
        assert "passed" in capsys.readouterr().out

    def test_full_default_exit_one_with_fail(self, monkeypatch):
        monkeypatch.setattr(ec, "build_checks", lambda: [("bad", lambda: (False, ""))])
        assert ec.main([]) == 1

    def test_full_default_exit_zero_all_pass(self, monkeypatch):
        monkeypatch.setattr(ec, "build_checks", lambda: [("ok", lambda: (True, ""))])
        assert ec.main([]) == 0

    def test_full_with_regressed_fails(self, tmp_path, monkeypatch):
        features = tmp_path / "features.yaml"
        features.write_text("items:\n- id: F2\n  state: regressed\n", encoding="utf-8")
        monkeypatch.setattr(ec, "FEATURES_PATH", features)
        monkeypatch.setattr(ec, "_SOURCE_DIRS", ())
        monkeypatch.setattr(ec, "_SOURCE_FILES", ())
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, ""))
        assert ec.main([]) == 1

    def test_full_without_regressed_passes(self, tmp_path, monkeypatch):
        features = tmp_path / "features.yaml"
        features.write_text("items:\n- id: F1\n  state: passed\n", encoding="utf-8")
        monkeypatch.setattr(ec, "FEATURES_PATH", features)
        monkeypatch.setattr(ec, "_SOURCE_DIRS", ())
        monkeypatch.setattr(ec, "_SOURCE_FILES", ())
        monkeypatch.setattr(ec, "_run", lambda cmd, timeout: (0, ""))
        assert ec.main([]) == 0

    def test_invalid_mode_errors(self):
        with pytest.raises(SystemExit):
            ec.main(["--mode", "bogus"])
