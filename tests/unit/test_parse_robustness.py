"""unit: 012-2 F41 运行时健壮性——失败分类/熔断/统计/warm_up/失败清单/subprocess 执行。"""
import json
import subprocess

import pytest

from indexing.parse_backends import robustness
from indexing.parse_backends.failure_list import FailureList, reset_failure_list
from indexing.parse_backends import subprocess_runner


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测健壮性逻辑, 无需重依赖。"""
    yield


@pytest.fixture(autouse=True)
def _reset():
    reset_failure_list()
    yield
    reset_failure_list()


class TestClassifyError:
    def test_timeout(self):
        assert robustness.classify_error(TimeoutError("t")) == robustness.CATEGORY_TIMEOUT

    def test_vlm_unavailable(self):
        assert robustness.classify_error(RuntimeError("VLM 不可用"), "vlm") == robustness.CATEGORY_VLM_UNAVAILABLE

    def test_colpali_unavailable(self):
        assert robustness.classify_error(RuntimeError("ColPali 不可用"), "colpali") == robustness.CATEGORY_COLPALI_UNAVAILABLE

    def test_import_error(self):
        assert robustness.classify_error(RuntimeError("docling 未安装")) == robustness.CATEGORY_IMPORT_ERROR

    def test_model_load_error(self):
        assert robustness.classify_error(RuntimeError("模型加载失败")) == robustness.CATEGORY_MODEL_LOAD_ERROR

    def test_empty(self):
        assert robustness.classify_error(RuntimeError("产出为空")) == robustness.CATEGORY_EMPTY

    def test_quality_fail(self):
        assert robustness.classify_error(RuntimeError("质量未达标")) == robustness.CATEGORY_QUALITY_FAIL

    def test_parse_error_default(self):
        assert robustness.classify_error(RuntimeError("boom")) == robustness.CATEGORY_PARSE_ERROR


class TestCircuitBreaker:
    def test_opens_after_three_failures(self):
        cb = robustness.CircuitBreaker(max_failures=3)
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        assert cb.is_open("docling") is False
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        assert cb.is_open("docling") is True

    def test_import_error_opens_immediately(self):
        cb = robustness.CircuitBreaker(max_failures=3)
        cb.record_failure("docling", robustness.CATEGORY_IMPORT_ERROR)
        assert cb.is_open("docling") is True

    def test_success_resets_counter(self):
        cb = robustness.CircuitBreaker(max_failures=3)
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        cb.record_success("docling")
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        assert cb.is_open("docling") is False

    def test_reset(self):
        cb = robustness.CircuitBreaker(max_failures=1)
        cb.record_failure("docling", robustness.CATEGORY_PARSE_ERROR)
        assert cb.is_open("docling") is True
        cb.reset()
        assert cb.is_open("docling") is False


class TestRunStats:
    def test_summary(self):
        stats = robustness.RunStats()
        stats.record("docling", 1.0, ok=True)
        stats.record("docling", 3.0, ok=True)
        stats.record("mineru", 2.0, ok=False, reason=robustness.CATEGORY_TIMEOUT)
        stats.record("docling", 5.0, ok=True)
        s = stats.summary()
        assert s["total"] == 4
        assert s["success"] == 3
        assert s["backend_distribution"]["docling"] == 3
        assert s["degradation_reasons_top5"][0][0] == robustness.CATEGORY_TIMEOUT
        assert s["colpali_triggered"] == 0  # L6: colpali 触发率只经 record_colpali_trigger 接线
        assert s["p50_p95"]["docling"]["p50"] == 3.0


class TestWarmUp:
    def test_probe_success_and_failure(self):
        def _probe(name):
            if name == "bad":
                raise RuntimeError("boom")
            return "1.0"
        result = robustness.warm_up(["good", "bad"], _probe, 5.0)
        assert result == {"good": True, "bad": False}


class TestFailureList:
    def test_record_and_write(self, tmp_path):
        fl = FailureList(tmp_path / "failure_list.json")
        fl.record("a.pdf", "text", "parse_error", ["docling:parse_error"], "1.0")
        fl.write()
        data = json.loads((tmp_path / "failure_list.json").read_text(encoding="utf-8"))
        assert data["records"][0]["file_path"] == "a.pdf"
        assert data["records"][0]["route_decision"] == "text"
        assert data["records"][0]["backend_attempts"] == ["docling:parse_error"]

    def test_reset_clears(self):
        from indexing.parse_backends.failure_list import failure_list
        failure_list.record("a.pdf", "text", "x")
        assert len(failure_list) == 1
        reset_failure_list()
        assert len(failure_list) == 0

    def test_record_writes_incrementally(self, tmp_path):
        # H2: record() 即增量落盘, 无需显式 write() 也保证持久化
        fl = FailureList(tmp_path / "failure_list.json")
        fl.record("a.pdf", "text", "parse_error", ["docling:parse_error"], "1.0")
        data = json.loads((tmp_path / "failure_list.json").read_text(encoding="utf-8"))
        assert data["records"][0]["file_path"] == "a.pdf"
        assert data["records"][0]["version_string"] == "1.0"


class TestRunStatsReset:
    def test_reset_clears_all_fields(self):
        stats = robustness.RunStats()
        stats.record("docling", 1.0, ok=True)
        stats.record("mineru", 2.0, ok=False, reason=robustness.CATEGORY_TIMEOUT)
        stats.record_colpali_trigger()
        stats.reset()
        s = stats.summary()
        assert s["total"] == 0
        assert s["success"] == 0
        assert s["colpali_triggered"] == 0
        assert s["backend_distribution"] == {}

    def test_record_colpali_trigger(self):
        # L6: 独立接线 ColPali 触发率统计
        stats = robustness.RunStats()
        stats.record_colpali_trigger()
        stats.record_colpali_trigger()
        assert stats.summary()["colpali_triggered"] == 2


class TestSubprocessRunnerInProcess:
    def test_run_extract_in_process(self, monkeypatch, tmp_path):
        from indexing.parse_backends import ParseResult
        class _Fake:
            def extract(self, path):
                return ParseResult(markdown="# 标题", format_meta={"backend": "docling"})
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", False)
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        markdown, meta = subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)
        assert markdown == "# 标题"
        assert meta["backend"] == "docling"

    def test_run_extract_in_process_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", False)
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: None)
        with pytest.raises(RuntimeError, match="不可用"):
            subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)

    def test_probe_in_process(self, monkeypatch):
        class _Fake:
            def version(self):
                return "2.0"
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", False)
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        assert subprocess_runner.probe("docling", 10) == "2.0"


class TestSubprocessRunnerSubprocess:
    def test_run_extract_subprocess(self, monkeypatch, tmp_path):
        payload = json.dumps({"markdown": "# 标题", "format_meta": {"backend": "docling"}})
        class _Completed:
            returncode = 0
            stdout = payload + "\n"
            stderr = ""
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", True)
        monkeypatch.setattr(subprocess_runner.subprocess, "run", lambda *a, **k: _Completed())
        markdown, meta = subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)
        assert markdown == "# 标题"
        assert meta["backend"] == "docling"

    def test_timeout_raises(self, monkeypatch, tmp_path):
        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired("cmd", 10)
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", True)
        monkeypatch.setattr(subprocess_runner.subprocess, "run", _timeout)
        with pytest.raises(TimeoutError, match="超时"):
            subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)

    def test_nonzero_exit_raises(self, monkeypatch, tmp_path):
        class _Completed:
            returncode = 1
            stdout = ""
            stderr = "boom"
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", True)
        monkeypatch.setattr(subprocess_runner.subprocess, "run", lambda *a, **k: _Completed())
        with pytest.raises(RuntimeError, match="boom"):
            subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)

    def test_probe_subprocess(self, monkeypatch):
        class _Completed:
            returncode = 0
            stdout = '{"version": "2.0"}\n'
            stderr = ""
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", True)
        monkeypatch.setattr(subprocess_runner.subprocess, "run", lambda *a, **k: _Completed())
        assert subprocess_runner.probe("docling", 10) == "2.0"

    def test_worker_uses_project_root_not_cwd(self, monkeypatch, tmp_path):
        # H3: 子进程 worker 用显式传入的项目根注入 sys.path, 不依赖 CWD(硬约束 #7)
        from indexing.parse_backends.subprocess_runner import _WORKER, _PROJECT_ROOT
        captured = {}
        class _Completed:
            returncode = 0
            stdout = json.dumps({"markdown": "# 标题", "format_meta": {"backend": "docling"}}) + "\n"
            stderr = ""
        def _fake_run(cmd, *a, **k):
            captured["cmd"] = cmd
            return _Completed()
        monkeypatch.setattr(subprocess_runner, "SUBPROCESS_EXECUTION", True)
        monkeypatch.setattr(subprocess_runner.subprocess, "run", _fake_run)
        subprocess_runner.run_extract("docling", str(tmp_path / "a.pdf"), 10)
        cmd = captured["cmd"]
        # 命令列表: [0]=python, [1]="-c", [2]=_WORKER, [3]=mode, [4]=backend_name,
        #           [5]=doc_path, [6]=project_root(worker 内读 sys.argv[4])
        assert cmd[6] == str(_PROJECT_ROOT)  # 项目根作为第 6 个 argv 显式传入
        assert "os.getcwd()" not in _WORKER  # worker 不再依赖 CWD
