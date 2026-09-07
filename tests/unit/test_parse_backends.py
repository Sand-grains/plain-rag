"""unit: indexing/parse_backends 重型解析后端 adapter(012-1)——归一化/门控/兜底链/导入守卫。"""
import importlib.util

import pytest

from indexing.parse_backends import (
    ParseResult,
    heavy_chain_extract,
    parse_text_pipeline,
    resolve,
    _ensure_registered,
    _BACKEND_REGISTRY,
    _BACKEND_GATES,
)
from indexing.parse_backends._normalize import normalize_to_contract


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测 adapter 逻辑, 无需重依赖。"""
    yield


class TestNormalizeToContract:
    def test_heading_missing_space(self):
        assert normalize_to_contract("#foo\n##bar") == "# foo\n## bar"

    def test_line_endings_and_blank_lines(self):
        text = "# 标题\n\r\n\n\n正文\r\n\r\n\n\n"
        assert normalize_to_contract(text) == "# 标题\n\n正文"

    def test_table_separator_cells_normalized(self):
        text = "| a | b |\n| - | --- |\n| 1 | 2 |"
        assert normalize_to_contract(text) == "| a | b |\n|---|---|\n| 1 | 2 |"

    def test_empty(self):
        assert normalize_to_contract("") == ""


class TestDoclingBackend:
    def test_registered_when_available(self):
        _ensure_registered()
        assert _BACKEND_REGISTRY.get("docling") is not None or not importlib.util.find_spec("docling")

    def test_extract_raises_when_not_available(self, monkeypatch):
        from indexing.parse_backends import docling
        monkeypatch.setattr(docling, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="未安装"):
            docling.DoclingBackend().extract("x.pdf")


class TestMarkItDownBackend:
    def test_extract_raises_when_not_available(self, monkeypatch):
        from indexing.parse_backends import markitdown
        monkeypatch.setattr(markitdown, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="未安装"):
            markitdown.MarkItDownBackend().extract("x.html")


class TestMinerUBackend:
    def test_discover_markdown_prefers_stem_match(self, tmp_path):
        from indexing.parse_backends import mineru
        (tmp_path / "doc.md").write_text("同名校", encoding="utf-8")
        assert mineru._discover_markdown(str(tmp_path), str(tmp_path / "doc.pdf")) == "同名校"

    def test_discover_markdown_raises_when_none(self, tmp_path):
        from indexing.parse_backends import mineru
        with pytest.raises(mineru.MinerUError, match="未产出"):
            mineru._discover_markdown(str(tmp_path), "doc.pdf")

    def test_run_cli_nonzero_raises(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        class _Completed:
            returncode = 1
            stderr = "boom"
            stdout = ""
        monkeypatch.setattr(mineru, "MINERU_ROOT", None)
        monkeypatch.setattr(mineru.subprocess, "run", lambda *a, **k: _Completed())
        with pytest.raises(mineru.MinerUError, match="boom"):
            mineru._run_cli(str(tmp_path / "x.pdf"), str(tmp_path), 10)

    def test_run_cli_timeout_raises(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        def _timeout(*a, **k):
            raise mineru.subprocess.TimeoutExpired("cmd", 10)
        monkeypatch.setattr(mineru, "MINERU_ROOT", None)
        monkeypatch.setattr(mineru.subprocess, "run", _timeout)
        with pytest.raises(mineru.MinerUError, match="超时"):
            mineru._run_cli(str(tmp_path / "x.pdf"), str(tmp_path), 10)

    def test_extract_raises_when_not_available(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "is_available", lambda: False)
        with pytest.raises(mineru.MinerUError, match="未安装"):
            mineru.MinerUBackend().extract("x.pdf")

    def test_python_executable_uses_mineru_python_when_set(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "MINERU_PYTHON", r"D:\MinerU\python.exe")
        assert mineru._python_executable() == r"D:\MinerU\python.exe"

    def test_python_executable_falls_back_to_sys_executable(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "MINERU_PYTHON", None)
        assert mineru._python_executable() == mineru.sys.executable

    def test_is_available_true_when_mineru_python_set(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "MINERU_PYTHON", r"D:\MinerU\python.exe")
        monkeypatch.setattr(mineru.importlib.util, "find_spec", lambda name: None)
        assert mineru.is_available() is True

    def test_run_cli_uses_mineru_python(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        captured = {}
        class _Completed:
            returncode = 0
            stderr = ""
            stdout = ""
        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _Completed()
        monkeypatch.setattr(mineru, "MINERU_PYTHON", r"D:\MinerU\python.exe")
        monkeypatch.setattr(mineru, "MINERU_ROOT", None)
        monkeypatch.setattr(mineru.subprocess, "run", _fake_run)
        mineru._run_cli(str(tmp_path / "x.pdf"), str(tmp_path), 10)
        assert captured["cmd"][0] == r"D:\MinerU\python.exe"

    def test_version_queries_standalone_python(self, monkeypatch):
        from indexing.parse_backends import mineru
        class _Completed:
            returncode = 0
            stdout = "3.4.0\n"
            stderr = ""
        monkeypatch.setattr(mineru, "MINERU_PYTHON", r"D:\MinerU\python.exe")
        monkeypatch.setattr(mineru.subprocess, "run", lambda *a, **k: _Completed())
        assert mineru.version() == "3.4.0"

    def test_version_standalone_unknown_on_failure(self, monkeypatch):
        from indexing.parse_backends import mineru
        class _Completed:
            returncode = 1
            stdout = ""
            stderr = "boom"
        monkeypatch.setattr(mineru, "MINERU_PYTHON", r"D:\MinerU\python.exe")
        monkeypatch.setattr(mineru.subprocess, "run", lambda *a, **k: _Completed())
        assert mineru.version() == "standalone-unknown"

    def test_extract_success_path(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "is_available", lambda: True)
        monkeypatch.setattr(mineru, "_run_cli", lambda *a, **k: None)
        monkeypatch.setattr(mineru, "_discover_markdown", lambda *a, **k: "# 标题\n正文")
        monkeypatch.setattr(mineru, "version", lambda: "3.4.0")
        result = mineru.MinerUBackend().extract(str(tmp_path / "x.pdf"))
        assert result.markdown == "# 标题\n正文"
        assert result.format_meta["backend"] == "mineru"
        assert result.format_meta["version"] == "3.4.0"

    def test_mineru_env_built_from_root(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "MINERU_ROOT", r"D:\MinerU\MinerU\MinerU")
        env = mineru._mineru_env()
        assert env is not None
        assert env["PYTHONPATH"] == r"D:\MinerU\MinerU\MinerU\src"
        assert env["MINERU_MODEL_SOURCE"] == "modelscope"
        assert env["HF_HUB_OFFLINE"] == "1"

    def test_mineru_env_none_without_root(self, monkeypatch):
        from indexing.parse_backends import mineru
        monkeypatch.setattr(mineru, "MINERU_ROOT", None)
        assert mineru._mineru_env() is None

    def test_ensure_config_runs_generate_when_missing(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        root = tmp_path / "mineru"
        (root / "src").mkdir(parents=True)
        (root / "config").mkdir()
        (root / "src" / "generate_config.py").write_text("", encoding="utf-8")
        monkeypatch.setattr(mineru, "MINERU_ROOT", str(root))
        monkeypatch.setattr(mineru, "_python_executable", lambda: "py")
        captured = {}
        monkeypatch.setattr(mineru.subprocess, "run",
                            lambda cmd, **k: captured.update(cmd=cmd) or type("C", (), {"returncode": 0})())
        mineru._ensure_config()
        assert captured["cmd"][1].endswith("generate_config.py")

    def test_ensure_config_skips_when_exists(self, tmp_path, monkeypatch):
        from indexing.parse_backends import mineru
        root = tmp_path / "mineru"
        (root / "config").mkdir(parents=True)
        (root / "config" / "mineru.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(mineru, "MINERU_ROOT", str(root))
        called = []
        monkeypatch.setattr(mineru.subprocess, "run", lambda *a, **k: called.append(1))
        mineru._ensure_config()
        assert called == []


class TestRunBackend:
    def test_success_records_stats(self, monkeypatch, tmp_path):
        from indexing.parse_backends import _run_backend, reset_runtime, run_stats
        reset_runtime()
        class _Fake:
            def extract(self, path):
                return ParseResult(markdown="ok", format_meta={"backend": "docling"})
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        markdown, meta = _run_backend("docling", str(tmp_path / "a.pdf"))
        assert markdown == "ok"
        assert run_stats.summary()["success"] == 1

    def test_open_breaker_raises(self, monkeypatch, tmp_path):
        from indexing.parse_backends import _run_backend, circuit_breaker, reset_runtime
        reset_runtime()
        circuit_breaker._open.add("docling")
        with pytest.raises(RuntimeError, match="已熔断"):
            _run_backend("docling", str(tmp_path / "a.pdf"))

    def test_failure_records_breaker(self, monkeypatch, tmp_path):
        from indexing.parse_backends import _run_backend, circuit_breaker, reset_runtime
        reset_runtime()
        class _Fake:
            def extract(self, path):
                raise RuntimeError("boom")
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        with pytest.raises(RuntimeError, match="boom"):
            _run_backend("docling", str(tmp_path / "a.pdf"))
        assert circuit_breaker._failures["docling"] == 1


class TestHeavyChainExtract:
    def test_all_gates_off_raises_and_no_lightweight(self, monkeypatch):
        monkeypatch.setitem(_BACKEND_GATES, "docling", False)
        monkeypatch.setitem(_BACKEND_GATES, "mineru", False)
        monkeypatch.setitem(_BACKEND_GATES, "markitdown", False)
        with pytest.raises(RuntimeError, match="不允许落轻量"):
            heavy_chain_extract("x.pdf")

    def test_falls_through_empty_output_docling_to_mineru(self, monkeypatch):
        class _Fake:
            def __init__(self, text): self.text = text
            def extract(self, path): return ParseResult(markdown=self.text, format_meta={"backend": self.name})
        docling_backend = type("B", (_Fake,), {"name": "docling"})("")
        mineru_backend = type("B", (_Fake,), {"name": "mineru"})("正文ok")
        def _fake_resolve(name):
            return {"docling": docling_backend, "mineru": mineru_backend}.get(name)
        monkeypatch.setattr("indexing.parse_backends.resolve", _fake_resolve)
        markdown, meta = heavy_chain_extract("x.pdf")
        assert markdown == "正文ok"
        assert meta["backend"] == "mineru"


class TestParseTextPipeline:
    def test_quality_driven_fallback_to_mineru(self, monkeypatch):
        # docling 输出表格丢失(质量不达标) -> 落到 mineru
        class _Fake:
            def __init__(self, text, name): self.text, self.name = text, name
            def extract(self, path): return ParseResult(markdown=self.text, format_meta={"backend": self.name})
        docling_backend = _Fake("正文无表格", "docling")
        mineru_backend = _Fake("| a |\n|---|\n| 1 |", "mineru")
        def _fake_resolve(name):
            return {"docling": docling_backend, "mineru": mineru_backend}.get(name)
        monkeypatch.setattr("indexing.parse_backends.resolve", _fake_resolve)
        markdown, meta = parse_text_pipeline("x.pdf", original_table_count=1)
        assert markdown == "| a |\n|---|\n| 1 |"
        assert meta["backend"] == "mineru"
        assert "table_loss" in meta["retry_stats"]["quality_reasons"]["docling"]

    def test_quality_ok_stops_at_first(self, monkeypatch):
        class _Fake:
            def __init__(self, text, name): self.text, self.name = text, name
            def extract(self, path): return ParseResult(markdown=self.text, format_meta={"backend": self.name})
        docling_backend = _Fake("# 标题\n正文", "docling")
        mineru_backend = _Fake("其他", "mineru")
        def _fake_resolve(name):
            return {"docling": docling_backend, "mineru": mineru_backend}.get(name)
        monkeypatch.setattr("indexing.parse_backends.resolve", _fake_resolve)
        markdown, meta = parse_text_pipeline("x.pdf")
        assert meta["backend"] == "docling"
        assert meta["retry_stats"]["attempted"] == ["docling"]

    def test_all_fail_quality_raises(self, monkeypatch):
        class _Fake:
            def __init__(self, text, name): self.text, self.name = text, name
            def extract(self, path): return ParseResult(markdown=self.text, format_meta={"backend": self.name})
        docling_backend = _Fake("", "docling")
        mineru_backend = _Fake("", "mineru")
        def _fake_resolve(name):
            return {"docling": docling_backend, "mineru": mineru_backend}.get(name)
        monkeypatch.setattr("indexing.parse_backends.resolve", _fake_resolve)
        with pytest.raises(RuntimeError, match="不允许落轻量"):
            parse_text_pipeline("x.pdf")


class TestReviewFixes:
    """012-2 审理报告修复回归: M4/M5/M6/M7。"""

    def test_chain_error_carries_real_backend_name(self, monkeypatch):
        # M4: 链式失败异常携带真实失败后端名(最后尝试的后端)
        class _Fake:
            def extract(self, path): raise RuntimeError("boom")
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        with pytest.raises(RuntimeError) as excinfo:
            heavy_chain_extract("x.pdf")
        assert getattr(excinfo.value, "backend_name", None) == "markitdown"

    def test_chain_skips_open_breaker_backend(self, monkeypatch):
        # M5: 熔断后端在链中被跳过, 不进入 _run_backend
        from indexing.parse_backends import circuit_breaker, reset_runtime
        reset_runtime()
        circuit_breaker._open.add("docling")
        calls = []
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: type("B", (), {})())
        monkeypatch.setattr("indexing.parse_backends._run_backend",
                            lambda name, path: calls.append(name) or ("ok", {"backend": name}))
        markdown, meta = heavy_chain_extract("x.pdf")
        assert "docling" not in calls
        assert "mineru" in calls
        assert meta["backend"] == "mineru"

    def test_colpali_not_in_gates_or_enabled(self):
        # M7: ColPali 是独立管线, 不注册进 ParseBackend 门控/启用列表
        from indexing.parse_backends import enabled_backends
        assert "colpali" not in _BACKEND_GATES
        assert "colpali" not in enabled_backends()

    def test_vlm_correction_uses_run_backend(self, monkeypatch):
        # M6: 页级 VLM 修正经 _run_backend(熔断+统计), 而非直接调 backend.extract
        calls = []
        def _fake_run_backend(name, path):
            calls.append((name, path))
            if name == "vlm":
                return ("# VLM 修正", {"backend": "vlm"})
            raise RuntimeError("backend fail")
        monkeypatch.setattr("indexing.parse_backends._run_backend", _fake_run_backend)
        def _fake_paged(doc_path, original_table_count, expected_code_blocks, chain=None,
                        quality_fn=None, vlm_extract=None, page_splitter=None,
                        page_renderer=None, page_chain=None):
            return vlm_extract("img.png")
        monkeypatch.setattr("indexing.parse_backends.paged_pipeline.parse_text_pipeline_paged",
                            _fake_paged)
        class _Fake:
            def extract(self, path): return ParseResult(markdown="", format_meta={"backend": "docling"})
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        markdown, meta = parse_text_pipeline("x.pdf")
        assert markdown == "# VLM 修正"
        assert ("vlm", "img.png") in calls
