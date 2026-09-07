"""unit: indexing/parse_backends VLM 薄封装(012-2 F39)——base64/可用性/调用/错误路径。"""
import pytest

from indexing.parse_backends import vlm


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测 VLM 封装逻辑, 无需重依赖。"""
    yield


class TestImageToDataUrl:
    def test_encodes_png(self, tmp_path):
        png = tmp_path / "a.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")
        url = vlm._image_to_data_url(str(png))
        assert url.startswith("data:image/png;base64,")

    def test_encodes_jpg_mime(self, tmp_path):
        jpg = tmp_path / "a.jpg"
        jpg.write_bytes(b"jpegdata")
        url = vlm._image_to_data_url(str(jpg))
        assert url.startswith("data:image/jpeg;base64,")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(vlm.VlmUnavailableError, match="不存在"):
            vlm._image_to_data_url(str(tmp_path / "nope.png"))


class TestVersion:
    def test_returns_model_when_available(self, monkeypatch):
        monkeypatch.setattr(vlm, "is_available", lambda: True)
        monkeypatch.setattr(vlm, "VLM_MODEL", "deepseek-v4-flash-vision-exp")
        assert vlm.version() == "deepseek-v4-flash-vision-exp"

    def test_not_installed_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(vlm, "is_available", lambda: False)
        assert vlm.version() == "not-installed"


class TestRegister:
    def test_registers_when_available(self, monkeypatch):
        from indexing.parse_backends import _BACKEND_REGISTRY
        monkeypatch.setattr(vlm, "is_available", lambda: True)
        vlm.register()
        assert _BACKEND_REGISTRY.get("vlm") is not None


class TestIsAvailable:
    def test_false_when_disabled(self, monkeypatch):
        monkeypatch.setattr(vlm, "VLM_ENABLED", False)
        monkeypatch.setattr(vlm, "VLM_API_KEY", "sk-x")
        assert vlm.is_available() is False

    def test_false_when_no_key(self, monkeypatch):
        monkeypatch.setattr(vlm, "VLM_ENABLED", True)
        monkeypatch.setattr(vlm, "VLM_API_KEY", "")
        assert vlm.is_available() is False


class TestExtract:
    def test_returns_parse_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vlm, "is_available", lambda: True)
        monkeypatch.setattr(vlm, "_image_to_data_url", lambda p: "data:image/png;base64,x")
        monkeypatch.setattr(vlm, "_call_vlm", lambda url: "# 标题\n正文")
        png = tmp_path / "a.png"
        png.write_bytes(b"x")
        result = vlm.VlmBackend().extract(str(png))
        assert result.markdown == "# 标题\n正文"
        assert result.format_meta["backend"] == "vlm"
        assert result.format_meta["vlm_provider"] == "deepseek_official"

    def test_raises_when_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vlm, "is_available", lambda: False)
        with pytest.raises(vlm.VlmUnavailableError, match="不可用"):
            vlm.VlmBackend().extract(str(tmp_path / "a.png"))


class TestCallVlm:
    def test_success_returns_content(self, monkeypatch):
        class _Choice:
            message = type("M", (), {"content": "# 标题\n正文"})()
        class _Resp:
            choices = [_Choice()]
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        return _Resp()
        monkeypatch.setattr(vlm, "VLM_API_KEY", "sk-x")
        monkeypatch.setattr(vlm, "VLM_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setattr(vlm, "VLM_MODEL", "deepseek-v4-flash-vision-exp")
        monkeypatch.setattr(vlm, "VLM_TIMEOUT_S", 60)
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        assert vlm._call_vlm("data:image/png;base64,x") == "# 标题\n正文"

    def test_api_failure_raises(self, monkeypatch):
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        raise RuntimeError("boom")
        monkeypatch.setattr(vlm, "VLM_API_KEY", "sk-x")
        monkeypatch.setattr(vlm, "VLM_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setattr(vlm, "VLM_MODEL", "deepseek-v4-flash-vision-exp")
        monkeypatch.setattr(vlm, "VLM_TIMEOUT_S", 60)
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        with pytest.raises(vlm.VlmUnavailableError, match="API 调用失败"):
            vlm._call_vlm("data:image/png;base64,x")
