"""unit: indexing/parse_backends ColPali 薄封装(012-2 F40)——可用性/版本/触发/编码。"""
import sys
import types

import pytest

from indexing.parse_backends import colpali


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测 ColPali 封装逻辑, 无需重依赖。"""
    yield


class TestIsAvailable:
    def test_false_when_disabled(self, monkeypatch):
        monkeypatch.setattr(colpali, "COLPALI_ENABLED", False)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")
        assert colpali.is_available() is False

    def test_false_when_no_model_path(self, monkeypatch):
        monkeypatch.setattr(colpali, "COLPALI_ENABLED", True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "")
        assert colpali.is_available() is False

    def test_false_when_engine_not_importable(self, monkeypatch):
        monkeypatch.setattr(colpali, "COLPALI_ENABLED", True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert colpali.is_available() is False

    def test_true_when_enabled_path_and_engine(self, monkeypatch):
        monkeypatch.setattr(colpali, "COLPALI_ENABLED", True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")
        monkeypatch.setattr("importlib.util.find_spec", lambda name: types.SimpleNamespace())
        assert colpali.is_available() is True


class TestVersion:
    def test_returns_model_path_when_available(self, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")
        assert colpali.version() == "D:/models/colpali"

    def test_not_installed_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: False)
        assert colpali.version() == "not-installed"


class TestTriggered:
    def test_true_when_colpali_triggered(self):
        result = types.SimpleNamespace(colpali_triggered=True)
        assert colpali.triggered(result) is True

    def test_false_when_not_triggered(self):
        result = types.SimpleNamespace(colpali_triggered=False)
        assert colpali.triggered(result) is False

    def test_false_when_attr_missing(self):
        result = types.SimpleNamespace()
        assert colpali.triggered(result) is False


class TestEncodeImage:
    def test_raises_when_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: False)
        png = tmp_path / "a.png"
        png.write_bytes(b"x")
        with pytest.raises(colpali.ColPaliUnavailableError, match="不可用"):
            colpali.encode_image(str(png))

    def test_raises_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: True)
        with pytest.raises(colpali.ColPaliUnavailableError, match="不存在"):
            colpali.encode_image(str(tmp_path / "nope.png"))

    def test_returns_embeddings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")

        class _FakeEmb:
            def tolist(self):
                return [[0.1, 0.2], [0.3, 0.4]]

        class _FakeModel:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def eval(self):
                return self

            def __call__(self, **batch):
                return [_FakeEmb()]

        class _FakeProcessor:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

        fake_models = types.ModuleType("colpali_engine.models")
        fake_models.ColPali = _FakeModel
        fake_models.ColPaliProcessor = _FakeProcessor
        fake_utils = types.ModuleType("colpali_engine.utils.processing_utils")
        fake_utils.process_images = lambda images, processor: {"pixel_values": images}

        class _FakeTorch:
            bfloat16 = "bfloat16"

            class no_grad:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

        monkeypatch.setitem(sys.modules, "colpali_engine.models", fake_models)
        monkeypatch.setitem(sys.modules, "colpali_engine.utils.processing_utils", fake_utils)
        monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

        png = tmp_path / "a.png"
        png.write_bytes(b"x")
        assert colpali.encode_image(str(png)) == [[0.1, 0.2], [0.3, 0.4]]

    def test_encoding_failure_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(colpali, "is_available", lambda: True)
        monkeypatch.setattr(colpali, "COLPALI_MODEL_PATH", "D:/models/colpali")

        class _BoomModel:
            @classmethod
            def from_pretrained(cls, *a, **k):
                raise RuntimeError("model load boom")

        fake_models = types.ModuleType("colpali_engine.models")
        fake_models.ColPali = _BoomModel
        fake_models.ColPaliProcessor = object
        fake_utils = types.ModuleType("colpali_engine.utils.processing_utils")
        fake_utils.process_images = lambda images, processor: {}
        monkeypatch.setitem(sys.modules, "colpali_engine.models", fake_models)
        monkeypatch.setitem(sys.modules, "colpali_engine.utils.processing_utils", fake_utils)

        png = tmp_path / "a.png"
        png.write_bytes(b"x")
        with pytest.raises(colpali.ColPaliUnavailableError, match="编码失败"):
            colpali.encode_image(str(png))
