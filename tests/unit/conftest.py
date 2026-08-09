"""unit 层 conftest：autouse 装 install_all_fakes（隔离重依赖，monkeypatch teardown 自动 restore）。"""
import pytest

from tests._fakes import install_all_fakes


@pytest.fixture(autouse=True)
def auto_install_fakes(monkeypatch):
    """每个 unit 测试前置安装 Fake 层：embedding/reranker/cache/generator 全替，绝不触真实模型。"""
    install_all_fakes(monkeypatch)
