"""regression 层 conftest：autouse 装 install_all_fakes（回归测试多走真实检索链路，隔离重依赖）。"""
import pytest

from tests._fakes import install_all_fakes


@pytest.fixture(autouse=True)
def auto_install_fakes(monkeypatch):
    """每个 regression 测试前置安装 Fake 层（含 STORAGE_BACKEND → memory）。"""
    install_all_fakes(monkeypatch)
