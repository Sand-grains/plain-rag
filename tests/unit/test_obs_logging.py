"""unit：obs/logging_setup 日志底座——落盘格式含 request_id / OBS_ENABLED 开关 / 幂等 / force 重配。"""
import logging
import logging.handlers
import os

import pytest

import config
from obs import logging_setup
from obs.trace import request_id_var


@pytest.fixture(autouse=True)
def _clean_logging_state(monkeypatch):
    """每个测试前后重置 _configured 标记/request_id_var, 并还原 root handler, 防跨测试污染。"""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    monkeypatch.setattr(logging_setup, "_configured", False)
    yield
    request_id_var.set("-")
    root.handlers[:] = original_handlers


def _file_handlers():
    return [handler for handler in logging.getLogger().handlers
            if isinstance(handler, logging.handlers.RotatingFileHandler)]


class TestSetupLogging:
    def test_writes_log_with_request_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_ENABLED", True)
        logging_setup.setup_logging(tmp_path)
        request_id_var.set("eval-Q001")
        logging.getLogger("test.module").info("hello world")
        log_file = tmp_path / "plain_rag.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "hello world" in content
        assert "[eval-Q001]" in content

    def test_default_request_id_is_dash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_ENABLED", True)
        logging_setup.setup_logging(tmp_path)
        logging.getLogger("test.module").info("no request id")
        content = (tmp_path / "plain_rag.log").read_text(encoding="utf-8")
        assert "[-]" in content

    def test_obs_disabled_skips_file_handler(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_ENABLED", False)
        logging_setup.setup_logging(tmp_path)
        assert _file_handlers() == []
        assert not (tmp_path / "plain_rag.log").exists()

    def test_idempotent_no_duplicate_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_ENABLED", True)
        logging_setup.setup_logging(tmp_path)
        logging_setup.setup_logging(tmp_path)
        assert len(_file_handlers()) == 1

    def test_force_reconfigures_target_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_ENABLED", True)
        logging_setup.setup_logging(tmp_path)
        other_dir = tmp_path / "other"
        logging_setup.setup_logging(other_dir, force=True)
        base_filenames = {handler.baseFilename for handler in _file_handlers()}
        assert os.path.join(str(other_dir), "plain_rag.log") in base_filenames


class TestStdioEncodingSafe:
    def test_make_stdio_encoding_safe_sets_replace(self, monkeypatch):
        import sys as sys_module

        class FakeStream:
            def __init__(self):
                self.errors = None

            def reconfigure(self, errors=None):
                self.errors = errors

        fake_out, fake_err = FakeStream(), FakeStream()
        monkeypatch.setattr(sys_module, "stdout", fake_out)
        monkeypatch.setattr(sys_module, "stderr", fake_err)
        logging_setup.make_stdio_encoding_safe()
        assert fake_out.errors == "replace"
        assert fake_err.errors == "replace"

    def test_yen_encodes_with_replace_strategy_in_gbk(self):
        # ¥(U+00A5) 在 GBK 缺码, strict 抛错; replace 策略替换为 ? 不抛
        encoded = "估算成本: ¥0.00".encode("cp936", errors="replace")
        assert b"?" in encoded

    def test_final_report_survives_gbk_stdout(self, monkeypatch):
        import io
        import sys as sys_module

        from obs.monitor_metrics import get_metrics, reset_metrics
        from obs.monitor_panel import MonitorPanel

        raw = io.BytesIO()
        gbk_stdout = io.TextIOWrapper(raw, encoding="cp936", errors="replace")
        monkeypatch.setattr(sys_module, "stdout", gbk_stdout)

        reset_metrics()
        metrics = get_metrics()

        class _Result:
            query_id = "Q1"
            recall_at_k = 1.0
            precision_at_k = 1.0
            hit_at_k = 1
            mrr = 1.0
            map_at_k = 1.0
            ndcg_at_k = 1.0
            diagnosis = "accept"

        metrics.layer1_results = [_Result()]
        panel = MonitorPanel(metrics, previous_per_query=None)
        panel.set_total(1)
        panel.set_meta(benchmark_name="bench.json", eval_mode="retrieval")
        panel.start_time = 0.0
        panel.final_report()
        gbk_stdout.flush()
        content = raw.getvalue().decode("cp936", errors="replace")
        assert "估算成本" in content
        assert "最终报告生成失败" not in content
