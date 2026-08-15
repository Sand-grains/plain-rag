"""unit：报告器(F06), PROGRESS.md 快照 golden 对比 + 人工保护区保留。"""
from harness.core.models import FeatureItem, FeatureList
from harness.service.reporter import Reporter
from harness.core.states import ACTIVE, BLOCKED, NOT_STARTED, PASSED
from harness.service.tracker import judge_health

FIXED_TIME = "2026-08-13 12:00:00"


def _feature_list() -> FeatureList:
    return FeatureList(schema=1, milestone="007-harness-governance", items=[
        FeatureItem(id="F01", behavior="harness-状态机", gate="pytest test_state_machine.py -q",
                    state=PASSED, note=None,
                    last_verify={"exit_code": 0, "tests": 8, "coverage": 100.0, "duration": 1.2}),
        FeatureItem(id="F02", behavior="harness-清单解析", gate="pytest test_registry.py -q",
                    state=ACTIVE, note="卡点: 等接口冻结"),
        FeatureItem(id="F03", behavior="harness-验证器", gate="pytest test_verifier.py -q",
                    state=BLOCKED, note="待确认超时阈值"),
    ])


class TestRender:
    def test_render_golden_structure(self, tmp_path):
        feature_list = _feature_list()
        reporter = Reporter(tmp_path / "features.yaml", tmp_path / "PROGRESS.md")
        rendered = reporter.render(feature_list, judge_health(feature_list), FIXED_TIME)

        assert "# 项目进度" in rendered
        assert f"生成时间: {FIXED_TIME}" in rendered
        assert "## 当前点位" in rendered and "007-harness-governance" in rendered
        assert "## 状态分布" in rendered and "- passed: 1" in rendered
        assert "## 健康度" in rendered
        assert "## 功能项状态表" in rendered
        assert "| F02 | harness-清单解析 | active | - | 卡点: 等接口冻结 |" in rendered
        assert "## 验证结果" in rendered
        assert "F01: exit_code=0 tests=8 coverage=100.0 duration=1.2s" in rendered
        assert "<!-- manual -->" in rendered

    def test_narrative_only_from_note(self, tmp_path):
        feature_list = _feature_list()
        reporter = Reporter(tmp_path / "features.yaml", tmp_path / "PROGRESS.md")
        rendered = reporter.render(feature_list, judge_health(feature_list), FIXED_TIME)
        # 叙述区只出现 items 里真实存在的 note, 报告器不自己造句
        assert "等接口冻结" in rendered and "待确认超时阈值" in rendered
        assert "瓶颈" not in rendered


class TestWriteManualZone:
    def test_manual_zone_preserved_on_rewrite(self, tmp_path):
        progress_path = tmp_path / "PROGRESS.md"
        features_path = tmp_path / "features.yaml"
        progress_path.write_text(
            "# 旧快照\n<!-- manual -->\n\n人工保护区: 执行前必读 007 文档\n",
            encoding="utf-8",
        )
        reporter = Reporter(features_path, progress_path)
        feature_list = _feature_list()
        reporter.write(feature_list, judge_health(feature_list), FIXED_TIME)

        content = progress_path.read_text(encoding="utf-8")
        assert "人工保护区: 执行前必读 007 文档" in content
        assert "# 旧快照" not in content  # 自动区整段重建
        assert "## 健康度" in content

    def test_rewrite_without_manual_marker(self, tmp_path):
        progress_path = tmp_path / "PROGRESS.md"
        features_path = tmp_path / "features.yaml"
        progress_path.write_text("# 旧内容, 无人工区\n", encoding="utf-8")
        reporter = Reporter(features_path, progress_path)
        feature_list = _feature_list()
        reporter.write(feature_list, judge_health(feature_list), FIXED_TIME)

        content = progress_path.read_text(encoding="utf-8")
        assert "# 旧内容" not in content
        assert "<!-- manual -->" in content
