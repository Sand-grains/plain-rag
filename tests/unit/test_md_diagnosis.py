"""unit：文档质量诊断三路由键（preprocess/md_diagnosis.py diagnose）。"""
from preprocess.md_diagnosis import diagnose


class TestDiagnose:
    def test_structured_doc_has_h1_and_standard(self):
        report = diagnose("# Title\n\n## A\n\nbody a\n\n## B\n\nbody b")
        assert report.has_h1 is True
        assert report.heading_continuous is True
        assert report.heading_density_ok is True

    def test_heading_continuity_fails_on_skip_level(self):
        # 出现 ## 跳 ####(缺 ###) → heading_continuous False, 密度独立仍可 True
        text = "# Title\n\n## A\n\n#### 深钻缺三级\n\n" + "内容。" * 60
        report = diagnose(text)
        assert report.heading_continuous is False
        assert report.heading_density_ok is True

    def test_heading_density_fails_when_sparse(self):
        # 标题过稀(平均字符间隔 > 2000) → heading_density_ok False, 连续性独立仍可 True
        text = "# Title\n\n" + ("正文段落。" * 900) + "\n\n## 结尾"
        report = diagnose(text)
        assert report.heading_continuous is True
        assert report.heading_density_ok is False

    def test_no_h1_fragmented(self):
        report = diagnose("短文无标题。")
        assert report.has_h1 is False
        assert report.too_fragmented is True

    def test_no_h1_not_fragmented(self):
        report = diagnose("长文无标题。" + "内容段落。" * 200)
        assert report.has_h1 is False
        assert report.too_fragmented is False

    def test_encoding_issue_flag(self):
        report = diagnose("正文 � 异味")
        assert report.has_encoding_issues is True
