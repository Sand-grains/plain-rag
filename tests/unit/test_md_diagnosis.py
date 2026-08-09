"""unit：文档质量诊断三路由键（preprocess/md_diagnosis.py diagnose）。"""
from preprocess.md_diagnosis import diagnose


class TestDiagnose:
    def test_structured_doc_has_h1_and_standard(self):
        report = diagnose("# Title\n\n## A\n\nbody a\n\n## B\n\nbody b")
        assert report.has_h1 is True
        assert report.heading_connection_standard is True

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
