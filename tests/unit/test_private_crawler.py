"""unit: scripts/private_crawler 011-2 爬虫——slug/URL 读取/manifest/单条爬取(mock 网络)。"""
import json

import pytest

from scripts import private_crawler as pc


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测爬虫逻辑, 无需重依赖。"""
    yield


class TestSlugify:
    def test_strips_query_and_fragment(self):
        assert pc.slugify("https://juejin.cn/post/123?from=search#top", 0) == "0000-123"

    def test_illegal_chars_replaced(self):
        assert pc.slugify("https://x.com/a/b: c", 1) == "0001-b-c"

    def test_empty_stem_falls_back(self):
        assert pc.slugify("https://x.com/", 2) == "0002-x-com"


class TestReadUrls:
    def test_skips_blank_and_comments(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("# comment\nhttps://a.com\n\nhttps://b.com\n", encoding="utf-8")
        assert pc.read_urls(f) == ["https://a.com", "https://b.com"]


class TestSourceOf:
    def test_juejin(self):
        assert pc._source_of("https://juejin.cn/post/1") == "juejin"

    def test_csdn(self):
        assert pc._source_of("https://blog.csdn.net/x") == "csdn"

    def test_zhihu(self):
        assert pc._source_of("https://zhuanlan.zhihu.com/p/1") == "zhihu"

    def test_other(self):
        assert pc._source_of("https://example.com/x") == "other"


class TestWriteManifest:
    def test_writes_json(self, tmp_path):
        pc.write_manifest(tmp_path, [{"url": "u", "path": "p", "source": "juejin", "title": "t", "chars": 3}])
        data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert data["articles"][0]["source"] == "juejin"


class TestCrawlOne:
    def test_success_saves_and_cleans(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "fetch_html", lambda url, timeout_s=20: "<html><body>正文\n关注公众号：x</body></html>")
        monkeypatch.setattr(pc, "extract_main_text", lambda html: "正文\n关注公众号：x")
        entry = pc.crawl_one("https://juejin.cn/post/9", tmp_path, 0, delay_s=0)
        assert entry["source"] == "juejin"
        assert entry["title"] == "0000-9"  # L4: title 与文件名语义一致(序号+slug)
        saved = (tmp_path / "0000.md").read_text(encoding="utf-8")
        assert "正文" in saved
        assert "关注公众号" not in saved  # cleaner 生效

    def test_empty_text_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "fetch_html", lambda url, timeout_s=20: "<html></html>")
        monkeypatch.setattr(pc, "extract_main_text", lambda html: "")
        with pytest.raises(RuntimeError, match="正文为空"):
            pc.crawl_one("https://x.com/a", tmp_path, 0, delay_s=0)


class TestImportHtml:
    def test_imports_local_html(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "extract_main_text", lambda html: "正文\n关注公众号：x")
        src = tmp_path / "page.html"
        src.write_text("<html><body>正文\n关注公众号：x</body></html>", encoding="utf-8")
        entry = pc.import_html_file(src, tmp_path / "out", 0)
        assert entry["source"] != ""
        assert entry["title"] == "page"  # L4: 本地导入用源文件名作 title
        saved = (tmp_path / "out" / "0000.md").read_text(encoding="utf-8")
        assert "正文" in saved
        assert "关注公众号" not in saved

    def test_empty_html_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "extract_main_text", lambda html: "")
        src = tmp_path / "page.html"
        src.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(RuntimeError, match="正文为空"):
            pc.import_html_file(src, tmp_path / "out", 0)


class TestMain:
    def test_success_and_failure_paths(self, tmp_path, monkeypatch):
        urls = tmp_path / "urls.txt"
        urls.write_text("https://a.com/1\nhttps://b.com/2\n", encoding="utf-8")
        calls = {"n": 0}

        def _fake_crawl(url, out, index, delay):
            calls["n"] += 1
            if url.endswith("2"):
                raise RuntimeError("boom")
            return {"url": url, "path": "p", "source": "other", "title": "t", "chars": 3}

        monkeypatch.setattr(pc, "crawl_one", _fake_crawl)
        rc = pc.main(["--urls", str(urls), "--out", str(tmp_path), "--delay", "0"])
        assert rc == 1  # 有失败
        assert calls["n"] == 2
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["articles"]) == 1

    def test_empty_corpus_returns_1(self, tmp_path, monkeypatch):
        urls = tmp_path / "urls.txt"
        urls.write_text("", encoding="utf-8")
        assert pc.main(["--urls", str(urls), "--out", str(tmp_path), "--delay", "0"]) == 0

    def test_import_dir_mode(self, tmp_path, monkeypatch):
        html_dir = tmp_path / "html"
        html_dir.mkdir()
        (html_dir / "a.html").write_text("<html><body>正文A</body></html>", encoding="utf-8")
        (html_dir / "b.html").write_text("<html><body>正文B</body></html>", encoding="utf-8")
        monkeypatch.setattr(pc, "extract_main_text", lambda html: f"正文{html[-2]}")
        rc = pc.main(["--import-dir", str(html_dir), "--out", str(tmp_path / "out"), "--delay", "0"])
        assert rc == 0
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["articles"]) == 2  # 两篇都导入

    def test_neither_arg_returns_2(self, tmp_path):
        assert pc.main(["--out", str(tmp_path)]) == 2
