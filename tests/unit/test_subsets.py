"""unit: eval/core/subsets 评测多子集基建(011-0)——公开集独立 store 装配/交集断言/缺失即 skip/多子集加载。"""
import json
from pathlib import Path

import pytest

import eval.core.subsets as subsets


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测加载/装配逻辑, 无需重依赖(避免 ~18s 导入成本)。"""
    yield


class _FakeStore:
    def __init__(self, chunk_ids=()):
        self._ids = set(chunk_ids)

    @property
    def chunk_ids(self) -> set[str]:
        return self._ids


def _write_benchmark(tmp_path, name: str, items) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return str(path)


class TestBuildPublicStore:
    def test_returns_restored_store(self, monkeypatch):
        fake = _FakeStore({"pub-1"})
        monkeypatch.setattr(subsets.IndexStore, "_restore_memory", classmethod(lambda cls, cache_dir: fake))
        assert subsets.build_public_store("cache") is fake

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(subsets.IndexStore, "_restore_memory", classmethod(lambda cls, cache_dir: None))
        with pytest.raises(RuntimeError):
            subsets.build_public_store("cache")


class TestAssertPublicStoreIsolated:
    def test_raises_on_intersection(self):
        public = _FakeStore({"pub-1", "shared"})
        local = _FakeStore({"local-1", "shared"})
        with pytest.raises(RuntimeError):
            subsets.assert_public_store_isolated(public, local)

    def test_passes_when_disjoint(self):
        public = _FakeStore({"pub-1"})
        local = _FakeStore({"local-1"})
        subsets.assert_public_store_isolated(public, local)  # 不抛即通过


class TestLoadSubsets:
    def test_private_subset_loads_with_local_store(self, tmp_path, monkeypatch):
        local = _FakeStore({"c1"})
        bench = _write_benchmark(tmp_path, "private.json",
                                 [{"query_id": "Q1", "query": "q", "expected_parent_ids": ["c1"]}])
        loaded = subsets.load_subsets([bench], local, public_paths=set(),
                                      public_cache_dir="cache", skip_if_missing=set())
        assert len(loaded) == 1
        subset = loaded[0]
        assert subset.kind == "private"
        assert not subset.skipped
        assert subset.retriever.index_store is local
        assert len(subset.items) == 1

    def test_public_subset_uses_independent_store_and_no_validation(self, tmp_path, monkeypatch):
        local = _FakeStore({"local-1"})
        public = _FakeStore({"pub-1"})
        monkeypatch.setattr(subsets.IndexStore, "_restore_memory", classmethod(lambda cls, cache_dir: public))
        # 公开子集 expected_parent_ids 指向公开索引, 不校验本地 store(valid_chunk_ids=None)
        bench = _write_benchmark(tmp_path, "public.json",
                                 [{"query_id": "Q1", "query": "q", "expected_parent_ids": ["pub-1"]}])
        loaded = subsets.load_subsets([bench], local, public_paths={bench},
                                      public_cache_dir="cache", skip_if_missing=set())
        assert len(loaded) == 1
        subset = loaded[0]
        assert subset.kind == "public"
        assert subset.retriever.index_store is public
        assert subset.load_result.invalid_chunk_ids == {}

    def test_public_subset_raises_on_intersection(self, tmp_path, monkeypatch):
        local = _FakeStore({"shared"})
        public = _FakeStore({"shared"})
        monkeypatch.setattr(subsets.IndexStore, "_restore_memory", classmethod(lambda cls, cache_dir: public))
        bench = _write_benchmark(tmp_path, "public.json", [])
        with pytest.raises(RuntimeError):
            subsets.load_subsets([bench], local, public_paths={bench},
                                 public_cache_dir="cache", skip_if_missing=set())

    def test_missing_skip_file_is_skipped(self, tmp_path):
        local = _FakeStore({"c1"})
        missing = str(tmp_path / "private_crawler.json")
        loaded = subsets.load_subsets([missing], local, public_paths=set(),
                                      public_cache_dir="cache", skip_if_missing={missing})
        assert len(loaded) == 1
        assert loaded[0].skipped
        assert loaded[0].retriever is None

    def test_missing_non_skip_file_raises(self, tmp_path):
        local = _FakeStore({"c1"})
        missing = str(tmp_path / "private_v6.json")
        with pytest.raises(FileNotFoundError):
            subsets.load_subsets([missing], local, public_paths=set(),
                                 public_cache_dir="cache", skip_if_missing=set())
