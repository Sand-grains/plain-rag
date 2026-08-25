"""regression：.md/.txt 分块输出字节级不变（008 gate #0 基线）。

008 计划的核心承诺（条例四/五）：.md/.txt 走原路径，private_v6 benchmark
分块输出字节级不变。本测试作为基线，在 008-3 字段拆解与 008-2 splitter
表格保护（均只对新格式归一化 MD 开启）落地前后，喂真实 data/ 下的
.md/.txt 语料走生产分块路径（load → diagnose → Router → split），
断言产出子块的字节序列指纹恒定。

- 基线指纹：2026-08-24 生成，216 文件 / 2278 子块 / 0 跳过，覆盖纯文本、
  标题树、超长 section、代码块等形态。
- 任何会改变 .md/.txt 分块结果的核心改动都会使指纹漂移 → 测试失败，
  从而守住 private_v6 不变承诺。
- 本测试不触碰 embedding / 索引 / 检索，无 infra 依赖，可离线运行。
"""
import hashlib
from pathlib import Path

import pytest

from indexing.loader import load
from indexing.router import Router
from preprocess.md_diagnosis import diagnose

# 基线指纹（.md/.txt 全语料分块产出的 sha256）
GOLDEN_SHA256 = "d15c24f3915dbc5369ec74ecf39362dab5feadf573cb22354dff6a4c2ed47510"

_SUPPORTED_SUFFIXES = (".txt", ".md")


def _real_corpus(root: Path = Path("data")) -> list[Path]:
    """发现 data/ 下的真实 .md/.txt 语料（稳定排序，保证确定性）。"""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


# data/ 是 gitignored 输出, CI 全新 checkout 无此目录 -> 语料为空时跳过
# (本地有 data/ 时照常跑, 守住 private_v6 字节级不变承诺)
pytestmark = pytest.mark.skipif(
    not _real_corpus(),
    reason="data/ 语料缺失(gitignored, CI 无此目录), 跳过字节级不变回归",
)


def _fingerprint() -> str:
    """跑生产分块路径，返回子块内容字节序列的 sha256 指纹。"""
    router = Router()
    blocks = []
    for file_path in _real_corpus():
        for doc in load(str(file_path)):
            report = diagnose(doc.content)
            splitter = router.route(report)
            result = splitter.split(
                doc.content,
                {"doc_id": doc.doc_id, "doc_meta": doc.origin_metadata},
            )
            parents, children = result if isinstance(result, tuple) else (result, result)
            blocks.extend(child.content.encode("utf-8") for child in children)
    return hashlib.sha256(b"\x00".join(blocks)).hexdigest()


class TestSplitterByteInvariance:
    def test_corpus_not_empty(self):
        assert len(_real_corpus()) > 0, "data/ 下没有 .md/.txt 语料，无法建立基线"

    def test_md_txt_chunk_output_byte_invariant(self):
        """.md/.txt 分块输出与基线指纹一致（private_v6 字节级不变承诺）。"""
        assert _fingerprint() == GOLDEN_SHA256

    def test_split_is_deterministic(self):
        """同语料两次分块指纹一致（确定性）。"""
        assert _fingerprint() == _fingerprint()
