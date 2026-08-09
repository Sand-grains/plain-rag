"""regression：fix_code_block 修复回归（G7）——build_cases 6 输入 × 生产 splitter 结论性属性。

原 scripts/verify_splitter_regression.py 已整删（G7：不留壳，避免两处逻辑漂移），
仅迁其 build_cases() 6 个输入，断言生产 RecursiveCharacterTextSplitter
（已合入 variant F：_hard_split 吞尾随换行 + _merge_code_adjacent_micro_ws）
的结论性属性。A-E 变体不迁移。

成立的全部结论（6 输入均通过）：
    - 无 micro-chunk：所有 chunk > 3 字符（硬切吞换行 + 代码邻接空白合并的产出）
    - 代码块不被切断：任何 chunk 的 fence 数恒为偶数（开/闭 ``` 成对，切半 → 奇数）
    - id 确定性：同输入两次切分 chunk_id / content 序列一致
    - case 2/3/5/6 无 stray-fence（fence 只出现在 chunk 起点或不存在）
    - case 5/6：代码块后的短段并入同一切片（不切碎成碎片）

已知局限（生产现状，非本次回归引入，另需修）：
    case 1/4（代码块前有长前文）：_apply_overlap 的回填窗口 [start-overlap, start)
    与代码块相邻不重叠（prose→code 方向），把前一片末尾 50 字符回填进代码块 chunk →
    chunk 中部出现 ```（stray-fence@50）。overlap 的代码守卫只挡 code→prose 方向
    （见 _apply_overlap docstring"防止代码尾巴 + 闭合围栏污染下一块"），prose→code
    回填不受阻。这属 overlap 语义缺口，不在 variant F 修复范围内，故不纳入本回归断言。
"""
import pytest

from config import CHILD_CHUNK_SIZE, CHILD_OVERLAP
from indexing.splitter.recursive_splitter import RecursiveCharacterTextSplitter

# 无 stray-fence 的干净 case（case 1/4 见模块 docstring 已知局限）
_CLEAN_STRAY_FENCE_CASES = [2, 3, 5, 6]


def _long_code_block() -> str:
    code = "\n".join(f"print({index})  # line {index:03d}" for index in range(20))
    return "```\n" + code + "\n```"


def _short_code_block() -> str:
    return "```\ncode()\n```"


def build_cases() -> list[tuple[str, str]]:
    long_block = _long_code_block()
    short_block = _short_code_block()
    preamble = "开头说明段落，介绍下文内容。" + "背景信息，" * 10
    after_prose = "代码块之后的第一段正文。" + "这是后续正文内容，" * 20
    return [
        ("1-原始bug-超长代码块+正文", preamble + "\n\n" + long_block + "\n\n" + after_prose),
        ("2-反例A-纯文本连续串+短文", "一" * 200 + "\n\n" + "H" * 400 + "\n\n" + "二" * 50),
        ("3-反例B-短代码块+300字长段", short_block + "\n\n" + "X" * 300),
        ("4-文末代码块", preamble + "\n\n" + long_block + "\n"),
        ("5-代码块后跟heading", short_block + "\n\n## Usage\n\n用法说明正文。"),
        ("6-短代码块+短文", short_block + "\n\n短文内容，合并场景。"),
    ]


def _split(text: str):
    splitter = RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP)
    return splitter.split(text, {"doc_id": "t"})


def _case_ids() -> list[str]:
    return [case_name for case_name, _ in build_cases()]


class TestNoMicroChunk:
    @pytest.mark.parametrize("case_name,text", build_cases(), ids=_case_ids())
    def test_all_chunks_longer_than_3_chars(self, case_name, text):
        assert all(len(chunk.content) > 3 for chunk in _split(text))


class TestCodeBlockIntegrity:
    @pytest.mark.parametrize("case_name,text", build_cases(), ids=_case_ids())
    def test_fences_balanced_never_cut(self, case_name, text):
        # 代码块被切断 → 某 chunk 只剩开/闭一个 fence（奇数）；生产保证成对
        for chunk in _split(text):
            assert chunk.content.count("```") % 2 == 0

    @pytest.mark.parametrize("case_index", _CLEAN_STRAY_FENCE_CASES,
                             ids=[f"case{index}" for index in _CLEAN_STRAY_FENCE_CASES])
    def test_no_stray_fence_clean_cases(self, case_index):
        _, text = build_cases()[case_index - 1]
        for chunk in _split(text):
            position = chunk.content.find("```")
            assert position <= 0  # fence 只能出现在 chunk 起点或不存在


class TestIdDeterminism:
    @pytest.mark.parametrize("case_name,text", build_cases(), ids=_case_ids())
    def test_split_is_deterministic(self, case_name, text):
        first = _split(text)
        second = _split(text)
        assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
        assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


class TestCodeAdjacentShortSegment:
    def test_case6_short_code_block_short_text_merged(self):
        _, text = build_cases()[5]
        chunks = _split(text)
        assert len(chunks) == 1
        assert chunks[0].content.startswith("```")
        assert "短文内容" in chunks[0].content

    def test_case5_code_block_followed_by_heading_merged(self):
        _, text = build_cases()[4]
        chunks = _split(text)
        assert len(chunks) == 1
        assert "## Usage" in chunks[0].content
