"""unit：父子映射包装器（indexing/splitter/parent_child.py）——p{i} / p{i}:c{j} / parent_id / chunk_level。"""
from indexing.splitter.parent_child import ParentChildMappingWrapper
from indexing.splitter.recursive_splitter import RecursiveCharacterTextSplitter


def _make_wrapper():
    parent_splitter = RecursiveCharacterTextSplitter(10, 0, separators=[""], chunk_level="parent")
    child_splitter = RecursiveCharacterTextSplitter(5, 0, separators=[""])
    return ParentChildMappingWrapper(parent_splitter, child_splitter)


class TestSplit:
    def test_parents_and_children_ids_and_parent_id(self):
        wrapper = _make_wrapper()
        text = "aaaaabbbbbcccccddddd"  # 20 字符，无自然分隔符 → 父硬切 2 块
        parents, children = wrapper.split(text, {"doc_id": "doc"})

        assert [parent.chunk_id for parent in parents] == ["doc:p0", "doc:p1"]
        assert all(parent.origin_metadata.chunk_level == "parent" for parent in parents)

        assert [child.chunk_id for child in children] == [
            "doc:p0:c0", "doc:p0:c1", "doc:p1:c0", "doc:p1:c1",
        ]
        assert all(child.origin_metadata.chunk_level == "child" for child in children)
        assert children[0].metadata["parent_id"] == "doc:p0"
        assert children[1].metadata["parent_id"] == "doc:p0"
        assert children[2].metadata["parent_id"] == "doc:p1"
        assert children[3].metadata["parent_id"] == "doc:p1"

    def test_children_content_is_parent_subslice(self):
        wrapper = _make_wrapper()
        parents, children = wrapper.split("aaaaabbbbbcccccddddd", {"doc_id": "doc"})
        # 子块拼接 = 父块内容
        assert "".join(child.content for child in children[:2]) == parents[0].content
        assert "".join(child.content for child in children[2:]) == parents[1].content
