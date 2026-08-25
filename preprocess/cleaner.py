"""确定性清洗器: 一个确定性的纯函数 clean(text) -> text, 对格式归一化后的 Markdown 做文本级去噪

四步清洗（每步自身幂等，顺序不影响幂等）
1. 零宽/控制字符清理：BOM、soft hyphen、零宽空格/连接符、LRM/RLM 等不可见字符
2. HTML 残留标签清理：<[^>]*> 整体剔除（不处理转义实体，避免误伤 &lt; 语义）
3. URL 截断去参 + 邮箱小写: URL 去 ?query/#fragment 保留可读路径；邮箱统一小写
4. 样板行去除: 版权/免责/关注公众号/read more 等整行剔除

预处理管线切入时机: loader 根据格式产出归一化 Markdown 之后, 诊断/路由之前

设计原则：
- 幂等：clean(clean(x)) == clean(x)。
- 不做跨页去重与页眉页脚去除：输入无页边界，纯 text→text 无法安全判定。
- 不做占位符替换：URL 可能是可检索内容，截断去参但保留可读路径；
  因不产占位符, 负向断言（排除清洗器自身占位符）在本清洗器内平凡成立，无需额外排除表。
- 只对新格式归一化 MD 开；.md/.txt 默认不过 cleaner（CLEAN_PLAIN_TEXT=False），
  保 private_v6 字节级不变。
"""
import re

# 零宽/不可见/控制字符（含 BOM、soft hyphen、零宽空格/连接符、LRM/RLM、LRE/RLE 等）
_CONTROL_RE = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u00ad"
    r"\u200b-\u200f\u202a-\u202e\u2060\ufeff]"
)

# HTML 残留标签（含属性），整体剔除；不处理转义实体，避免误伤正常 &lt; 语义
_HTML_TAG_RE = re.compile(r"<[^>]*>")

# URL：匹配到空白或右括号为止（保留 query/fragment 以在截断时丢弃），再按 ?/# 截断
_URL_RE = re.compile(r"https?://[^\s)]+")

# 邮箱：归一化为小写（幂等）
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# 样板行（整行命中即剔除）：版权/免责/关注公众号/read more 等
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"copyright\s*©?|版权所有|版权声明|免责声明|声明[:：]|"
    r"关注公众号|扫描二维码|长按识别|"
    r"read\s+more|阅读原文|查看更多|"
    r"本文(?:由|来自)|转载(?:请|须|需)|未经授权|"
    r"如(?:有|涉及)侵权|版权归.*?所有"
    r").*$",
    re.IGNORECASE | re.MULTILINE,
)


def _truncate_url(match: re.Match) -> str:
    """截断 URL：去掉 query 与 fragment，保留 scheme/domain/path。"""
    token = match.group(0)
    return token.split("?", 1)[0].split("#", 1)[0]


def clean(text: str) -> str:
    """确定性、幂等地清理文本，返回清洗后的 text。

    Args:
        text: 待清洗的归一化 Markdown 文本。

    Returns:
        str：清洗后的文本；对任意输入满足 clean(clean(text)) == clean(text)。
    """
    text = _CONTROL_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _URL_RE.sub(_truncate_url, text)
    text = _EMAIL_RE.sub(lambda m: m.group(0).lower(), text)
    text = _BOILERPLATE_RE.sub("", text)
    return text
