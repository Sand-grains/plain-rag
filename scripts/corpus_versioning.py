"""语料本地版本化脚本
备份 data/ + benchmark/*.json, 独立本地 git 仓库
（与 docs-backup.sh 的区别在于 docs-backup.sh 备份的是 docs/ 下的内部迭代文档, 而这个文件版本化的是 data/ 下的语料）

解决 data/ 被主仓库 gitignore 导致语料文档无版本、不可复现的问题;
docs-backup.sh 只备份 docs/ 不备份 data/, 语料版本化需单独落地。

核心机制:
    - SHA 固化: 每次快照生成 manifest(相对路径 -> sha256), 校验语料完整性, 换机/重装后可校验恢复。
    - 可复现性 = 工件版本化 + SHA 固化, 而非生成过程确定性(对齐 MimirQ 思路)。
    - 与 benchmark 对接: 自建标注(private_v5/v6/private_builtin)与 public.json 保留跟踪;
      private_crawler.json 不进主仓库(.gitignore 排除)但纳入本地版本化, 缺失即 skip。

用法示例::

    uv run python scripts/corpus_versioning.py snapshot -m "语料快照"
    uv run python scripts/corpus_versioning.py verify
    uv run python scripts/corpus_versioning.py restore
    uv run python scripts/corpus_versioning.py manifest

触发时机为语料变更后手动跑 uv run python scripts/corpus_versioning.py snapshot

公共接口:
    - build_manifest: 计算语料 SHA 清单(相对路径 -> sha256)
    - snapshot_corpus: 快照语料到独立仓库并提交
    - verify_corpus: 校验语料完整性(SHA 比对)
    - restore_corpus: 从独立仓库恢复语料
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 独立运行(uv run python scripts/corpus_versioning.py)时确保能 import config(硬约束 7: __file__ 推导)
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from config import CORPUS_REPO_DIR, _PROJECT_ROOT

# 语料纳入范围: data/ 全量 + benchmark 标注 JSON(private_crawler.json 缺失即跳过)
_CORPUS_INCLUDE_DIRS = ("data",)
_CORPUS_INCLUDE_GLOBS = ("benchmark/*.json",)

# 独立仓库内 manifest 文件名
_MANIFEST_NAME = ".corpus-manifest.json"


def _git(repo_dir: Path, *args: str) -> str:
    """在独立仓库内执行 git 命令, 返回 stdout(去尾换行)。

    Args:
        repo_dir: 独立仓库根目录。
        args: git 子命令与参数。

    Returns:
        str: git 命令 stdout。

    Raises:
        RuntimeError: git 命令失败(非零退出)。
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {result.stderr.strip()}")
    return result.stdout.strip()


def _ensure_repo(repo_dir: Path) -> None:
    """确保独立仓库存在且为 git 仓库(不存在则 init, 并配置本地 user)。

    Args:
        repo_dir: 独立仓库根目录。
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        _git(repo_dir, "init", "-q")
        # 本地 user 配置(独立仓库, 不依赖全局 git config)
        _git(repo_dir, "config", "user.name", "plain-rag-corpus")
        _git(repo_dir, "config", "user.email", "corpus@local")


def _collect_files(corpus_root: Path) -> list[Path]:
    """收集纳入版本化的语料文件(data/ 全量 + benchmark/*.json, 缺失即跳过)。

    Args:
        corpus_root: 项目根目录。

    Returns:
        list[Path]: 纳入版本化的文件绝对路径列表(按相对路径排序)。
    """
    files: list[Path] = []
    for include_dir in _CORPUS_INCLUDE_DIRS:
        base = corpus_root / include_dir
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file())
    for glob_pattern in _CORPUS_INCLUDE_GLOBS:
        files.extend(path for path in corpus_root.glob(glob_pattern) if path.is_file())
    # 去重 + 按相对路径排序, 保证 manifest 确定性
    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(files, key=lambda p: str(p.relative_to(corpus_root))):
        rel = str(path.relative_to(corpus_root)).replace("\\", "/")
        if rel not in seen:
            seen.add(rel)
            unique.append(path)
    return unique


def _sha256_file(path: Path) -> str:
    """计算单文件 sha256(分块读, 内存友好)。

    Args:
        path: 文件路径。

    Returns:
        str: 64 位十六进制 sha256。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(corpus_root: Path) -> dict:
    """计算语料 SHA 清单(相对路径 -> sha256)。

    Args:
        corpus_root: 项目根目录。

    Returns:
        dict: {"version", "created_at", "files": {相对路径: sha256}}。
    """
    files = _collect_files(corpus_root)
    return {
        "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            str(path.relative_to(corpus_root)).replace("\\", "/"): _sha256_file(path)
            for path in files
        },
    }


def snapshot_corpus(corpus_root: Path, repo_dir: Path, message: str) -> str:
    """快照语料到独立仓库并提交, 返回提交哈希。

    流程: 确保仓库 -> 镜像语料文件进仓库工作树 -> 写 manifest -> git add + commit。
    镜像采用复制(内容字节不变), 独立仓库工作树即语料快照。

    Args:
        corpus_root: 项目根目录。
        repo_dir: 独立仓库根目录。
        message: 快照提交信息。

    Returns:
        str: 快照提交哈希(短 12 位)。
    """
    _ensure_repo(repo_dir)
    manifest = build_manifest(corpus_root)
    # 镜像语料文件进仓库工作树(先清空旧镜像, 保证与当前语料一致)
    for include_dir in _CORPUS_INCLUDE_DIRS:
        target = repo_dir / include_dir
        if target.exists():
            shutil.rmtree(target)
    for glob_pattern in _CORPUS_INCLUDE_GLOBS:
        for path in corpus_root.glob(glob_pattern):
            if path.is_file():
                target = repo_dir / path.name
                if target.exists():
                    target.unlink()
    for path in _collect_files(corpus_root):
        rel = path.relative_to(corpus_root)
        target = repo_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # 写 manifest 到仓库根
    (repo_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _git(repo_dir, "add", "-A")
    try:
        _git(repo_dir, "commit", "-q", "-m", message)
    except RuntimeError as exc:
        # L5: 语料无变化时 git commit 返回非零("nothing to commit"), 视为幂等成功而非失败
        if "nothing to commit" not in str(exc).lower():
            raise
    return _git(repo_dir, "rev-parse", "--short", "HEAD")


def verify_corpus(corpus_root: Path, repo_dir: Path) -> tuple[bool, list[str]]:
    """校验语料完整性: 重算 SHA 与仓库内 manifest 比对。

    Args:
        corpus_root: 项目根目录。
        repo_dir: 独立仓库根目录。

    Returns:
        tuple[bool, list[str]]: (是否一致, 不一致项描述列表)。
    """
    manifest_path = repo_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return False, ["独立仓库无 manifest, 尚未快照"]
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_manifest(corpus_root)
    mismatches: list[str] = []
    stored_files = stored.get("files", {})
    current_files = current.get("files", {})
    for rel, digest in current_files.items():
        if stored_files.get(rel) != digest:
            mismatches.append(f"{rel}: 内容已变更")
    for rel in stored_files:
        if rel not in current_files:
            mismatches.append(f"{rel}: 已缺失")
    return (not mismatches), mismatches


def restore_corpus(repo_dir: Path, target_root: Path) -> int:
    """从独立仓库工作树恢复语料到目标目录(覆盖式复制)。

    Args:
        repo_dir: 独立仓库根目录。
        target_root: 目标项目根目录。

    Returns:
        int: 恢复的文件数。
    """
    if not (repo_dir / _MANIFEST_NAME).exists():
        raise RuntimeError("独立仓库无 manifest, 无法恢复")
    manifest = json.loads((repo_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    restored = 0
    for rel in manifest.get("files", {}):
        source = repo_dir / rel
        if not source.is_file():
            continue
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored += 1
    return restored


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: snapshot / verify / restore / manifest 四个子命令。

    --repo 通过 parent parser 挂到每个子命令, 子命令前后均可传(全局与子命令级都接受)。
    """
    parser = argparse.ArgumentParser(prog="corpus_versioning", description="语料本地版本化(011-5)")
    parser.add_argument("--repo", default=CORPUS_REPO_DIR, help="独立仓库路径(缺省 CORPUS_REPO_DIR)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # parent 的 --repo 用 SUPPRESS: 子命令后未显式传时不覆盖主 parser 的值(支持前后两种写法)
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--repo", default=argparse.SUPPRESS, help="独立仓库路径(缺省 CORPUS_REPO_DIR)")

    parser_snapshot = subparsers.add_parser("snapshot", help="快照语料到独立仓库并提交", parents=[parent])
    parser_snapshot.add_argument("-m", "--message", default="corpus snapshot", help="快照提交信息")

    subparsers.add_parser("verify", help="校验语料完整性(SHA 比对)", parents=[parent])
    subparsers.add_parser("restore", help="从独立仓库恢复语料", parents=[parent])
    subparsers.add_parser("manifest", help="打印当前语料 SHA 清单摘要", parents=[parent])
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 返回进程退出码(0=成功, 1=失败)。

    Args:
        argv: 命令行参数列表, 缺省取 sys.argv。

    Returns:
        int: 进程退出码。
    """
    args = _build_parser().parse_args(argv)
    repo_dir = Path(args.repo)
    corpus_root = _PROJECT_ROOT
    if args.command == "snapshot":
        commit = snapshot_corpus(corpus_root, repo_dir, args.message)
        print(f"语料已快照: {commit} -> {repo_dir}")
        return 0
    if args.command == "verify":
        ok, mismatches = verify_corpus(corpus_root, repo_dir)
        if ok:
            print("语料完整性校验通过")
            return 0
        print("语料完整性校验失败:")
        for item in mismatches:
            print(f"  - {item}")
        return 1
    if args.command == "restore":
        count = restore_corpus(repo_dir, corpus_root)
        print(f"已从独立仓库恢复 {count} 个文件")
        return 0
    if args.command == "manifest":
        manifest = build_manifest(corpus_root)
        print(f"语料版本: {manifest['version']}")
        print(f"文件数: {len(manifest['files'])}")
        return 0
    return 1  # pragma: no cover  # argparse required subparsers 保证 command 恒为四者之一


if __name__ == "__main__":
    sys.exit(main())
