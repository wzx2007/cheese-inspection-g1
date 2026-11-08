# -*- coding: utf-8 -*-
"""提交前自检：仓库里有没有混进真实密钥 / 密码 / 内网地址。

仓库是公开的（github.com/wzx2007/cheese-inspection-g1），真实 key 只放在
``scripts/voice/.env`` 里，已由 ``.gitignore`` 挡住。但"忘记加新文件"是最常见的
事故，推之前跑一下这个脚本比靠记忆靠谱：

    python scripts/check_secrets.py            # 只查工作树
    python scripts/check_secrets.py --history  # 连全部 git 历史一起查
    python scripts/check_secrets.py --staged   # 只查已 git add 的内容（当钩子用）

退出码 0 = 干净，1 = 有命中。命中只打印前 8 个字符，不把密钥本身刷到终端/日志里。
被 ``.gitignore`` 正常挡住的 ``.env`` 只作提示，不算违规——否则这个脚本永远是红的，
红久了的检查等于没有检查。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 只认能确认是密钥的形状，不认 "api_key" 这种普通单词——否则满屏注释全是误报
RULES = [
    ("DashScope / OpenAI key", r"sk-[A-Za-z0-9]{20,}"),
    ("AWS AccessKey", r"AKIA[0-9A-Z]{16}"),
    ("GitHub token", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    ("Slack token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("Google API key", r"AIza[0-9A-Za-z_-]{35}"),
    ("私钥块", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("硬编码 Bearer", r"Bearer\s+[A-Za-z0-9._\-]{24,}"),
    ("内网 IP", r"192\.168\.\d{1,3}\.\d{1,3}"),
]
# 刻意不放"机器人 SSH 密码"这条规则：要检出它就得把这个密码本身写成字面量，
# 等于把密码贴进公开仓库——检测规则不能成为泄露源。SSH 密码靠 .gitignore 兜底。
COMPILED = [(name, re.compile(pat)) for name, pat in RULES]

# 占位符要放过，否则历史检查永远是红的（历史改不掉，红久了的检查等于没有检查）。
# 只认"明显的假货"：整段同一个字母重复 / 带 REPLACE_ME 之类字样。
FAKE = re.compile(
    r"^(?:sk|gh[pousr]|AKIA|xox[baprs])[-_]"
    r"(?:([A-Za-z0-9])\1{5,}|.*(?:REPLACE|YOUR|PLACEHOLDER|EXAMPLE|DUMMY|填))",
    re.IGNORECASE)


TEXT_SUFFIX = {".py", ".md", ".json", ".sh", ".txt", ".example", ".yaml", ".yml",
               ".toml", ".cfg", ".ini", ".env", ".html"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist"}
SKIP_FILES = {"check_secrets.py"}          # 本文件自己写着这些正则，别自证


def mask(text: str) -> str:
    return text[:8] + "…" if len(text) > 8 else text


def is_ignored(path: Path) -> bool:
    proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                          cwd=ROOT, capture_output=True)
    return proc.returncode == 0


def scan_text(text: str, label: str, hits: list, fakes: list) -> None:
    for name, pat in COMPILED:
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            if FAKE.match(m.group(0)):
                fakes.append("%s:%d %s" % (label, line, m.group(0)))
                continue
            hits.append((label, line, name, mask(m.group(0))))


def scan_worktree() -> tuple:
    hits, notes, fakes = [], [], []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIX and path.name not in (
                ".env", ".gitignore", ".gitattributes"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        label = str(path.relative_to(ROOT)).replace("\\", "/")

        found: list = []
        scan_text(text, label, found, fakes)
        if not found:
            continue
        if all(name in ("DashScope / OpenAI key",) for _, _, name, _ in found) \
                and path.name == ".env" and is_ignored(path):
            notes.append("%s  真实 key 在这，已被 .gitignore 挡住（正常）" % label)
        else:
            hits.extend(found)
    return hits, notes, fakes


def scan_staged() -> tuple:
    names = subprocess.run(["git", "diff", "--cached", "--name-only"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    hits, fakes = [], []
    for name in names:
        proc = subprocess.run(["git", "show", ":" + name],
                              cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            scan_text(proc.stdout, "[staged] " + name, hits, fakes)
    return hits, fakes


def scan_history() -> tuple:
    """逐提交 grep。历史里的密钥没法靠改文件消除，只能吊销。

    一条合并正则跑一次就够：9 条规则 x 4 个提交要起 36 个 git 进程，
    每个都要遍历 100 个文件（含 10 MB 的 onnx），实测会拖到分钟级。
    """
    hits, fakes = [], []
    combined = "|".join("(%s)" % pat for _, pat in RULES)
    commits = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    for commit in commits:
        proc = subprocess.run(["git", "grep", "-nIaE", combined, commit, "--", "."],
                              cwd=ROOT, capture_output=True, text=True)
        for raw in proc.stdout.splitlines():
            # 形如 <commit>:<path>:<line>:<content>
            parts = raw.split(":", 3)
            if len(parts) < 4:
                continue
            content = parts[3]
            if parts[1].rsplit("/", 1)[-1] in SKIP_FILES:   # 本文件写着这些正则，别自证
                continue
            body = content.split("=", 1)[-1].strip()
            if FAKE.match(body) or FAKE.match(content):
                fakes.append("[%s] %s:%s %s" % (commit[:7], parts[1], parts[2], body))
                continue
            name = "命中"
            for rule_name, pat in RULES:
                if re.search(pat, content):
                    name = rule_name
                    break
            hits.append(("[%s] %s" % (commit[:7], parts[1]),
                         int(parts[2]) if parts[2].isdigit() else 0,
                         name, mask(content)))
    return hits, fakes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仓库密钥自检（默认只查工作树）")
    parser.add_argument("--history", action="store_true", help="连全部 git 历史一起查")
    parser.add_argument("--staged", action="store_true", help="只查已 git add 的内容")
    args = parser.parse_args()

    notes: list = []
    fakes: list = []
    if args.staged:
        hits, fakes = scan_staged()
    else:
        hits, notes, fakes = scan_worktree()
    if args.history:
        more_hits, more_fakes = scan_history()
        hits += more_hits
        fakes += more_fakes

    for note in notes:
        print("ℹ %s" % note)
    if fakes:
        print("ℹ 放过 %d 处占位符（历史里的旧占位符也算）" % len(fakes))
        for fake in fakes[:5]:
            print("   %s" % fake)
        if len(fakes) > 5:
            print("   …还有 %d 处" % (len(fakes) - 5))

    if not hits:
        print("✅ 干净：没有发现密钥、密码或内网地址")
        return 0

    print("⚠ 发现 %d 处可疑内容：" % len(hits))
    for label, line, name, sample in hits:
        where = "%s:%d" % (label, line) if line else label
        print("  %-52s %-22s %s" % (where, name, sample))
    print()
    print("真实 key 只能放 scripts/voice/.env（已在 .gitignore 里）。")
    print("已经提交过的密钥，改文件没用——要去控制台吊销重发。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
