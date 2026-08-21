#!/usr/bin/env python3
"""scripts/df1_sync.py — 成果物を**ホストの外**へ出し続ける。

★ なぜこれが要るか(2026-08-20 に $17.90 を失って学んだ)

    ラン detector-firstlight-01 は、学習3本とマージを終え、検出器まで到達していた
    可能性が高い。だが**成果物をホストの外へ出す段がどこにも無かった**ため、
    ハード期限の自動 terminate でインスタンスのディスクごと消えた。
    残ったのは、人が途中で1回だけ手で回収した 13:24Z 時点のものだけである。

    ⛔ 事故の原因は「人が見ていなかったこと」ではない。**回収が人の常駐に依存する設計**
       だったことである。人が寝ても・接続が切れても・期限が先に来ても、結果は同じだった。

    ★ よってこの道具は、**人が一度も撃たなくても成果物が外に出ている**ことを保証する:
        ① 段の境目ごとに push(orchestrate 側の say() から呼ばれる)
        ② それとは別に、常駐して一定間隔で push(長い段の途中で落ちても失わない)
        ③ ★ 学習を始める**前に**導通を確かめ、通らなければランを始めさせない

★ 置き先は**別の private リポジトリ**である(本体リポジトリは public)。
  書き込みは deploy key(そのリポジトリ**だけ**に効く鍵)で行う。⛔ 個人アクセストークンは置かない。

⛔ モデルの重み(models/)は送らない —— 数 GB あり、再現には要らない
   (学習は決定論的で、レシピと注入集合から作り直せる)。

    python3 scripts/df1_sync.py init                    # 導通確認(★ 学習の前に)
    python3 scripts/df1_sync.py once --why "[T] 学習 1"  # 1回だけ送る
    python3 scripts/df1_sync.py daemon --interval 900   # 常駐して送り続ける
    python3 scripts/df1_sync.py status                  # 最後に送れた時刻を見る
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 送るもの(ホスト上の場所 -> 置き先での名前)
#   ⛔ data/jmmlu.jsonl(問題文そのもの)は送らない。
#   ★ data/injection は .ids と manifest のみ —— 問題文ではなく「どれを注入したか」の記録。
SYNC_DIRS: list[tuple[str, str]] = [
    ("reports", "reports"),
    ("data/cache", "cache"),
    ("data/injection", "injection"),
]

# 1ファイルの上限。これを超えるものは送らない(モデルの取り違えを防ぐ最後の砦)
MAX_FILE_BYTES = 200 * 1024 * 1024

SYNC_DIR = REPO / ".df1-sync"           # 置き先リポジトリの作業コピー
LOCK = REPO / "reports" / ".df1-sync.lock"
LOG = REPO / "reports" / "df1-sync.log"
STATE = REPO / "reports" / "df1-sync.json"
PIDFILE = REPO / "reports" / ".df1-sync.pid"
KEY = Path.home() / ".ssh" / "contamlab-artifacts"

PUSH_TRIES = 5
PUSH_WAIT_SEC = 20


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{ts()}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeError:
        # ★ 端末の文字コード(Windows の cp932 など)で落ちない。
        #   ⛔ ログが書けないことより、そこで例外を投げて同期が止まるほうが害である。
        sys.stdout.buffer.write(line.encode("utf-8", "replace") + b"\n")
        sys.stdout.flush()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args: str, cwd: Path | None = None, check: bool = True,
        timeout: int = 300) -> subprocess.CompletedProcess:
    """置き先リポジトリへの git。★ deploy key を明示し、鍵を1本に固定する。"""
    env = dict(os.environ)
    # ★ 鍵の場所は引用して渡す —— Windows の `\` は ssh に食われて消える
    #   (2026-08-21 に実際に踏んだ。実行先は Linux だが、手元で導通を試せないと意味がない)
    env["GIT_SSH_COMMAND"] = (
        f'ssh -i "{KEY.as_posix()}" -o IdentitiesOnly=yes '
        "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
    )
    env.setdefault("GIT_TERMINAL_PROMPT", "0")   # ⛔ 認証を人に聞きに行かせない
    # ★ git の出力は **UTF-8 として読む**。端末の文字コード任せにすると、
    #   日本語のコミットメッセージで読み取りスレッドが死に、同期が止まる
    #   (2026-08-21 に手元の cp932 で実際に踏んだ)。
    return subprocess.run(["git", *args], cwd=str(cwd or SYNC_DIR), env=env,
                          capture_output=True, check=check, timeout=timeout,
                          encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 置き場所の用意と導通確認
# ---------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    """★ 学習を始める**前に**、実際に push できることを確かめる。

    ⛔ ここで通らなければランを始めてはいけない。通らないまま回すと、
       前回と同じ「作ったのに外に出せないまま消える」に戻る。
    """
    url = args.url
    if not SYNC_DIR.exists():
        log(f"置き先を clone する: {url}")
        try:
            git("clone", "--depth", "1", url, str(SYNC_DIR), cwd=REPO)
        except subprocess.CalledProcessError as exc:
            log(f"★ clone に失敗: {exc.stderr.strip()[:400]}")
            return 1
    git("config", "user.email", "df1-sync@contamlab.invalid")
    git("config", "user.name", "df1-sync")

    # ★ 導通確認 —— 実際に1コミット push してみる。読めるだけでは足りない。
    beat = SYNC_DIR / "HEARTBEAT"
    beat.write_text(f"{ts()}  init from {platform.node()}\n", encoding="utf-8")
    git("add", "HEARTBEAT")
    git("commit", "-m", f"導通確認 {ts()}", check=False)
    if not push():
        log("★ 導通確認に失敗した。⛔ この状態で GPU を回してはいけない")
        return 1
    log("導通確認 OK —— 成果物は外へ出せる")
    save_state(ok=True, why="init")
    return 0


def push() -> bool:
    """push は諦めない(前回の事故は、諦めた先に起きた)。"""
    for attempt in range(1, PUSH_TRIES + 1):
        try:
            git("push", "origin", "HEAD")
            return True
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip()[:300]
            log(f"push が失敗(試行 {attempt}/{PUSH_TRIES}): {err}")
            if "non-fast-forward" in err or "fetch first" in err:
                # 別のホストが書いた場合。★ 相手を消さない —— rebase して自分の分を載せる
                try:
                    git("pull", "--rebase", "origin", "HEAD")
                except subprocess.CalledProcessError:
                    log("★ rebase にも失敗した")
        except subprocess.TimeoutExpired:
            log(f"push が時間切れ(試行 {attempt}/{PUSH_TRIES})")
        if attempt < PUSH_TRIES:
            time.sleep(PUSH_WAIT_SEC)
    return False


# ---------------------------------------------------------------------------
# 1回ぶんの送出
# ---------------------------------------------------------------------------
def collect_into_sync_dir() -> list[str]:
    """送るものを置き先の作業コピーへ写す。★ 中身が変わったものだけ。"""
    changed: list[str] = []
    for src_rel, dst_rel in SYNC_DIRS:
        src = REPO / src_rel
        if not src.is_dir():
            continue
        dst_root = SYNC_DIR / dst_rel
        for path in sorted(src.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                log(f"★ 大きすぎるので送らない({size/1e6:.0f}MB): {path.name}")
                continue
            dst = dst_root / path.relative_to(src)
            if dst.exists() and dst.stat().st_size == size \
                    and dst.stat().st_mtime >= path.stat().st_mtime:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            changed.append(str(dst.relative_to(SYNC_DIR)))
    return changed


def save_state(*, ok: bool, why: str, files: int = 0) -> None:
    STATE.write_text(json.dumps({
        "last_attempt_utc": ts(),
        "last_ok": ok,
        "why": why,
        "files": files,
        "sync_dir": str(SYNC_DIR),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_once(args: argparse.Namespace) -> int:
    if not SYNC_DIR.exists():
        log("★ 置き先がまだ無い(init を先に通すこと)")
        return 1
    lock = acquire_lock()
    if lock is None:
        log("別の同期が走っている。今回は見送る")
        return 0
    try:
        changed = collect_into_sync_dir()
        git("add", "-A")
        st = git("status", "--porcelain", check=False)
        if not st.stdout.strip():
            save_state(ok=True, why=args.why, files=0)
            return 0
        git("commit", "-m", f"{args.why} {ts()}", check=False)
        ok = push()
        save_state(ok=ok, why=args.why, files=len(changed))
        log(("送った" if ok else "★ 送れなかった") + f": {len(changed)} ファイル / {args.why}")
        return 0 if ok else 1
    except subprocess.SubprocessError as exc:
        log(f"★ 同期に失敗: {type(exc).__name__}: {exc}")
        save_state(ok=False, why=args.why)
        return 1
    finally:
        release_lock(lock)


def acquire_lock():
    """⛔ 常駐と段ごとの呼び出しがぶつかると、git の index が壊れる。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:      # Windows(テスト用)。実行はインスタンス上の Linux
        return LOCK.open("w")
    fh = LOCK.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def release_lock(fh) -> None:
    if fh is not None:
        fh.close()


# ---------------------------------------------------------------------------
# 常駐 —— ★ 長い段(検出器は 2.4 時間)の途中で落ちても失わないため
# ---------------------------------------------------------------------------
def cmd_daemon(args: argparse.Namespace) -> int:
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"常駐を開始({args.interval}秒ごと)")
    while True:
        cmd_once(argparse.Namespace(why="[定期]"))
        time.sleep(args.interval)


def cmd_status(args: argparse.Namespace) -> int:
    if not STATE.exists():
        print("まだ一度も同期していない")
        return 1
    print(STATE.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="置き先を用意し、実際に push できることを確かめる")
    p.add_argument("--url", default=os.environ.get(
        "DF1_SYNC_URL", "git@github.com:shati-napo/contamlab-artifacts.git"))
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("once", help="1回だけ送る")
    p.add_argument("--why", default="[手動]")
    p.set_defaults(func=cmd_once)

    p = sub.add_parser("daemon", help="常駐して送り続ける")
    p.add_argument("--interval", type=int, default=900)
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("status", help="最後に送れた時刻を見る")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
