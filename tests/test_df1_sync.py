"""scripts/df1_sync.py の**保全としての性質**を試す。

★ なぜ試すのか —— 2026-08-20 に、学習3本ぶんの成果物が
  「外へ出す段が無い」という理由だけで消えた($17.90)。
  この道具はその穴を塞ぐためだけに在る。よって試すのは速さでも綺麗さでもなく:

    ① 送る対象に **models/(数 GB の重み)と問題文そのもの**が入っていないこと
    ② push できないとき、**init は失敗を返す**こと(= ランを始めさせない)
    ③ 段の途中で送れなくても、次の機会に**送り直す**こと
    ④ 常駐と段ごとの呼び出しが**同時に走らない**こと(git の index が壊れる)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import df1_sync as S  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """ホスト側と置き先を tmp に作り、git も push も呼ばせない。"""
    host = tmp_path / "host"
    sync = tmp_path / "sync"
    for rel, _ in S.SYNC_DIRS:
        (host / rel).mkdir(parents=True, exist_ok=True)
    sync.mkdir()
    (host / "reports").mkdir(exist_ok=True)
    monkeypatch.setattr(S, "REPO", host)
    monkeypatch.setattr(S, "SYNC_DIR", sync)
    monkeypatch.setattr(S, "LOCK", host / "reports" / ".lock")
    monkeypatch.setattr(S, "LOG", host / "reports" / "sync.log")
    monkeypatch.setattr(S, "STATE", host / "reports" / "sync.json")
    monkeypatch.setattr(S, "PUSH_WAIT_SEC", 0)
    return host, sync


# ---------------------------------------------------------------------------
# ① 送る対象 —— ⛔ 重みと問題文は外へ出さない
# ---------------------------------------------------------------------------
def test_重みと問題文は送る対象に入っていない():
    srcs = [src for src, _ in S.SYNC_DIRS]
    assert "models" not in srcs, "★ 数 GB の重みを送ろうとしている"
    assert not any(s.endswith("jmmlu.jsonl") for s in srcs), "★ 問題文そのものを送ろうとしている"
    assert "reports" in srcs and "data/cache" in srcs


def test_大きすぎるファイルは送らない(sandbox, monkeypatch):
    host, sync = sandbox
    monkeypatch.setattr(S, "MAX_FILE_BYTES", 100)
    (host / "reports" / "小.txt").write_text("ok", encoding="utf-8")
    (host / "reports" / "大.bin").write_bytes(b"x" * 500)
    changed = S.collect_into_sync_dir()
    assert any("小.txt" in c for c in changed)
    assert not any("大.bin" in c for c in changed)
    assert not (sync / "reports" / "大.bin").exists()


def test_変わっていないものは送り直さない(sandbox):
    host, _ = sandbox
    (host / "reports" / "a.txt").write_text("1", encoding="utf-8")
    assert len(S.collect_into_sync_dir()) == 1
    assert S.collect_into_sync_dir() == []
    (host / "reports" / "a.txt").write_text("22", encoding="utf-8")
    assert len(S.collect_into_sync_dir()) == 1


# ---------------------------------------------------------------------------
# ② 導通確認 —— ★ 通らなければランを始めさせない
# ---------------------------------------------------------------------------
def test_push_できなければ_init_は失敗を返す(sandbox, monkeypatch):
    monkeypatch.setattr(S, "git", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(S, "push", lambda: False)
    rc = S.cmd_init(argparse.Namespace(url="git@example.invalid:x/y.git"))
    assert rc == 1, "★ 外へ出せないのに 0 を返すと、そのまま GPU が回ってしまう"


def test_push_できれば_init_は成功する(sandbox, monkeypatch):
    monkeypatch.setattr(S, "git", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(S, "push", lambda: True)
    assert S.cmd_init(argparse.Namespace(url="git@example.invalid:x/y.git")) == 0


# ---------------------------------------------------------------------------
# ③ 1回ぶんの送出
# ---------------------------------------------------------------------------
def calls_recorder(monkeypatch, *, status_dirty=True):
    seen = []

    def fake_git(*args, **kw):
        seen.append(args)
        out = " M reports/a.txt" if (args and args[0] == "status" and status_dirty) else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(S, "git", fake_git)
    return seen


def test_変化が無ければ_commit_も_push_もしない(sandbox, monkeypatch):
    seen = calls_recorder(monkeypatch, status_dirty=False)
    monkeypatch.setattr(S, "push", lambda: pytest.fail("★ 変化が無いのに push した"))
    assert S.cmd_once(argparse.Namespace(why="[T]")) == 0
    assert not any(a and a[0] == "commit" for a in seen)


def test_変化があれば_commit_して_push_する(sandbox, monkeypatch):
    host, _ = sandbox
    (host / "reports" / "a.txt").write_text("x", encoding="utf-8")
    seen = calls_recorder(monkeypatch)
    pushed = []
    monkeypatch.setattr(S, "push", lambda: (pushed.append(1), True)[1])
    assert S.cmd_once(argparse.Namespace(why="[T] 学習 1")) == 0
    assert any(a and a[0] == "commit" for a in seen) and pushed


def test_送れなければ失敗を残す(sandbox, monkeypatch):
    host, _ = sandbox
    (host / "reports" / "a.txt").write_text("x", encoding="utf-8")
    calls_recorder(monkeypatch)
    monkeypatch.setattr(S, "push", lambda: False)
    assert S.cmd_once(argparse.Namespace(why="[T]")) == 1
    assert '"last_ok": false' in S.STATE.read_text(encoding="utf-8")


def test_置き先が無ければ何もせず失敗を返す(sandbox, monkeypatch):
    _, sync = sandbox
    sync.rmdir()
    monkeypatch.setattr(S, "git", lambda *a, **k: pytest.fail("★ 置き先が無いのに git を呼んだ"))
    assert S.cmd_once(argparse.Namespace(why="[T]")) == 1


# ---------------------------------------------------------------------------
# push の粘り —— ★ 諦めた先に前回の事故がある
# ---------------------------------------------------------------------------
def test_push_は何度も試し_割り込まれたら_rebase_する(sandbox, monkeypatch):
    tries = []

    def fake_git(*args, **kw):
        tries.append(args[0])
        if args[0] == "push" and len([t for t in tries if t == "push"]) == 1:
            raise subprocess.CalledProcessError(1, args, stderr="! [rejected] non-fast-forward")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(S, "git", fake_git)
    assert S.push() is True
    assert "pull" in tries, "★ 割り込まれたのに rebase していない"
    assert tries.count("push") == 2


def test_push_は諦めたときだけ_False_を返す(sandbox, monkeypatch):
    def always_fail(*args, **kw):
        raise subprocess.CalledProcessError(1, args, stderr="fatal: 繋がらない")

    monkeypatch.setattr(S, "git", always_fail)
    assert S.push() is False


# ---------------------------------------------------------------------------
# ④ 同時実行 —— 常駐と段ごとの呼び出しがぶつからない
# ---------------------------------------------------------------------------
def test_別の同期が走っていれば見送る(sandbox, monkeypatch):
    monkeypatch.setattr(S, "acquire_lock", lambda: None)
    monkeypatch.setattr(S, "git", lambda *a, **k: pytest.fail("★ 錠が取れていないのに git を呼んだ"))
    assert S.cmd_once(argparse.Namespace(why="[T]")) == 0


# ---------------------------------------------------------------------------
# 鍵の扱い —— ⛔ 個人アクセストークンを置かない
# ---------------------------------------------------------------------------
def test_書き込みは_deploy_key_だけで行う():
    src = Path(S.__file__).read_text(encoding="utf-8")
    for word in ("ghp_", "gho_", "GITHUB_TOKEN", "x-access-token", "https://"):
        assert word not in src, f"★ トークンでの書き込みが混じっている: {word}"
    assert "IdentitiesOnly=yes" in src, "★ 鍵を1本に固定していない"
    assert "GIT_TERMINAL_PROMPT" in src, "★ 認証を人に聞きに行く余地がある"


# ---------------------------------------------------------------------------
# オーケストレータ側の配線 —— ★ 道具が在っても呼ばれなければ意味がない
# ---------------------------------------------------------------------------
def test_オーケストレータは学習より先に導通を確かめる():
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "df1-orchestrate.sh").read_text(encoding="utf-8")
    i_init = src.index('df1_sync.py" init') if 'df1_sync.py" init' in src \
        else src.index('"$SYNC" init')
    i_train = src.index("train_one 1")
    assert i_init < i_train, "★ 外へ出せるか確かめる前に学習を始めている"
    # 導通が取れなければ止まる(⛔ 失敗しても続けてはいけない唯一の同期)
    tail = src[i_init:i_init + 300]
    assert "stop_run" in tail, "★ 導通に失敗してもランを始めてしまう"


def test_段の境目と停止時に必ず送る():
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "df1-orchestrate.sh").read_text(encoding="utf-8")
    say = src[src.index("say() {"):src.index("say() {") + 120]
    assert "sync_now" in say, "★ 段の境目で送っていない"
    stop = src[src.index("stop_run() {"):src.index("stop_run() {") + 400]
    assert "sync_now" in stop, "★ 止まるときに送っていない(失敗の記録が消える)"
    assert "sync_now \"[DONE] 最終\"" in src, "★ 完走時の最後の1回が無い"


def test_同期の失敗ではランを止めない():
    """⛔ 導通確認**以外**の同期は、失敗しても止めない(常駐が拾う)。"""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "df1-orchestrate.sh").read_text(encoding="utf-8")
    fn = src[src.index("sync_now() {"):src.index("sync_now() {") + 200]
    assert "|| true" in fn, "★ 同期の失敗でランが落ちる"
    assert "timeout" in fn, "★ 同期が固まるとランが止まる"
