"""scripts/df1_closeout.py の**分岐だけ**を試す。

★ なぜ試すのか —— この道具が誤ると、失うのは測定そのものである:

    ⛔ 回収より先に terminate  -> reports/ と応答キャッシュが永久に消える
    ⛔ 異常なのに PC を落とす  -> ユーザーは「無事に終わった」と誤読する
       (取り決め: **PC が落ちていないこと自体が異常の合図**)

  どちらも**離席中に一度だけ起きる**種類の事故で、手で試す機会が無い。
  だから ssh も scp も API も呼ばずに、**分岐の表**だけを押さえる。

★ ここで試すのは「終わったか / 途中で止まったか」の分岐だけである。
  ⛔ 測定結果(drop・p 値)による分岐は**そもそも存在しない**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import df1_closeout as C  # noqa: E402

TAG = "lambda-a100-df1-20260820"
DONE_STATE = {"tag": TAG, "stopped": "no", "done": "1", "out": "yes",
              "stage": "=== [DONE]", "stopped_text": "", "watchdog": ""}


def state(**over) -> dict:
    st = dict(DONE_STATE)
    st.update(over)
    return st


# ---------------------------------------------------------------------------
# 正常終了の定義(3つ全部そろったときだけ)
# ---------------------------------------------------------------------------
def test_正常終了は3条件がそろったときだけ():
    ok, reasons = C.verdict(state())
    assert ok and reasons == []


@pytest.mark.parametrize("over", [
    {"stopped": "yes"},          # 停止条件に当たった
    {"done": "0"},               # [DONE] が無い
    {"out": "no"},               # 検出器の出力が無い
])
def test_1つでも欠ければ異常(over):
    ok, reasons = C.verdict(state(**over))
    assert not ok and len(reasons) == 1


def test_判定は測定結果の中身を読まない():
    """drop も p 値も detected も、正常/異常の判定に入れない。"""
    src = Path(C.__file__).read_text(encoding="utf-8")
    body = src.split("def verdict", 1)[1].split("def ", 1)[0]
    for word in ("drop", "p_holm", "p_value", "adjusted_lcb", "detected"):
        assert word not in body, f"verdict() が {word} を読んでいる"


# ---------------------------------------------------------------------------
# close の分岐 —— ssh も scp も API も呼ばせない
# ---------------------------------------------------------------------------
class Args:
    def __init__(self, **kw):
        self.terminate = self.force_terminate = self.shutdown = False
        self.shutdown_delay = 60
        self.tag = TAG
        self.__dict__.update(kw)


@pytest.fixture
def stub(monkeypatch):
    calls = {"terminate": 0, "shutdown": 0, "power": 0, "collect": 0}

    def fake_terminate(args):
        calls["terminate"] += 1
        return calls.get("terminate_result", True)

    monkeypatch.setattr(C, "terminate", fake_terminate)
    monkeypatch.setattr(C, "shutdown_pc", lambda d: calls.__setitem__("shutdown", 1))
    monkeypatch.setattr(C, "restore_power", lambda: calls.__setitem__("power", 1))
    return calls


def arrange(monkeypatch, calls, *, st, collect_ok=True, verify_ok=True):
    monkeypatch.setattr(C, "remote_state", lambda a: st)
    monkeypatch.setattr(C, "collect",
                        lambda a: (calls.__setitem__("collect", 1),
                                   (collect_ok, [] if collect_ok else ["scp 失敗"]))[1])
    monkeypatch.setattr(C, "verify_local",
                        lambda t: (verify_ok, [] if verify_ok else ["JSON が壊れている"]))


def test_正常なら回収してから落とす(monkeypatch, stub, capsys):
    arrange(monkeypatch, stub, st=state())
    rc = C.cmd_close(Args(terminate=True, shutdown=True))
    assert rc == 0
    assert stub["collect"] == 1 and stub["terminate"] == 1 and stub["shutdown"] == 1


def test_異常なら_terminate_も_shutdown_もしない(monkeypatch, stub):
    arrange(monkeypatch, stub, st=state(stopped="yes", stopped_text="G4 に落ちた"))
    rc = C.cmd_close(Args(terminate=True, shutdown=True))
    assert rc == 1
    assert stub["terminate"] == 0, "★ 異常なのに GPU を落とした(再学習になる)"
    assert stub["shutdown"] == 0, "★ 異常なのに PC を落とした(異常の合図が消える)"


def test_回収に失敗したら落とさない(monkeypatch, stub):
    """⛔ reports/ と応答キャッシュはホスト上のものが原本である。"""
    arrange(monkeypatch, stub, st=state(), collect_ok=False)
    rc = C.cmd_close(Args(terminate=True, shutdown=True))
    assert rc == 1 and stub["terminate"] == 0 and stub["shutdown"] == 0


def test_手元で読めなければ落とさない(monkeypatch, stub):
    arrange(monkeypatch, stub, st=state(), verify_ok=False)
    rc = C.cmd_close(Args(terminate=True, shutdown=True))
    assert rc == 1 and stub["terminate"] == 0 and stub["shutdown"] == 0


def test_force_terminate_は異常でも落とすが回収失敗は覆さない(monkeypatch, stub):
    arrange(monkeypatch, stub, st=state(done="0"))
    assert C.cmd_close(Args(terminate=True, force_terminate=True)) == 1
    assert stub["terminate"] == 1, "人が明示したときは落とす"

    stub["terminate"] = 0
    arrange(monkeypatch, stub, st=state(done="0"), collect_ok=False)
    assert C.cmd_close(Args(terminate=True, force_terminate=True)) == 1
    assert stub["terminate"] == 0, "★ 回収の失敗だけは --force でも覆さない"


def test_terminate_の確認が取れなければ_PC_は落とさない(monkeypatch, stub):
    """一覧から id が消えるのを見届けられなかった = まだ課金されているかもしれない。"""
    arrange(monkeypatch, stub, st=state())
    monkeypatch.setattr(C, "terminate", lambda a: (stub.__setitem__("terminate", 1), False)[1])
    C.cmd_close(Args(terminate=True, shutdown=True))
    assert stub["shutdown"] == 0


def test_電源設定は正常でも異常でも戻す(monkeypatch, stub):
    arrange(monkeypatch, stub, st=state(stopped="yes"))
    C.cmd_close(Args())
    assert stub["power"] == 1


def test_terminate_を渡さなければ落ちない(monkeypatch, stub):
    arrange(monkeypatch, stub, st=state())
    assert C.cmd_close(Args(shutdown=True)) == 0
    assert stub["terminate"] == 0 and stub["shutdown"] == 0


# ---------------------------------------------------------------------------
# 手元での検証 —— scp の成功は中身の保証ではない
# ---------------------------------------------------------------------------
def test_手元検証はアーム数と統計量とキャッシュを見る(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "LOCAL_DEST", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "cache").mkdir()
    out = tmp_path / "reports" / f"{C.RUN}.{TAG}.json"

    ok, problems = C.verify_local(TAG)
    assert not ok and "手元に無い" in problems[0]

    def write(models):
        out.write_text(json.dumps({"models": models}), encoding="utf-8")

    full = {"drop": 0.0, "p_holm": 1.0, "adjusted_lcb": 0.0, "detected": False}
    write([dict(full, name="a")])                      # 1 本しかない
    assert not C.verify_local(TAG)[0]

    write([dict(full, name="a"), dict(full, name="b")])
    ok, problems = C.verify_local(TAG)
    assert not ok and any("キャッシュ" in p for p in problems)

    (tmp_path / "cache" / f"responses.{TAG}.jsonl").write_text("{}\n", encoding="utf-8")
    assert C.verify_local(TAG)[0]

    write([dict(full, name="a"), {"name": "b", "drop": 0.0}])   # 統計量が欠けている
    assert not C.verify_local(TAG)[0]


def test_壊れた_JSON_を通さない(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "LOCAL_DEST", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / f"{C.RUN}.{TAG}.json").write_text("{壊れ", encoding="utf-8")
    ok, problems = C.verify_local(TAG)
    assert not ok and "JSON" in problems[0]


# ---------------------------------------------------------------------------
# ホストへの問い合わせ —— ★ 読むだけ。書き込む語を持たない
# ---------------------------------------------------------------------------
def test_問い合わせは読み取りだけで自分自身に当たらない():
    probe = C.PROBE
    for word in ("rm ", "mv ", "sed -i", "kill", "systemctl", "> reports", "touch "):
        assert word not in probe, f"問い合わせに書き込みの語がある: {word}"
    # pgrep が**自分の ssh コマンド行**に当たらないよう角括弧で1文字外してある
    assert "df1-orchestrat[e].sh" in probe
    assert "train_lor[a].py" in probe
