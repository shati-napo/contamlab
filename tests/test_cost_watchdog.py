"""tests/test_cost_watchdog.py — 費用ウォッチドッグの検査。

★ ラン `lambda-ladder-01` の事故($27.82 の超過)の直接の原因は
  「見張りを一度も試さなかった」ことである。ここで押さえるのは、事故の再発に
  直結する4点に絞る:

    ① 期限は**絶対時刻で凍結**され、プロセスを上げ直しても伸びない
    ② terminate は「送れた」ではなく「**一覧から消えた**」で成功と呼ぶ
    ③ 対象が1台に決まらないときは**推測しない**
    ④ 課金される機械の外では arm できない

  ⛔ ネットワークには一切出ない。実 API は叩かない。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "cost_watchdog", REPO_ROOT / "scripts" / "cost_watchdog.py"
)
cw = importlib.util.module_from_spec(_SPEC)
sys.modules["cost_watchdog"] = cw
_SPEC.loader.exec_module(cw)


class FakeApi:
    """terminate の呼ばれ方と、一覧の見え方を制御できる偽 API。"""

    def __init__(self, alive=("i-target",), fail_times=0):
        self.alive = list(alive)
        self.fail_times = fail_times
        self.terminate_calls: list[list[str]] = []
        self.list_calls = 0

    def terminate(self, ids):
        self.terminate_calls.append(list(ids))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("API が落ちている")
        self.alive = [i for i in self.alive if i not in ids]
        return {}

    def list_instances(self):
        self.list_calls += 1
        return [{"id": i} for i in self.alive]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cw.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# ① 期限の凍結
# ---------------------------------------------------------------------------
def test_期限は単価と上限から時間に直される():
    started = 1_000_000.0
    # $18 / $1.99 per h = 9.045… h
    deadline = cw.compute_deadline(started, hard_usd=18.0, price_per_hour=1.99)
    assert deadline == pytest.approx(started + (18.0 / 1.99) * 3600.0)


def test_期限は絶対時刻で保存され読み直しても伸びない(tmp_path):
    state_path = str(tmp_path / "s.json")
    cw.write_state(state_path, {"deadline_epoch": 1_234_567.0, "fired": False})
    # プロセスが何度上がり直しても、読むのは同じ絶対時刻である。
    for _ in range(3):
        assert cw.read_state(state_path)["deadline_epoch"] == 1_234_567.0


def test_状態の書き込みは途中で壊れた_JSON_を残さない(tmp_path):
    state_path = tmp_path / "s.json"
    cw.write_state(str(state_path), {"a": 1})
    cw.write_state(str(state_path), {"a": 2})
    assert json.loads(state_path.read_text(encoding="utf-8"))["a"] == 2
    assert not (tmp_path / "s.json.tmp").exists()


@pytest.mark.parametrize("bad", [(0.0, 1.99), (18.0, 0.0), (-1.0, 1.99)])
def test_単価や上限が非正なら止まる(bad):
    hard, price = bad
    with pytest.raises(SystemExit):
        cw.compute_deadline(0.0, hard_usd=hard, price_per_hour=price)


# ---------------------------------------------------------------------------
# ② 発火 —— 「送れた」を成功と呼ばない
# ---------------------------------------------------------------------------
def test_一覧から消えて初めて成功と呼ぶ(tmp_path):
    api = FakeApi(alive=["i-target"])
    ok = cw.fire(api, {"instance_id": "i-target", "dry_run": False},
                 str(tmp_path / "log"), max_attempts=1)
    assert ok is True
    assert api.terminate_calls == [["i-target"]]
    assert api.list_calls >= 1  # ★ 確認しないで成功と言っていない


def test_terminate_が返っても消えていなければ再試行する(tmp_path):
    class Stubborn(FakeApi):
        def terminate(self, ids):
            self.terminate_calls.append(list(ids))
            return {}  # 受理されるが消えない

    api = Stubborn(alive=["i-target"])
    ok = cw.fire(api, {"instance_id": "i-target", "dry_run": False},
                 str(tmp_path / "log"), max_attempts=3)
    assert ok is False
    assert len(api.terminate_calls) == 3


def test_API_が落ちていても諦めずに再試行して成功する(tmp_path):
    api = FakeApi(alive=["i-target"], fail_times=2)
    ok = cw.fire(api, {"instance_id": "i-target", "dry_run": False},
                 str(tmp_path / "log"), max_attempts=5)
    assert ok is True
    assert len(api.terminate_calls) == 3  # 2回失敗 → 3回目で成功


def test_dry_run_は_terminate_を1回も呼ばない(tmp_path):
    api = FakeApi(alive=["i-target"])
    assert cw.fire(api, {"instance_id": "i-target", "dry_run": True},
                   str(tmp_path / "log")) is True
    assert api.terminate_calls == []
    assert api.alive == ["i-target"]


def test_殺すのは対象の1台だけ(tmp_path):
    api = FakeApi(alive=["i-target", "i-other"])
    cw.fire(api, {"instance_id": "i-target", "dry_run": False},
            str(tmp_path / "log"), max_attempts=1)
    assert api.terminate_calls == [["i-target"]]
    assert api.alive == ["i-other"]  # ★ 巻き添えにしていない


# ---------------------------------------------------------------------------
# ③ 対象の決定 —— 曖昧なら決めない
# ---------------------------------------------------------------------------
def test_名前が1台に一致すればそれを選ぶ():
    inst = cw.resolve_instance(
        [{"id": "a", "name": "x"}, {"id": "b", "name": "y"}], name="y", instance_id=None
    )
    assert inst["id"] == "b"


@pytest.mark.parametrize(
    "instances,name",
    [
        ([], "x"),                                             # 0 件
        ([{"id": "a", "name": "x"}, {"id": "b", "name": "x"}], "x"),  # 2 件
    ],
)
def test_一意に決まらなければ推測せず止まる(instances, name):
    with pytest.raises(SystemExit):
        cw.resolve_instance(instances, name=name, instance_id=None)


def test_id_を渡せば名前の重複を回避できる():
    instances = [{"id": "a", "name": "x"}, {"id": "b", "name": "x"}]
    assert cw.resolve_instance(instances, name=None, instance_id="b")["id"] == "b"


def test_名前も_id_も無ければ止まる():
    with pytest.raises(SystemExit):
        cw.resolve_instance([{"id": "a", "name": "x"}], name=None, instance_id=None)


# ---------------------------------------------------------------------------
# ④ 設置場所 —— 課金される機械の外では arm させない
# ---------------------------------------------------------------------------
def test_Linux_でなければ_arm_を拒む(monkeypatch):
    monkeypatch.setattr(cw.platform, "system", lambda: "Windows")
    with pytest.raises(SystemExit) as e:
        cw.check_running_on_target({"ip": "203.0.113.9"}, allow_remote=False)
    assert "課金される側" in str(e.value)


def test_allow_remote_を明示すれば通るが警告が残る(monkeypatch):
    monkeypatch.setattr(cw.platform, "system", lambda: "Windows")
    note = cw.check_running_on_target({"ip": "203.0.113.9"}, allow_remote=True)
    assert "⛔" in note  # 状態ファイルに残り、後から事故として読める


def test_自分の_IP_なら自機と確認できる(monkeypatch):
    monkeypatch.setattr(cw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cw, "local_ip_addresses", lambda: {"203.0.113.9"})
    assert cw.check_running_on_target({"ip": "203.0.113.9"}, allow_remote=False).startswith("✅")


def test_NAT_で_IP_が一致しなくても_Linux_上なら止めない(monkeypatch):
    monkeypatch.setattr(cw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cw, "local_ip_addresses", lambda: {"10.0.0.5"})
    assert cw.check_running_on_target({"ip": "203.0.113.9"}, allow_remote=False).startswith("△")


# ---------------------------------------------------------------------------
# HTTP の作法 —— 2026-08-17 に実測した2つの罠を固定する
# ---------------------------------------------------------------------------
def test_User_Agent_を必ず送る():
    """⛔ 既定の Python-urllib は Cloudflare に 403 で弾かれ、キー失効に見える。"""
    captured = {}

    class _Resp:
        def read(self):
            return b'{"data": []}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        captured["auth"] = req.get_header("Authorization")
        captured["url"] = req.full_url
        return _Resp()

    cw.LambdaApi("KEY", opener=opener).list_instances()
    assert captured["ua"] == cw.USER_AGENT
    assert captured["ua"] and "urllib" not in captured["ua"].lower()
    assert captured["auth"].startswith("Basic ")
    assert captured["url"].endswith("/instances")


def test_terminate_は_instance_ids_を本文に入れて_POST_する():
    sent = {}

    class _Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        return _Resp()

    cw.LambdaApi("KEY", opener=opener).terminate(["i-1"])
    assert sent["url"].endswith("/instance-operations/terminate")
    assert sent["body"] == {"instance_ids": ["i-1"]}


# ---------------------------------------------------------------------------
# 生存判定 —— 事故のとき、ログは 13:41Z で途切れていた
# ---------------------------------------------------------------------------
def _status(tmp_path, state) -> int:
    p = tmp_path / "s.json"
    cw.write_state(str(p), state)
    return cw.cmd_status(argparse.Namespace(state=str(p), log=str(tmp_path / "log")))


def test_鼓動が止まっていれば異常と判定する(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "utc_now", lambda: cw.parse_iso("2026-08-16T14:41:00Z"))
    assert _status(tmp_path, {
        "instance_id": "i-1", "instance_name": "n", "deadline_utc": "2026-08-16T21:00:00Z",
        "hard_usd": 18, "fired": False, "interval_sec": 300,
        "last_heartbeat_utc": "2026-08-16T13:41:00Z",   # 事故当日、ログが途切れた時刻
    }) == 1


def test_鼓動が新しければ正常と判定する(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "utc_now", lambda: cw.parse_iso("2026-08-16T13:43:00Z"))
    assert _status(tmp_path, {
        "instance_id": "i-1", "instance_name": "n", "deadline_utc": "2026-08-16T21:00:00Z",
        "hard_usd": 18, "fired": False, "interval_sec": 300,
        "last_heartbeat_utc": "2026-08-16T13:41:00Z",
    }) == 0


def test_一度も鼓動していなければ異常と判定する(tmp_path):
    assert _status(tmp_path, {
        "instance_id": "i-1", "instance_name": "n", "deadline_utc": "2026-08-16T21:00:00Z",
        "hard_usd": 18, "fired": False, "interval_sec": 300, "last_heartbeat_utc": None,
    }) == 1


def test_状態ファイルが無ければ異常と判定する(tmp_path):
    assert cw.cmd_status(argparse.Namespace(
        state=str(tmp_path / "none.json"), log=str(tmp_path / "log"))) == 1


def test_UTC_の_ISO_以外は受け付けない():
    with pytest.raises(ValueError):
        cw.parse_iso("2026-08-16 21:00:00")
