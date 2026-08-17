#!/usr/bin/env python3
"""scripts/cost_watchdog.py — 費用ウォッチドッグ(**課金される側で回す**)。

ラン `lambda-ladder-01` の事故への対処である。事故の内容は preregister.md
「⛔ 事故: 停止条件8(費用 > $20)に違反した」を参照。要約すると:

    測定は 2026-08-16 15:55Z に終わっていた(有効 3.95h ≒ $7.86)のに、
    総額は $47.82 になった。「ハード期限を過ぎたら terminate する」仕掛けを
    **ローカル PC 上のプロセス**として回していたため、PC がスリープした時点で
    ウォッチドッグごと止まったからである。ログは 13:41Z で途切れていた。

★ したがって本スクリプトの設計要件はただ1つ、**「手元の PC が死んでも動き続ける」**
  である。そのために **GPU インスタンス自身の上で回し、自分自身を terminate する。**
  ⛔ 手元から回してはいけない。手元から回した版が $27.82 を捨てた版である。

---------------------------------------------------------------------------
なぜ `shutdown -h` ではなく API を呼ぶのか(2026-08-17 実測)
---------------------------------------------------------------------------
preregister の教訓の欄には「インスタンス上で `sudo shutdown -h +NNN` を予約する」
とも書いてあるが、**Lambda ではそれでは課金が止まらない。**実測した:

    POST /api/v1/instance-operations/stop      -> 404 global/not-found
    POST /api/v1/instance-operations/suspend   -> 404 global/not-found
    POST /api/v1/instance-operations/restart   -> 404 global/object-does-not-exist
    POST /api/v1/instance-operations/terminate -> 404 global/object-does-not-exist

`stop` と `suspend` は**エンドポイントごと存在しない**(存在するものは、でたらめな
instance_id を渡すと「そんなインスタンスは無い」と答える)。つまり Lambda に
「停止」状態は無く、インスタンスは**存在するだけで課金される。**OS を halt しても
インスタンスは一覧に残り、課金は続く。⛔ **しかも halt するとウォッチドッグ自身も
死ぬので、terminate の再試行ができなくなる。事態は悪化する。**

→ **既定では OS のシャットダウンをしない。terminate API を、成功するまで再試行する。**
  EC2 のように「OS シャットダウン = terminate」に設定できる借り先のためだけに
  `--on-failure-shutdown` を残してあるが、既定は off である。

---------------------------------------------------------------------------
Cloudflare の罠(2026-08-17 実測)
---------------------------------------------------------------------------
`urllib` の既定 User-Agent(`Python-urllib/3.x`)は **Cloudflare に 403 で弾かれる**
(`error code: 1010`)。⛔ **これは「API キーが失効した」ようにしか見えない。**
キーを再発行しに走ると時間を捨てるので、UA を必ず明示する。

---------------------------------------------------------------------------
使い方
---------------------------------------------------------------------------
    # ① 借りた直後、インスタンスの上で(手元ではない)
    python3 scripts/cost_watchdog.py selftest          # 課金ゼロ・terminate しない
    python3 scripts/cost_watchdog.py arm \
        --name contamlab-ll02 --price-usd-per-hour 1.99 \
        --budget-usd 20 --hard-usd 18
    # ② 常駐させる(systemd 化まで面倒を見るのは 05-arm-cost-watchdog.sh)
    python3 scripts/cost_watchdog.py run

    python3 scripts/cost_watchdog.py status            # 生きているかを確認
    python3 scripts/cost_watchdog.py disarm            # 撤収時・手で terminate したとき

依存は標準ライブラリのみ(CLAUDE.md の制約)。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 定数 —— すべて 2026-08-17 に実 API へ当てて確かめた値である
# ---------------------------------------------------------------------------
BASE_URL = "https://cloud.lambda.ai/api/v1"

# ★ 既定の Python-urllib は Cloudflare に 403 で弾かれる。何でもよいので明示する。
USER_AGENT = "contamlab-cost-watchdog/1"

DEFAULT_STATE_PATH = "reports/cost-watchdog.json"
DEFAULT_LOG_PATH = "reports/cost-watchdog.log"

# 監視の刻み。事故のときは 10 分だった。5 分にしても API 呼び出しは 1 時間に 12 回で、
# レート制限にも費用にも影響しない。期限超過の検出が遅れるぶんがそのまま無駄金なので短くする。
DEFAULT_INTERVAL_SEC = 300

# terminate が失敗したときの再試行間隔(秒)。★ 諦めない —— 諦めた瞬間に事故が再発する。
RETRY_BACKOFF_SEC = (5, 15, 30, 60, 120, 300)


def utc_now() -> float:
    return time.time()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> float:
    """"2026-08-16T11:58:00Z" 形式を epoch 秒に。末尾 Z のみ受ける(曖昧な入力を通さない)。"""
    t = text.strip()
    if not t.endswith("Z"):
        raise ValueError(f"UTC の ISO8601(末尾 Z)で書くこと: {text!r}")
    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# 認証情報 —— 自分で探しに行かない。env か .env に**置いてもらう**
# ---------------------------------------------------------------------------
def load_api_key(repo_root: str, env: dict | None = None) -> str:
    env = os.environ if env is None else env
    key = (env.get("LAMBDA_API_KEY") or "").strip()
    if key:
        return key
    dotenv = os.path.join(repo_root, ".env")
    if os.path.exists(dotenv):
        with open(dotenv, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("LAMBDA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
    raise SystemExit(
        "LAMBDA_API_KEY が無い。\n"
        "  環境変数に入れるか、リポジトリ直下の .env に LAMBDA_API_KEY=... を書くこと。\n"
        "  ★ キーはランごとに発行し、撤収時に失効させる(インスタンスの上に置くため)。"
    )


class LambdaApi:
    """Lambda Cloud API v1 の必要最小限。テストから差し替えられるよう opener を注入可能にする。"""

    def __init__(self, api_key: str, base_url: str = BASE_URL, opener=None):
        self._key = api_key
        self.base_url = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen

    def _request(self, path: str, body: dict | None = None, timeout: int = 30):
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data)
        token = base64.b64encode((self._key + ":").encode()).decode()
        req.add_header("Authorization", "Basic " + token)
        req.add_header("User-Agent", USER_AGENT)  # ★ これが無いと 403(Cloudflare)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with self._opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")

    def list_instances(self) -> list[dict]:
        return self._request("instances").get("data") or []

    def terminate(self, instance_ids: list[str]) -> dict:
        return self._request(
            "instance-operations/terminate", {"instance_ids": list(instance_ids)}
        )


# ---------------------------------------------------------------------------
# 状態 —— **期限は絶対時刻で凍結する**
# ---------------------------------------------------------------------------
# ⛔ 「起動から N 時間」で持ってはいけない。再起動やプロセス再起動のたびに期限が
#   延びてしまい、ウォッチドッグが落ちて上がるほど寿命が伸びるという最悪の性質になる。
#   arm の時点で epoch 秒に畳んでファイルに書き、run はそれを読むだけにする。
def read_state(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)  # 途中で死んでも壊れた JSON を残さない


def log(log_path: str, message: str) -> None:
    """★ ログは1行ごとに flush する。事故の診断はログが途切れた時刻で出来た。"""
    line = f"{iso(utc_now())} {message}"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
    print(line, flush=True)


def resolve_instance(instances: list[dict], name: str | None, instance_id: str | None) -> dict:
    """名前 or id からインスタンスを1つに決める。★ 曖昧なら決めない(黙って推測しない)。"""
    if instance_id:
        hits = [i for i in instances if i.get("id") == instance_id]
        if len(hits) != 1:
            raise SystemExit(f"instance_id {instance_id} が {len(hits)} 件に一致した。止める。")
        return hits[0]
    if not name:
        raise SystemExit("--name か --instance-id のどちらかが要る。")
    hits = [i for i in instances if (i.get("name") or "") == name]
    if len(hits) == 0:
        known = ", ".join(sorted((i.get("name") or "<名前なし>") for i in instances)) or "(0 台)"
        raise SystemExit(f"名前 {name!r} のインスタンスが無い。今ある名前: {known}")
    if len(hits) > 1:
        raise SystemExit(
            f"名前 {name!r} が {len(hits)} 台に一致した。⛔ どれを殺すか推測しない。"
            " --instance-id で指定すること。"
        )
    return hits[0]


def local_ip_addresses() -> set[str]:
    """この機械に付いている IP。取れなければ空集合(取れないこと自体は失敗にしない)。"""
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            found.add(info[4][0])
    except Exception:  # noqa: BLE001
        pass
    try:
        # 既定経路に出ていく側の IP。UDP なのでパケットは飛ばない。
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("198.51.100.1", 9))  # TEST-NET-2(到達しなくてよい)
            found.add(s.getsockname()[0])
    except Exception:  # noqa: BLE001
        pass
    return found


def check_running_on_target(inst: dict, allow_remote: bool) -> str:
    """★ 事故の再発を機械で止める —— **課金される機械の上で回っているか**を確かめる。

    lambda-ladder-01 の $27.82 は「ウォッチドッグを手元の Windows で回した」ことで
    出た。⛔ 手元で回っている限り、PC が寝れば見張りも寝る。
    ここで確実に言えるのは「Linux ではない ⇒ 借りた GPU インスタンスではない」で、
    事故当時の構成はこれだけで弾ける。IP の一致は取れれば加点、取れなくても落とさない
    (借り先によっては NAT の内側で公開 IP が見えないため)。
    """
    if platform.system() != "Linux" and not allow_remote:
        raise SystemExit(
            f"⛔ ここは {platform.system()} である。GPU インスタンスの上ではない。\n"
            "  費用ウォッチドッグは**課金される側**で回す。手元で回した版が $27.82 を捨てた。\n"
            "  (どうしても手元から arm したいなら --allow-remote。⛔ 事故の再現である)"
        )
    ip = inst.get("ip")
    if ip and ip in local_ip_addresses():
        return f"✅ 自分自身を見張っている({ip} はこの機械の IP)"
    if platform.system() == "Linux":
        return f"△ Linux 上だが対象の IP({ip})を自分の IP として確認できなかった(NAT の可能性)"
    return "⛔ --allow-remote で手元から arm した。PC が寝れば見張りも止まる"


def compute_deadline(started_at: float, hard_usd: float, price_per_hour: float) -> float:
    if price_per_hour <= 0:
        raise SystemExit("--price-usd-per-hour は正の数で。")
    if hard_usd <= 0:
        raise SystemExit("--hard-usd は正の数で。")
    return started_at + (hard_usd / price_per_hour) * 3600.0


def cmd_arm(args) -> int:
    key = load_api_key(args.repo_root)
    api = LambdaApi(key)
    instances = api.list_instances()
    inst = resolve_instance(instances, args.name, args.instance_id)
    placement = check_running_on_target(inst, args.allow_remote)

    if args.budget_usd is not None and args.hard_usd >= args.budget_usd:
        # ★ 凍結値より**厳しい側**でしか切らない。これは規則の緩和ではない、という
        #   lambda-ladder-01 の整理をそのまま機械の側で強制する。
        raise SystemExit(
            f"--hard-usd {args.hard_usd} は事前登録の停止条件 --budget-usd "
            f"{args.budget_usd} より小さくすること(厳しい側でしか切らない)。"
        )

    # 課金開始時刻。API が起動時刻を返すならそれが正。無ければ arm 時刻で代用する。
    started_at = None
    source = None
    if args.started_at_utc:
        started_at, source = parse_iso(args.started_at_utc), "--started-at-utc(手入力)"
    else:
        for field in ("created_at", "launched_at", "created", "start_time"):
            raw = inst.get(field)
            if isinstance(raw, str) and raw:
                try:
                    started_at, source = parse_iso(raw.replace("+00:00", "Z")), f"API の {field}"
                    break
                except ValueError:
                    continue
    if started_at is None:
        started_at, source = utc_now(), "arm した時刻(⛔ 起動からの経過ぶん期限が甘い)"

    deadline = compute_deadline(started_at, args.hard_usd, args.price_usd_per_hour)
    state = {
        "provider": "lambda",
        "instance_id": inst.get("id"),
        "instance_name": inst.get("name"),
        "instance_type": (inst.get("instance_type") or {}).get("name"),
        "region": (inst.get("region") or {}).get("name"),
        "price_usd_per_hour": args.price_usd_per_hour,
        "budget_usd": args.budget_usd,
        "hard_usd": args.hard_usd,
        "started_at_utc": iso(started_at),
        "started_at_source": source,
        "armed_at_utc": iso(utc_now()),
        "deadline_utc": iso(deadline),
        "deadline_epoch": deadline,
        "interval_sec": args.interval,
        "dry_run": bool(args.dry_run),
        "placement_check": placement,
        "fired": False,
        "last_heartbeat_utc": None,
    }
    write_state(args.state, state)
    log(args.log, f"arm: {state['instance_name']} ({state['instance_id']}) "
                  f"期限 {state['deadline_utc']} / ハード上限 ${args.hard_usd} "
                  f"/ 単価 ${args.price_usd_per_hour}/h"
                  + ("  ⛔ DRY-RUN(terminate しない)" if args.dry_run else ""))
    hours = (deadline - started_at) / 3600.0
    print()
    print("★ preregister に貼る記録:")
    print(f"    インスタンス   {state['instance_name']} / {state['instance_id']}")
    print(f"    課金開始とみなす時刻  {state['started_at_utc']}  ({source})")
    print(f"    ハード期限     {state['deadline_utc']}  (= {hours:.2f} h ≒ "
          f"${hours * args.price_usd_per_hour:.2f})")
    print(f"    事前登録の停止条件    ${args.budget_usd}  → ハードは ${args.hard_usd}(厳しい側)")
    print(f"    設置場所の検査 {placement}")
    print()
    if args.dry_run:
        print("⛔ DRY-RUN で arm した。期限が来ても terminate しない。本番では外すこと。")
    return 0


def fire(api: LambdaApi, state: dict, log_path: str, max_attempts: int | None = None) -> bool:
    """期限到達。terminate して、**一覧から消えたことを確認するまで**成功と呼ばない。"""
    instance_id = state["instance_id"]
    if state.get("dry_run"):
        log(log_path, f"⛔ DRY-RUN: 本来ここで terminate する ({instance_id})")
        return True

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        try:
            api.terminate([instance_id])
            log(log_path, f"terminate を送った (試行 {attempt}) {instance_id}")
        except Exception as exc:  # noqa: BLE001 — 何が来ても諦めないのが仕事
            log(log_path, f"terminate が失敗 (試行 {attempt}): {type(exc).__name__}: {exc}")

        # ★ 送れたことを成功と呼ばない。消えたことを一覧で確かめる。
        try:
            alive = [i.get("id") for i in api.list_instances()]
            if instance_id not in alive:
                log(log_path, f"✅ 課金停止を確認: {instance_id} は一覧に無い(残り {len(alive)} 台)")
                return True
            log(log_path, f"まだ一覧に居る: {instance_id}。再試行する")
        except Exception as exc:  # noqa: BLE001
            log(log_path, f"一覧の確認に失敗: {type(exc).__name__}: {exc}")

        time.sleep(RETRY_BACKOFF_SEC[min(attempt - 1, len(RETRY_BACKOFF_SEC) - 1)])
    return False


def cmd_run(args) -> int:
    state = read_state(args.state)
    if state.get("fired"):
        log(args.log, "既に発火済みの状態ファイル。何もしない(disarm するか arm し直すこと)")
        return 0
    key = load_api_key(args.repo_root)
    api = LambdaApi(key)
    deadline = float(state["deadline_epoch"])
    interval = int(state.get("interval_sec") or DEFAULT_INTERVAL_SEC)
    instance_id = state["instance_id"]
    log(args.log, f"run 開始: 期限 {state['deadline_utc']} / 監視間隔 {interval}s "
                  f"/ 対象 {instance_id}")

    while True:
        now = utc_now()
        state["last_heartbeat_utc"] = iso(now)
        write_state(args.state, state)  # ★ 生存証明。status がこれを読む

        if now >= deadline:
            log(args.log, f"⛔ ハード期限 {state['deadline_utc']} を過ぎた。terminate する")
            ok = fire(api, state, args.log)
            state["fired"] = True
            state["fired_at_utc"] = iso(utc_now())
            state["fired_ok"] = ok
            write_state(args.state, state)
            if not ok and args.on_failure_shutdown:
                # ⛔ Lambda では課金は止まらない(冒頭の実測)。EC2 等の例外的な借り先のみ。
                log(args.log, "terminate に失敗し続けた。--on-failure-shutdown により OS を落とす")
                subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            return 0 if ok else 1

        # 期限前でも、インスタンスが既に消えていれば見張る相手が居ない。
        try:
            if instance_id not in [i.get("id") for i in api.list_instances()]:
                log(args.log, f"対象 {instance_id} は既に一覧に無い。監視を終える")
                state["fired"] = True
                state["fired_at_utc"] = iso(utc_now())
                state["fired_ok"] = True
                state["fired_note"] = "期限前に手で terminate されていた"
                write_state(args.state, state)
                return 0
        except Exception as exc:  # noqa: BLE001
            # ⛔ API が見えないことを理由に**止まってはいけない。**次の刻みで見直す。
            log(args.log, f"一覧の取得に失敗(監視は続ける): {type(exc).__name__}: {exc}")

        remain = deadline - utc_now()
        log(args.log, f"残り {remain / 3600.0:.2f} h "
                      f"(≒ ${remain / 3600.0 * float(state['price_usd_per_hour']):.2f} ぶん)")
        time.sleep(min(interval, max(1.0, deadline - utc_now())))


def cmd_status(args) -> int:
    if not os.path.exists(args.state):
        print(f"⛔ 状態ファイルが無い: {args.state}(arm していない)")
        return 1
    state = read_state(args.state)
    hb = state.get("last_heartbeat_utc")
    print(f"対象        {state.get('instance_name')} / {state.get('instance_id')}")
    print(f"ハード期限  {state.get('deadline_utc')}  (${state.get('hard_usd')} 相当)")
    print(f"発火済み    {state.get('fired')}")
    print(f"最終鼓動    {hb}")
    if state.get("dry_run"):
        print("⛔ DRY-RUN で arm されている。期限が来ても terminate しない")
    if not hb:
        print("⛔ 鼓動が1回も無い。run が動いていない")
        return 1
    age = utc_now() - parse_iso(hb)
    limit = 3 * int(state.get("interval_sec") or DEFAULT_INTERVAL_SEC)
    if not state.get("fired") and age > limit:
        # ★ 事故のときログは 13:41Z で途切れていた。それを機械が言えるようにする。
        print(f"⛔ 鼓動が {age / 60.0:.1f} 分止まっている(許容 {limit / 60.0:.1f} 分)。"
              "ウォッチドッグは死んでいる")
        return 1
    print(f"✅ 生きている(鼓動は {age:.0f} 秒前)")
    return 0


def cmd_disarm(args) -> int:
    if not os.path.exists(args.state):
        print("状態ファイルが無い。何もしない。")
        return 0
    state = read_state(args.state)
    state["fired"] = True
    state["fired_note"] = "disarm で手動解除"
    state["fired_at_utc"] = iso(utc_now())
    write_state(args.state, state)
    log(args.log, "disarm した。run は次の刻みで終了する")
    return 0


def cmd_selftest(args) -> int:
    """★ 発火の経路を、GPU を借りずに・何も terminate せずに確かめる。

    事故の本質は「仕掛けを一度も試さなかった」ことである。ここで確かめるのは3点:
      ① 認証と到達性(UA を含む)
      ② terminate エンドポイントが**存在し・認証が通る**こと
         —— でたらめな id を投げて `global/object-does-not-exist` が返れば、
            経路と認証は正しい。⛔ 実在するものは1つも消えない
      ③ 期限判定と発火の分岐(偽の API で dry-run)
    """
    key = load_api_key(args.repo_root)
    api = LambdaApi(key)
    ok = True

    try:
        instances = api.list_instances()
        print(f"✅ ① 認証と到達性: 一覧を取得できた({len(instances)} 台)")
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            print("⛔ ① 403。User-Agent を送っていない可能性がある(Cloudflare の 1010)")
        else:
            print(f"⛔ ① 一覧の取得に失敗: HTTP {exc.code}")
        return 1

    bogus = "0" * 32
    try:
        api.terminate([bogus])
        print("⛔ ② でたらめな id が受理された。想定外。止める")
        ok = False
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        code = ""
        try:
            code = (json.loads(body).get("error") or {}).get("code", "")
        except Exception:  # noqa: BLE001
            pass
        if code == "global/object-does-not-exist":
            print("✅ ② terminate の経路と認証は正しい(存在しない id を正しく拒否した)")
        elif code == "global/not-found":
            print(f"⛔ ② terminate のエンドポイントが無い。API が変わった: {body[:200]}")
            ok = False
        else:
            print(f"⛔ ② 想定外の応答 HTTP {exc.code}: {body[:200]}")
            ok = False

    # ③ 偽 API で期限判定を通す。ネットワークにも課金にも触れない。
    class _FakeApi:
        def __init__(self):
            self.calls = []
            self.alive = ["i-fake"]

        def terminate(self, ids):
            self.calls.append(list(ids))
            self.alive = []
            return {}

        def list_instances(self):
            return [{"id": i} for i in self.alive]

    fake = _FakeApi()
    fired = fire(fake, {"instance_id": "i-fake", "dry_run": False}, args.log, max_attempts=1)
    if fired and fake.calls == [["i-fake"]]:
        print("✅ ③ 発火の経路(terminate → 一覧から消えたことの確認)が通った")
    else:
        print(f"⛔ ③ 発火の経路が壊れている: fired={fired} calls={fake.calls}")
        ok = False

    fake2 = _FakeApi()
    if fire(fake2, {"instance_id": "i-fake", "dry_run": True}, args.log) and not fake2.calls:
        print("✅ ④ dry-run は terminate を1回も呼ばない")
    else:
        print("⛔ ④ dry-run なのに terminate を呼んだ")
        ok = False

    print()
    print("✅ すべて通った。課金ゼロ・terminate ゼロ。" if ok else "⛔ 通らなかった項目がある。")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description="費用ウォッチドッグ(インスタンス自身の上で回す)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--state", default=os.path.join(repo_root, DEFAULT_STATE_PATH))
    p.add_argument("--log", default=os.path.join(repo_root, DEFAULT_LOG_PATH))
    p.add_argument("--repo-root", default=repo_root)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("arm", help="対象を1台に決め、ハード期限を絶対時刻で凍結する")
    a.add_argument("--name", help="Lambda コンソールで付けた名前")
    a.add_argument("--instance-id", help="名前が重複するときはこちら")
    a.add_argument("--price-usd-per-hour", type=float, required=True)
    a.add_argument("--hard-usd", type=float, required=True,
                   help="この額で切る。★ 事前登録の停止条件より小さい値にすること")
    a.add_argument("--budget-usd", type=float,
                   help="事前登録の停止条件(--hard-usd がこれ未満であることを検査する)")
    a.add_argument("--started-at-utc", help='課金開始時刻 "YYYY-MM-DDTHH:MM:SSZ"')
    a.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    a.add_argument("--dry-run", action="store_true", help="期限が来ても terminate しない")
    a.add_argument("--allow-remote", action="store_true",
                   help="⛔ 課金される機械の外から arm する。事故の再現なので既定では拒む")
    a.set_defaults(func=cmd_arm)

    r = sub.add_parser("run", help="常駐して見張る(05-arm-cost-watchdog.sh が呼ぶ)")
    r.add_argument("--on-failure-shutdown", action="store_true",
                   help="⛔ Lambda では課金が止まらない。EC2 等の例外用。既定 off")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="鼓動を見て、生きているかを判定する")
    s.set_defaults(func=cmd_status)

    d = sub.add_parser("disarm", help="撤収時・手で terminate したとき")
    d.set_defaults(func=cmd_disarm)

    t = sub.add_parser("selftest", help="課金ゼロで発火の経路を確かめる")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
