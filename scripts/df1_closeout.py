#!/usr/bin/env python3
"""scripts/df1_closeout.py — ラン detector-firstlight-01 の**撤収**を機械にやらせる。

    python scripts/df1_closeout.py status              # いまどの段にいるか(課金に触らない)
    python scripts/df1_closeout.py collect             # 成果物を手元へ回収する(同上)
    python scripts/df1_closeout.py close --terminate            # 回収 -> disarm -> terminate
    python scripts/df1_closeout.py close --terminate --shutdown # さらに PC を落とす

★ なぜ道具にするのか —— 撤収は**離席中に、疲れた頭で、取り返しのつかない順序で**行う。
  2026-08-17 の $47.82 は「止める仕掛けを手元に置いた」ことが原因だったが、
  撤収そのものにも順序の穴がある:

      ★ 回収より先に terminate すると reports/ と応答キャッシュが**永久に消える**
         (どちらも .gitignore 対象で、ホスト上のファイルが原本である)。
      ★ 異常終了なのに PC を落とすと、ユーザーは「無事に終わった」と誤読する
         (ユーザーとの取り決め: **落ちていないこと自体が異常の合図**)。

  -> **順序と、異常時の分岐を、人の判断から外す。**

---------------------------------------------------------------------------
★ 正常終了の定義(この 3 つが全部そろったときだけ「正常」と呼ぶ)
---------------------------------------------------------------------------
  1. ホストに reports/df1-STOPPED.txt が**無い**(停止条件に当たらなかった)
  2. オーケストレータのログに `[DONE]` の行が**ある**
  3. reports/detector-firstlight-01.<tag>.json が**在り、2 アーム分の判定を含む**

  ★ 1 つでも欠けたら **異常**として扱う:
      - PC は落とさない(★ --shutdown を渡しても落とさない)
      - GPU も落とさない(★ --terminate を渡しても落とさない。models/ には
        1 本 1 時間かけた LoRA が入っており、terminate は再学習を意味する。
        オーケストレータは冪等なので、診断してから続きを再開できる)
        -> どうしても落とすなら --force-terminate を明示的に渡す。
      ★ 課金の底は**インスタンス上のウォッチドッグ**が持っている。手元が死んでも
        ハード期限で自分自身を terminate する。撤収の失敗は破産にはつながらない。

★ この道具は preregister の判定を1つも読まない。**測定結果の中身で分岐しない。**
  分岐するのは「終わったか / 途中で止まったか」だけである。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_watchdog import LambdaApi, load_api_key  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUN = "detector-firstlight-01"
REMOTE_DIR = "contamlab"
LOCAL_DEST = REPO / "reports" / "from-instance"   # ★ reports/ は .gitignore 対象
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=accept-new"]

# ホストを1回だけ叩いて、撤収の判断に要るものを全部持ってくる問い合わせ。
# ★ 読むだけ。⛔ 走っているスクリプトには一切触らない。
PROBE = r"""
cd ~/contamlab 2>/dev/null || { echo "NOREPO"; exit 0; }
echo "TAG=$(cat reports/env-tag 2>/dev/null)"
echo "STOPPED=$([ -f reports/df1-STOPPED.txt ] && echo yes || echo no)"
echo "DONE=$(grep -c '^=== \[DONE\]' reports/df1-orchestrate.log 2>/dev/null || echo 0)"
echo "ALIVE=$(pgrep -c -f 'df1-orchestrat[e].sh' 2>/dev/null || echo 0)"
echo "TRAIN=$(pgrep -c -f 'train_lor[a].py' 2>/dev/null || echo 0)"
echo "OUT=$([ -f reports/detector-firstlight-01.$(cat reports/env-tag).json ] && echo yes || echo no)"
echo "STAGE=$(grep '^=== ' reports/df1-orchestrate.log 2>/dev/null | tail -1)"
echo "MODELS=$(ls -1 models 2>/dev/null | tr '\n' ' ')"
echo "---STOP---"
cat reports/df1-STOPPED.txt 2>/dev/null
echo "---WATCHDOG---"
python3 scripts/cost_watchdog.py status 2>&1 | tail -12
"""


def ssh(args, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-i", args.key, f"{args.user}@{args.host}", cmd],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout)


def remote_state(args) -> dict:
    p = ssh(args, PROBE, timeout=90)
    if p.returncode != 0:
        raise SystemExit(f"★ ホストに繋がらない(rc={p.returncode}): {p.stderr.strip()[:400]}")
    head, _, rest = p.stdout.partition("---STOP---")
    stop_txt, _, watchdog = rest.partition("---WATCHDOG---")
    st = {"raw": p.stdout, "stopped_text": stop_txt.strip(), "watchdog": watchdog.strip()}
    for line in head.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            st[k.strip().lower()] = v.strip()
    return st


def verdict(st: dict) -> tuple[bool, list[str]]:
    """★ 正常/異常の判定。**測定結果の中身は一切見ない。**"""
    reasons = []
    if st.get("stopped") != "no":
        reasons.append("停止条件に当たっている(reports/df1-STOPPED.txt が在る)")
    if st.get("done", "0") == "0":
        reasons.append("ログに [DONE] が無い(まだ終わっていない、または途中で死んだ)")
    if st.get("out") != "yes":
        reasons.append("検出器の出力 JSON がホストに無い")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# 回収 —— ★ terminate より**先に**、必ず成功させる
# ---------------------------------------------------------------------------
def collect(args) -> tuple[bool, list[str]]:
    LOCAL_DEST.mkdir(parents=True, exist_ok=True)
    ok, problems = True, []
    jobs = [
        (f"{REMOTE_DIR}/reports", "reports(判定・ログ・環境記録)"),
        (f"{REMOTE_DIR}/data/cache", "応答キャッシュ(★ 再現に要る原本)"),
        (f"{REMOTE_DIR}/data/injection", "注入 id と manifest"),
    ]
    for remote, label in jobs:
        print(f"  回収: {label}")
        p = subprocess.run(
            ["scp", "-r", *SSH_OPTS, "-i", args.key,
             f"{args.user}@{args.host}:{remote}", str(LOCAL_DEST)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=args.scp_timeout)
        if p.returncode != 0:
            ok = False
            problems.append(f"{label}: scp が失敗 ({p.stderr.strip()[:200]})")
    # ★ models/ は数十 GB あるので回収しない(GGUF の sha256 は reports/ に載っている)。
    return ok, problems


def verify_local(tag: str) -> tuple[bool, list[str]]:
    """★ 落とす前に「手元で読める」ことまで確かめる。scp の成功は中身の保証ではない。"""
    problems = []
    out = LOCAL_DEST / "reports" / f"{RUN}.{tag}.json"
    if not out.is_file():
        return False, [f"{out} が手元に無い"]
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except Exception as exc:                      # noqa: BLE001
        return False, [f"{out} が JSON として読めない: {exc}"]
    models = data.get("models") or []
    if len(models) != 2:
        problems.append(f"アームが {len(models)} 本(2 本のはず)")
    for m in models:
        for key in ("drop", "p_holm", "adjusted_lcb", "detected"):
            if key not in m:
                problems.append(f"{m.get('name')} に {key} が無い")
    if not list((LOCAL_DEST / "cache").glob(f"responses.{tag}.jsonl")):
        problems.append("応答キャッシュが手元に無い(★ 再現できなくなる)")
    return (not problems), problems


# ---------------------------------------------------------------------------
# 撤収 —— disarm してから terminate。★ id が一覧から消えるまで見届ける
# ---------------------------------------------------------------------------
def terminate(args) -> bool:
    print("\n★ ウォッチドッグを解除する(手で落とすので、二重に撃たせない)")
    p = ssh(args, "cd ~/contamlab && python3 scripts/cost_watchdog.py disarm", timeout=60)
    print("   " + (p.stdout.strip() or p.stderr.strip())[:300])

    api = LambdaApi(load_api_key(str(REPO)))
    before = api.list_instances()
    hit = [i for i in before if i.get("id") == args.instance_id]
    if not hit:
        print(f"★ instance {args.instance_id} は既に一覧に無い。課金は止まっている。")
        return True
    print(f"★ terminate を撃つ: {args.instance_id} "
          f"(name={hit[0].get('name')!r} ip={hit[0].get('ip')})")
    api.terminate([args.instance_id])

    # ★ 停止の確認は **id が一覧から消えるまで**。terminate 直後は status=terminating の
    #   まま ip だけが null になる段階があり、ip で読むと「消えた」と誤読する。
    for i in range(args.terminate_poll_tries):
        time.sleep(args.terminate_poll_sec)
        ids = {x.get("id") for x in api.list_instances()}
        if args.instance_id not in ids:
            print(f"  確認: 一覧から id が消えた({(i + 1) * args.terminate_poll_sec}s)。"
                  "★ 課金停止。")
            return True
        print(f"  まだ一覧に居る... ({(i + 1) * args.terminate_poll_sec}s)")
    print("★ 期限内に消えなかった。★ **PC を落とさない。**コンソールで確かめること。")
    return False


POWER_STATE = REPO / "reports" / "power-restore.json"


def restore_power() -> None:
    """ラン中だけ切っておいた自動スリープを元に戻す。

    ★ 手元が寝るとランが止まる(2026-08-17 の $47.82)ので、ラン中は切ってある。
      ⛔ 戻し忘れると、この PC は二度と勝手に寝ない。**借りた環境は返す。**
    """
    if os.name != "nt":
        print("  Windows ではないので電源設定はいじらない")
        return
    if not POWER_STATE.is_file():
        print(f"  ▲ {POWER_STATE} が無い。戻す値が判らないので触らない")
        return
    d = json.loads(POWER_STATE.read_text(encoding="utf-8"))
    for flag, key in (("standby-timeout-ac", "standby_timeout_ac_min"),
                      ("hibernate-timeout-ac", "hibernate_timeout_ac_min")):
        val = d.get(key)
        if val is None:
            continue
        subprocess.run(["powercfg", "/change", flag, str(val)], check=False)
        print(f"  電源設定を戻した: {flag} = {val} 分")


def shutdown_pc(delay: int) -> None:
    print(f"\n★ PC を {delay} 秒後にシャットダウンする(取り消しは shutdown /a)")
    subprocess.run(["shutdown", "/s", "/t", str(delay),
                    "/c", "contamlab detector-firstlight-01 の撤収完了"], check=False)


# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    st = remote_state(args)
    ok, reasons = verdict(st)
    print(f"tag       : {st.get('tag')}")
    print(f"段        : {st.get('stage')}")
    print(f"生存      : orchestrator={st.get('alive')} train={st.get('train')}")
    print(f"models    : {st.get('models')}")
    print(f"判定      : {'★ 正常終了' if ok else '未完 / 異常 ―― ' + ' / '.join(reasons)}")
    if st["stopped_text"]:
        print("STOPPED:\n  " + st["stopped_text"].replace("\n", "\n  "))
    print("ウォッチドッグ:\n  " + st["watchdog"].replace("\n", "\n  "))
    return 0 if ok else 1


def cmd_collect(args) -> int:
    st = remote_state(args)
    ok, problems = collect(args)
    print(f"  -> {LOCAL_DEST}")
    for p in problems:
        print(f"  ★ {p}")
    if not ok:
        return 1
    vok, vproblems = verify_local(st.get("tag") or args.tag)
    for p in vproblems:
        print(f"  ▲ {p}")
    print("  ★ 手元で読めることまで確認した" if vok else
          "  ▲ 回収はできたが検出器の出力は揃っていない(まだ終わっていないなら当然)")
    return 0


def cmd_close(args) -> int:
    st = remote_state(args)
    ok, reasons = verdict(st)
    print(f"tag {st.get('tag')} / 段 {st.get('stage')}")
    if not ok:
        print("★ **正常終了ではない**:")
        for r in reasons:
            print(f"   - {r}")
        if st["stopped_text"]:
            print("   STOPPED: " + st["stopped_text"].replace("\n", " | "))

    print("\n[1/4] 成果物の回収(★ terminate より先に必ず通す)")
    cok, problems = collect(args)
    for p in problems:
        print(f"  ★ {p}")
    vok, vproblems = verify_local(st.get("tag") or args.tag)
    for p in vproblems:
        print(f"  ▲ {p}")

    print("\n[2/4] 撤収の可否")
    may_terminate = args.terminate and ok and cok and vok
    if args.force_terminate:
        may_terminate = args.terminate and cok      # ★ 回収の失敗だけは覆さない
        print("  ★ --force-terminate: 異常終了でも落とす(人が明示した)")
    if not args.terminate:
        print("  terminate しない(--terminate が無い)")
    elif not may_terminate:
        print("  ★ terminate しない ―― 正常終了でないか、回収に失敗している。")
        print("     ★ models/ には 1 本 1 時間の LoRA が入っている。落とせば再学習である。")
        print("     ★ 課金の底はインスタンス上のウォッチドッグが持っている(ハード期限)。")

    print("\n[3/4] GPU")
    killed = terminate(args) if may_terminate else False

    print("\n[4/4] PC")
    # ★ 電源設定は**正常でも異常でも**戻す。見張りが終われば寝てよい。
    restore_power()
    if args.shutdown and ok and cok and vok and killed:
        shutdown_pc(args.shutdown_delay)
    elif args.shutdown:
        print("  ★ **落とさない。**正常終了・回収・停止確認のどれかが通っていない。")
        print("     ★ 取り決め: **PC が落ちていないこと自体が異常の合図**である。")
    else:
        print("  落とさない(--shutdown が無い)")
    return 0 if (ok and cok and vok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("DF1_HOST", "129.146.104.106"))
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "contamlab-pc06"))
    ap.add_argument("--instance-id", default="f9a16b8b47a14244b51cd4b953741d29")
    ap.add_argument("--tag", default="lambda-a100-df1-20260820")
    ap.add_argument("--scp-timeout", type=int, default=1800)
    ap.add_argument("--terminate-poll-sec", type=int, default=15)
    ap.add_argument("--terminate-poll-tries", type=int, default=20)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("status", cmd_status), ("collect", cmd_collect)):
        sub.add_parser(name).set_defaults(fn=fn)
    c = sub.add_parser("close")
    c.add_argument("--terminate", action="store_true", help="GPU を落とす")
    c.add_argument("--force-terminate", action="store_true",
                   help="異常終了でも GPU を落とす(再学習になる。人が明示するときだけ)")
    c.add_argument("--shutdown", action="store_true", help="最後に PC を落とす")
    c.add_argument("--shutdown-delay", type=int, default=60)
    c.set_defaults(fn=cmd_close)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
