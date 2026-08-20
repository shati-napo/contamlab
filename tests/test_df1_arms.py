"""tests/test_df1_arms.py — ラン detector-firstlight-01 の凍結表と副次の読みの検査。

★ 本ランは **12 ラン通じて初めて検出器を実モデルに当てる。**動かす軸は
  「検出器に通すこと」だけであり、学習の設定は ll-01 から1つも変えない。

  ここで押さえるのは5点:
    ① アームが **ll-01 と必ず別名**である(同名なら応答キャッシュが ll-01 の答えを
       返し、作り直したモデルが1度も呼ばれずに終わる)
    ② **λ は 0.8 の1段だけ**(⛔ 停止条件 7。L0 も L2〜L4 も渡せない)
    ③ **レシピは R1 だけ**(⛔ 事前登録の外に出る口を作らない)
    ④ ⛔ **`70-positive-control.sh` を書き換えていない**(6アーム / α=0.008333 のまま)
       ——`72` の差分がロースターと α の2点に限られている
    ⑤ 副次の読み(注入済み / 非注入で分けた drop)が**推論を増やさずに**動く

  ⛔ GPU にもネットワークにも出ない。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "finetune"))

train_lora = importlib.import_module("train_lora")
scale_adapter = importlib.import_module("scale_adapter")

RUN = "detector-firstlight-01"


@pytest.fixture(autouse=True)
def _書式をプロセスに漏らさない():
    """★ 出力書式はプロセス全体のグローバル状態である(runner.set_prompt_format)。

    本ファイルは書式 C で応答を組み立てるが、**他のテストの前提を壊してはいけない。**
    ⛔ グローバルを触ったら必ず戻す。
    """
    from contamlab.runner import current_prompt_format, set_prompt_format

    before = current_prompt_format()
    yield
    set_prompt_format(before)


# ---------------------------------------------------------------------------
# ① アーム名 —— ll-01 と必ず分かれる
# ---------------------------------------------------------------------------
def test_学習アームは複製3本で名前が分かれる():
    arms = train_lora.DF1_TRAIN_ARMS
    assert arms == {1: "df1t1-x40", 2: "df1t2-x40", 3: "df1t3-x40"}
    assert len(set(arms.values())) == 3


def test_λアームは3本で学習アームと衝突しない():
    scaled = set(scale_adapter.DF1_LAMBDA_ARMS.values())
    train = set(train_lora.DF1_TRAIN_ARMS.values())
    assert scaled == {"df1L08t1-x40", "df1L08t2-x40", "df1L08t3-x40"}
    assert not (scaled & train)


def test_ll01のアームと1本も衝突しない():
    """⛔ 同名なら応答キャッシュが ll-01 の答えを返し、何も測れない。"""
    df1 = set(train_lora.DF1_TRAIN_ARMS.values()) | set(
        scale_adapter.DF1_LAMBDA_ARMS.values())
    ll1 = set(train_lora.LL01_TRAIN_ARMS.values()) | set(
        scale_adapter.LL01_LAMBDA_ARMS.values())
    assert not (df1 & ll1)


def test_アーム名の末尾2桁は40のまま():
    """器は `arm[-2:]` を注入率として読む。本ランは 40% の1水準しか持たない。"""
    for arm in (list(train_lora.DF1_TRAIN_ARMS.values())
                + list(scale_adapter.DF1_LAMBDA_ARMS.values())):
        assert arm[-2:] == "40"


# ---------------------------------------------------------------------------
# ② λ は 0.8 の1段だけ
# ---------------------------------------------------------------------------
def test_λの段はL1の1つだけ():
    """⛔ 停止条件 7。λ を動かせば「検出器に通す」以外の軸が増える。"""
    assert scale_adapter.DF1_STEPS == ("L1",)
    assert scale_adapter.RUNS[RUN]["steps"] == ("L1",)
    assert scale_adapter.LAMBDA_STEPS["L1"] == 0.8


@pytest.mark.parametrize("step", ["L0", "L2", "L3", "L4"])
def test_L1以外の段は本ランに無い(step):
    assert step not in scale_adapter.RUNS[RUN]["steps"]


def test_LAMBDA_STEPSは1文字も変えていない():
    """pc-06 が凍結した表。本ランは引くだけである。"""
    assert scale_adapter.LAMBDA_STEPS == {
        "L0": 1.0, "L1": 0.8, "L2": 0.6, "L3": 0.4, "L4": 0.2}


# ---------------------------------------------------------------------------
# ③ レシピ R1 / ベース / 注入トークンは ll-01 と同一
# ---------------------------------------------------------------------------
def test_レシピはR1だけ():
    assert train_lora.DF1_RECIPE == "R1" == train_lora.LL01_RECIPE
    assert scale_adapter.RUNS[RUN]["recipe"] == "R1"


def test_ベースと注入トークンはll01と同一():
    assert train_lora.RUN_BASES[RUN] == train_lora.RUN_BASES["lambda-ladder-01"]
    assert train_lora.INJECTED_TOKENS_ONCE_BY_RUN[RUN] == 238_082


def test_replicateを取るランの表に入っている():
    assert train_lora.REPLICATE_TRAIN_ARMS[RUN] is train_lora.DF1_TRAIN_ARMS


def test_seedは複製ごとに変わらない():
    """複製は「同一条件の再現ばらつき」であって「別の学習3本」ではない。"""
    assert train_lora.SEED == 20260809


def test_アーム対応表は凍結表から引ける():
    assert [scale_adapter.source_arm(RUN, n) for n in (1, 2, 3)] == [
        "df1t1-x40", "df1t2-x40", "df1t3-x40"]
    assert [scale_adapter.target_arm(RUN, "L1", n) for n in (1, 2, 3)] == [
        "df1L08t1-x40", "df1L08t2-x40", "df1L08t3-x40"]


def test_既存ランのアーム対応を壊していない():
    """⛔ 本ランの実装で ll-01 / pc-06 / cc-01 の経路が動いてはいけない。"""
    assert scale_adapter.source_arm("lambda-ladder-01", 2) == "ll1t2-x40"
    assert scale_adapter.target_arm("lambda-ladder-01", "L2", 2) == "ll1L06t2-x40"
    assert scale_adapter.source_arm("positive-control-06") == "pc6r1-x40"
    assert scale_adapter.target_arm("positive-control-06", "L0") == "pc6L10-x40"
    assert scale_adapter.target_arm(
        scale_adapter.CC01_RUN, "L1", 1, "20") == "cc1L08t1-x20"


def test_注入集合の複製表は6本で複製元はpc_x40():
    prepare = importlib.import_module("prepare_df1_arms")
    arms = prepare.df01_arms()
    assert len(arms) == prepare.EXPECTED_N_ARMS == 6
    assert set(arms.values()) == {"pc-x40"}
    assert prepare.EXPECTED_N_INJECTED == 1896


# ---------------------------------------------------------------------------
# ④ 70 を書き換えていない / 72 の差分は2点だけ
# ---------------------------------------------------------------------------
SEVENTY = (REPO_ROOT / "scripts" / "70-positive-control.sh").read_text(encoding="utf-8")
SEVENTYTWO = (REPO_ROOT / "scripts" / "72-detector-firstlight.sh").read_text(
    encoding="utf-8")


def test_70は6アームとα0_008333のまま():
    """⛔ 凍結したスクリプトは書き換えない。較正曲線は将来のランのために残す。"""
    assert "PC_ARMS=(pc-x00 pc-x02 pc-x05 pc-x10 pc-x20 pc-x40)" in SEVENTY
    assert "--alpha 0.008333" in SEVENTY


def test_72はM2でα0_025():
    """★ α は Holm の規則 0.05/M の機械的な帰結(M=2 → 0.0250)。"""
    assert "ALPHA_EFFECTIVE=0.025" in SEVENTYTWO
    assert "BASE_ARM=pcbase-swallow31-8b-x00" in SEVENTYTWO
    assert "DIRTY_ARM=df1L08t1-x40" in SEVENTYTWO


def test_72はn_摂動器_効果量_想定ψを70から変えていない():
    for frozen in ("SAMPLE_N=4742", "TARGET_EFFECT=0.05", "EXPECTED_PSI=0.4050",
                   "--perturbator shuffle_choices", "--k 1", "--split dev"):
        assert frozen in SEVENTY, frozen
        assert frozen in SEVENTYTWO, frozen


def test_72は操作チェックの通過印を要求する():
    assert 'reports/manipulation-check.$TAG.ok' in SEVENTYTWO


def test_72はHOLDOUTを開けない():
    """⛔ 停止条件 6。K = 1/10 は消費済みで、同じ HOLDOUT では再検定できない。"""
    assert "--split dev" in SEVENTYTWO
    assert "--split holdout" not in SEVENTYTWO
    assert "holdout" not in SEVENTYTWO.replace("HOLDOUT", "")


# ---------------------------------------------------------------------------
# ⑤ 副次の読み —— 推論を増やさずに、注入済み / 非注入で分ける
# ---------------------------------------------------------------------------
def _load_split_tool():
    spec = importlib.util.spec_from_file_location(
        "split_drop_by_injection", REPO_ROOT / "tools" / "split_drop_by_injection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_副次の読みは注入済み側に集中したdropを分けて見せる(tmp_path, capsys):
    """★ 汚染を模した応答を仕込み、注入済み側にだけ drop が出ることを確かめる。

    ⛔ これは配管の検査であって測定ではない。**実モデルとは無関係。**
    """
    from contamlab.benchmark import Item
    from contamlab.perturb import get_perturbator, perturb_all
    from contamlab.runner import ResponseCache, format_prompt, set_prompt_format

    module = _load_split_tool()
    set_prompt_format("C")

    items = [
        Item(id=f"q{i:03d}", question=f"問題 {i}", choices=["ア", "イ", "ウ", "エ"],
             answer="ア")
        for i in range(40)
    ]
    perturbed = perturb_all(items, get_perturbator("shuffle_choices"), "test-seed")
    injected = {item.id for item in items[:20]}

    cache_path = tmp_path / "cache.jsonl"
    cache = ResponseCache(cache_path)
    arm = "fake-dirty"
    for original, shuffled in zip(items, perturbed):
        # 原文条件はどちらの群も正解する。
        cache.put(arm, format_prompt(original), "答え: ア")
        # 摂動版は**注入済みの問題だけ**落ちる(位置で覚えている、を模す)。
        wrong = original.id in injected
        cache.put(arm, format_prompt(shuffled), "答え: イ" if wrong else "答え: ア")

    missing: list[str] = []
    orig = module.outcomes(cache, arm, items, missing)
    pert = module.outcomes(cache, arm, perturbed, missing)
    assert not missing, "キャッシュにある応答だけを読む(推論を増やさない)"

    flags = [item.id in injected for item in items]
    rows = {}
    for name, want in (("注入済み", True), ("非注入", False)):
        sel = [i for i, f in enumerate(flags) if f == want]
        rows[name] = module.report(name, [orig[i] for i in sel],
                                   [pert[i] for i in sel], 0.025)

    assert rows["注入済み"]["drop"] > rows["非注入"]["drop"]
    assert rows["非注入"]["drop"] == pytest.approx(0.0)


def test_副次の読みは応答が欠けていたら推論せずに止まる(tmp_path):
    from contamlab.benchmark import Item
    from contamlab.runner import ResponseCache, set_prompt_format

    module = _load_split_tool()
    set_prompt_format("C")
    items = [Item(id="q1", question="問題", choices=["ア", "イ"], answer="ア")]
    cache = ResponseCache(tmp_path / "cache.jsonl")
    missing: list[str] = []
    got = module.outcomes(cache, "fake", items, missing)
    assert got == [None] and missing == ["q1"]


# ---------------------------------------------------------------------------
# ⑥ 関門(df1_gate.py)—— 停止条件を機械に守らせる部分
# ---------------------------------------------------------------------------
def _gate():
    spec = importlib.util.spec_from_file_location(
        "df1_gate", REPO_ROOT / "scripts" / "df1_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_file(tmp_path, rows) -> Path:
    """65-manipulation-check.sh の出力を模した検査ファイルを書く。

    rows: [(arm, 非注入群 正解率, 非注入群 解釈不能率, 注入群 正解率 or None, 解釈不能率)]
    """
    lines = []
    for arm, acc_n, unp_n, acc_i, unp_i in rows:
        lines.append(f"  {arm}  (注入率 40%)")
        lines.append(f"      非注入群 n= 400  正解率 {acc_n:.4f}  解釈不能 {unp_n*100:.2f}%")
        if acc_i is not None:
            lines.append(f"      注入群   n= 400  正解率 {acc_i:.4f}  解釈不能 {unp_i*100:.2f}%")
    p = tmp_path / "check.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_G0_はベースが帯の中なら通る(tmp_path):
    gate = _gate()
    path = _check_file(tmp_path, [("pcbase-swallow31-8b-x00", 0.6250, 0.0000, None, 0)])
    assert gate.cmd_base(_Args(check=str(path))) == 0


@pytest.mark.parametrize("acc,unp", [(0.5000, 0.0), (0.7000, 0.0), (0.6250, 0.05)])
def test_G0_は帯を外れたら止める(tmp_path, acc, unp):
    """★ ll-01 の第0段の帯 [0.562, 0.658] / 解釈不能率 1.6% をそのまま使う。"""
    gate = _gate()
    path = _check_file(tmp_path, [("pcbase-swallow31-8b-x00", acc, unp, None, 0)])
    assert gate.cmd_base(_Args(check=str(path))) == 1


def test_G4_はll01の実測値なら通る(tmp_path):
    """ll-01 L1 の非注入群 0.6450 / 0.6200 / 0.6300(b = 0.30 の線)。"""
    gate = _gate()
    path = _check_file(tmp_path, [
        ("df1L08t1-x40", 0.6450, 0.0025, 0.7575, 0.0075),
        ("df1L08t2-x40", 0.6200, 0.0000, 0.7650, 0.0025),
        ("df1L08t3-x40", 0.6300, 0.0050, 0.7650, 0.0050),
    ])
    assert gate.cmd_gate_b(_Args(check=str(path))) == 0


def test_G4_は素の能力が壊れていたら止める(tmp_path):
    gate = _gate()
    path = _check_file(tmp_path, [
        (f"df1L08t{k}-x40", 0.20, 0.0, 0.35, 0.0) for k in (1, 2, 3)
    ])
    assert gate.cmd_gate_b(_Args(check=str(path))) == 1


def test_G4_は3本そろっていなければ読まない(tmp_path):
    """⛔ 判定の単位は「複製 k=3 本の分布」である(rj-01 が凍結)。"""
    gate = _gate()
    path = _check_file(tmp_path, [("df1L08t1-x40", 0.6450, 0.0025, 0.7575, 0.0075)])
    assert gate.cmd_gate_b(_Args(check=str(path))) == 1


def test_G5_はll01の実測値なら通り_壊れていたら止める(tmp_path):
    """条件 c の読みは cc-01 の実装をそのまま呼ぶ(規則を二重に書かない)。"""
    gate = _gate()
    ok = _check_file(tmp_path, [
        (f"df1L08t{k}-x40", 0.63, 0.005, 0.76, 0.005) for k in (1, 2, 3)
    ])
    assert gate.cmd_gate_c(_Args(check=str(ok))) == 0
    broken = tmp_path / "broken.txt"
    broken.write_text(_check_file(tmp_path, [
        (f"df1L08t{k}-x40", 0.16, 0.60, 0.17, 0.65) for k in (1, 2, 3)
    ]).read_text(encoding="utf-8"), encoding="utf-8")
    assert gate.cmd_gate_c(_Args(check=str(broken))) == 1


def test_G3_は50パーセント超で止める(tmp_path):
    gate = _gate()
    ok = _check_file(tmp_path, [("df1L08t1-x40", 0.6450, 0.0025, 0.7575, 0.0075)])
    assert gate.cmd_anomaly(_Args(check=str(ok))) == 0
    path = tmp_path / "bad.txt"
    path.write_text("  df1L08t1-x40  (注入率 40%)\n"
                    "      非注入群 n= 400  正解率 0.1625  解釈不能 64.75%\n",
                    encoding="utf-8")
    assert gate.cmd_anomaly(_Args(check=str(path))) == 1


def test_aの読みは報告だけで決して止めない(tmp_path, capsys):
    """⛔ preregister「★ a を関門にしない」。止める口を持たないことを機械で確かめる。"""
    gate = _gate()
    # a が明確に不合格(差 2pt)でも、返り値は 0 でなければならない。
    path = _check_file(tmp_path, [
        (f"df1L08t{k}-x40", 0.63, 0.005, 0.65, 0.005) for k in (1, 2, 3)
    ])
    assert gate.cmd_report_a(_Args(check=str(path))) == 0
    out = capsys.readouterr().out
    assert "とは書かない" in out, "a が落ちたときの縛りを印字していない"

    # ll-01 の実測(差 11.25 / 14.50 / 13.50pt)なら「頑健に合格」と読む。
    path = _check_file(tmp_path, [
        ("df1L08t1-x40", 0.6450, 0.0025, 0.7575, 0.0075),
        ("df1L08t2-x40", 0.6200, 0.0000, 0.7650, 0.0025),
        ("df1L08t3-x40", 0.6300, 0.0050, 0.7650, 0.0050),
    ])
    assert gate.cmd_report_a(_Args(check=str(path))) == 0
    assert "頑健に合格" in capsys.readouterr().out


def test_閾値は事前登録の値から動いていない():
    gate = _gate()
    assert gate.BASE_ACC_BAND == (0.562, 0.658)
    assert gate.BASE_UNPARSED == 0.016
    assert gate.ANOMALY_UNPARSED == 0.50
    assert gate.COND_A_DIFF == 0.10
    assert gate.COND_B_ACC == 0.30
    assert gate.T_FACTOR_K3 == 1.686
    assert gate.LAMBDA_EXPECTED == 0.8
    assert gate.LAMBDA_RELATIVE_TOLERANCE == 1e-6
