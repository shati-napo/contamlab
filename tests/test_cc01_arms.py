"""tests/test_cc01_arms.py — ラン calibration-curve-01 の凍結表の検査。

★ 本ランは **pc-01 以来はじめて注入率を振る。**pc-04 以降のランはすべて `-x40` の
  1水準しか使っておらず、**器がアーム名の末尾2桁から注入集合を引く経路**は眠っていた。
  ⛔ preregister の停止条件 3 —— 「アーム名から引いた注入集合が凍結表の件数と
  一致しない、または入れ子が壊れている」—— を、機械が守れるようにする。

  ここで押さえるのは4点:
    ① 18本のアームが**注入率と複製で必ず名前が分かれる**(応答キャッシュの混線を防ぐ)
    ② **T は全アーム共通の固定値**であり、注入トークン × E ではない
    ③ 注入トークン数は**錨(x40 = 238,082)と単調性**で守られる
    ④ 注入率を振るランと振らないランが**取り違えられない**

  ⛔ GPU にもネットワークにも出ない。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "finetune"))

train_lora = importlib.import_module("train_lora")
scale_adapter = importlib.import_module("scale_adapter")


# ---------------------------------------------------------------------------
# ① アーム名 —— 注入率 × 複製で必ず分かれる
# ---------------------------------------------------------------------------
def test_学習アームは6水準_3複製の18本():
    arms = train_lora.CC01_TRAIN_ARMS
    assert len(arms) == 18
    assert len(set(arms.values())) == 18, "アーム名が重複すると応答キャッシュが混線する"


def test_λアームも18本で学習アームと名前が衝突しない():
    scaled = set(scale_adapter.CC01_LAMBDA_ARMS.values())
    train = set(train_lora.CC01_TRAIN_ARMS.values())
    assert len(scaled) == 18
    assert not (scaled & train)


def test_λの段は0_8の1つだけ():
    """⛔ 停止条件 15。λ と注入率の2軸を同時に動かすと較正曲線でなくなる。"""
    assert scale_adapter.CC01_STEPS == ("L1",)
    assert scale_adapter.LAMBDA_STEPS["L1"] == 0.8
    assert all("L08" in a for a in scale_adapter.CC01_LAMBDA_ARMS.values())


@pytest.mark.parametrize("rate", train_lora.CC01_RATES)
def test_アーム名の末尾2桁が注入率になっている(rate):
    """器は `arm[-2:]` を注入率として読む(65-manipulation-check.sh:70)。"""
    for k in train_lora.CC01_REPLICATES:
        assert train_lora.CC01_TRAIN_ARMS[(rate, k)][-2:] == rate
        assert scale_adapter.CC01_LAMBDA_ARMS[("L1", k, rate)][-2:] == rate


def test_過去ランのアームと1つも衝突しない():
    """★ 応答キャッシュのキーはモデル名。使い回すと過去ランの答えが返る。"""
    cc = set(train_lora.CC01_ARMS)
    others = set(train_lora.LADDER_ARMS) - cc
    assert not (cc & others)


def test_水準ごとに3本の複製が別名である():
    for rate in train_lora.CC01_RATES:
        names = {train_lora.CC01_TRAIN_ARMS[(rate, k)]
                 for k in train_lora.CC01_REPLICATES}
        assert len(names) == 3, f"x{rate} の複製3本が名前で分かれていない"


# ---------------------------------------------------------------------------
# ② T —— 全アーム共通の固定値(★ 他のランと決まり方が違う)
# ---------------------------------------------------------------------------
def test_T_は錨のトークン数掛けるE_と一致する():
    """T = 8,570,952 = x40 の注入 238,082 × E=36(ll-01 の実測)。"""
    e = train_lora.RECIPES[train_lora.CC01_RECIPE]["E"]
    assert e == 36
    assert train_lora.CC01_TOTAL_TOKENS_T == train_lora.CC01_ANCHOR_TOKENS * e


def test_注入問題数の凍結表がpc01と一致する():
    assert train_lora.CC01_N_INJECTED == {
        "00": 0, "02": 94, "05": 237, "10": 474, "20": 948, "40": 1896,
    }


def test_注入問題数は水準とともに増える():
    n = [train_lora.CC01_N_INJECTED[r] for r in train_lora.CC01_RATES]
    assert n == sorted(n) and len(set(n)) == len(n)


# ---------------------------------------------------------------------------
# ③ 注入トークン数 —— 錨と単調性で守る
# ---------------------------------------------------------------------------
def _manifest(tmp_path, table):
    p = tmp_path / "manifest-cc01.json"
    p.write_text(json.dumps({"injected_tokens_once": table}), encoding="utf-8")
    return p


GOOD = {"00": 0, "02": 11_000, "05": 29_000, "10": 59_000, "20": 119_000,
        "40": train_lora.CC01_ANCHOR_TOKENS}


def test_凍結表から注入トークン数を引ける(tmp_path):
    p = _manifest(tmp_path, GOOD)
    assert train_lora.cc01_injected_tokens("05", p) == 29_000
    assert train_lora.cc01_injected_tokens("00", p) == 0


def test_錨が合わなければ止まる(tmp_path):
    """⛔ x40 が過去ランの実測値と違えば tokenizer か注入集合が違う。"""
    bad = dict(GOOD, **{"40": 238_083})
    with pytest.raises(SystemExit) as e:
        train_lora.cc01_injected_tokens("05", _manifest(tmp_path, bad))
    assert "錨" in str(e.value)


def test_表が無ければ止まる(tmp_path):
    with pytest.raises(SystemExit) as e:
        train_lora.cc01_injected_tokens("05", tmp_path / "none.json")
    assert "measure-tokens" in str(e.value)


def test_水準が表に無ければ止まる(tmp_path):
    partial = {k: v for k, v in GOOD.items() if k != "05"}
    with pytest.raises(SystemExit):
        train_lora.cc01_injected_tokens("05", _manifest(tmp_path, partial))


# ---------------------------------------------------------------------------
# ④ ランの取り違え —— 注入率を振るのは cc-01 だけ
# ---------------------------------------------------------------------------
def test_注入率を振るのはcc01だけ():
    rated = {r for r, s in scale_adapter.RUNS.items() if s.get("rated")}
    assert rated == {train_lora.CC01_RUN}


def test_cc01は複製ランでもある():
    assert scale_adapter.RUNS[train_lora.CC01_RUN]["replicated"] is True


def test_ベースは過去ランと同じswallow8b():
    assert train_lora.RUN_BASES[train_lora.CC01_RUN] == train_lora.SWALLOW_8B
    assert train_lora.RUN_BASES["lambda-ladder-01"] == train_lora.SWALLOW_8B


def test_レシピはR1に固定されている():
    assert train_lora.CC01_RECIPE == "R1"
    assert scale_adapter.RUNS[train_lora.CC01_RUN]["recipe"] == "R1"


def test_出所と書き出しのアームが同じ注入率になる():
    for rate in train_lora.CC01_RATES:
        for k in train_lora.CC01_REPLICATES:
            src = scale_adapter.source_arm(train_lora.CC01_RUN, k, rate)
            dst = scale_adapter.target_arm(train_lora.CC01_RUN, "L1", k, rate)
            assert src[-2:] == dst[-2:] == rate


def test_注入率を振らないランはrateを持たない():
    for run in ("positive-control-06", "lambda-ladder-01"):
        assert not scale_adapter.RUNS[run].get("rated")
        # 既存ランのアームは 40 のまま。★ 挙動を1つも変えていないことの確認。
        assert scale_adapter.target_arm(
            run, "L1", 1 if scale_adapter.RUNS[run]["replicated"] else None
        ).endswith("40")


# ---------------------------------------------------------------------------
# 実データ —— 注入集合が凍結表どおりに複製されているか
# ---------------------------------------------------------------------------
INJ = REPO_ROOT / "data" / "injection"


@pytest.mark.skipif(not (INJ / "pc-x40.ids").exists(), reason="注入集合が未生成")
def test_pc01の注入集合が凍結表の件数と入れ子を満たす():
    ids = {r: set((INJ / f"pc-x{r}.ids").read_text(encoding="utf-8").split())
           for r in train_lora.CC01_RATES}
    for r in train_lora.CC01_RATES:
        assert len(ids[r]) == train_lora.CC01_N_INJECTED[r], f"x{r} の件数"
    ladder = [r for r in train_lora.CC01_RATES if r != "00"]
    for lo, hi in zip(ladder, ladder[1:]):
        assert ids[lo] <= ids[hi], f"x{lo} ⊄ x{hi}"


@pytest.mark.skipif(not (INJ / "manifest-cc01.json").exists(), reason="複製が未実行")
def test_複製された36本が凍結表と一致する():
    m = json.loads((INJ / "manifest-cc01.json").read_text(encoding="utf-8"))
    names = {a["name"] for a in m["arms"]}
    expected = (set(train_lora.CC01_TRAIN_ARMS.values())
                | set(scale_adapter.CC01_LAMBDA_ARMS.values()))
    assert names == expected
    assert m["total_tokens_t"] == train_lora.CC01_TOTAL_TOKENS_T
    for a in m["arms"]:
        assert a["n_injected"] == train_lora.CC01_N_INJECTED[a["name"][-2:]]
