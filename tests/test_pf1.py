"""ラン perturbation-floor-01 の回帰テスト。

★ **preregister の凍結値を、機械が守る。**
  人が後から「ちょっとだけ」動かせないようにするのがこのファイルの目的である。

⛔ **既存ランのスクリプトが1文字も変わっていないことも、ここで確かめる。**
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "scripts" / "lib.sh"
PULL = REPO / "scripts" / "80-pf1-pull.sh"
PILOT = REPO / "scripts" / "81-pf1-pilot.sh"
PROD = REPO / "scripts" / "82-pf1-production.sh"
JUDGE = REPO / "scripts" / "pf1_judge.py"
ORCH = REPO / "scripts" / "pf1-orchestrate.sh"


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# ロースター —— preregister「ロースター」の凍結表
# --------------------------------------------------------------------------

EXPECTED_ROSTER = [
    ("swallow31-8b", "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q4_K_M"),
    ("llmjp3-13b", "hf.co/mmnga/llm-jp-3-13b-instruct3-gguf:Q4_K_M"),
    ("elyza3-8b", "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf:Q4_K_M"),
    ("plamo2-8b", "hf.co/mmnga/plamo-2-8b-gguf:Q4_K_M"),
    ("cyberagent-nemo-12b",
     "hf.co/mmnga/cyberagent-Mistral-Nemo-Japanese-Instruct-2408-gguf:Q4_K_M"),
]


def _roster(var: str) -> list[tuple[str, str]]:
    body = _text(LIB)
    m = re.search(rf"^{var}=\(\n(.*?)^\)", body, re.S | re.M)
    assert m, f"{var} が lib.sh に無い"
    out = []
    for line in m.group(1).strip().splitlines():
        entry = line.strip().strip('"')
        name, _alias, repo = entry.split("|")
        out.append((name, repo))
    return out


def test_roster_is_frozen():
    """⛔ ロースターは preregister の凍結表そのもの。ここで選び直さない。"""
    assert _roster("PF1_ROSTER") == EXPECTED_ROSTER


def test_roster_is_five_distinct_models():
    """★ 選定基準 5(開発元が互いに異なる)。名前の重複が無いことで代理する。"""
    names = [n for n, _ in _roster("PF1_ROSTER")]
    assert len(names) == 5
    assert len(set(names)) == 5


def test_roster_is_all_mmnga_q4km():
    """⛔ 停止条件 3: mmnga 以外の提供元・Q4_K_M 以外の量子化を引かない。"""
    for name, repo in _roster("PF1_ROSTER"):
        assert repo.startswith("hf.co/mmnga/"), name
        assert repo.endswith(":Q4_K_M"), name


def test_existing_names_are_reused_not_renamed():
    """★ 既存ランの NAME をそのまま使う。

    runner.py:180 が応答キャッシュのキーに model_name を使うので、
    一度決めた名前は二度と変えない。
    """
    old = {n for n, _ in _roster("ROSTER")}
    new = {n for n, _ in _roster("PF1_ROSTER")}
    assert old <= new, "既存ランの NAME が PF1_ROSTER に含まれていない"


def test_existing_roster_untouched():
    """⛔ 既存ランの ROSTER は1文字も動かさない(2本のまま)。"""
    old = _roster("ROSTER")
    assert old == [
        ("llmjp3-13b", "hf.co/mmnga/llm-jp-3-13b-instruct3-gguf:Q4_K_M"),
        ("swallow31-8b", "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q4_K_M"),
    ]


# --------------------------------------------------------------------------
# 凍結した定数
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("PF1_PILOT_N", "150"),
    ("PF1_SAMPLE_N", "4742"),
    ("PF1_MAX_UNPARSABLE", "0.05"),
    ("PF1_MIN_SURVIVORS", "3"),
])
def test_frozen_constants(key, value):
    """⛔ preregister「段と関門」「停止条件」の数値。結果を見てから動かさない。"""
    m = re.search(rf"^{key}=(\S+)", _text(LIB), re.M)
    assert m, f"{key} が lib.sh に無い"
    assert m.group(1) == value


def test_df1_base_drop_is_the_measured_value():
    """★ 判定 D2・D3 の相手は df1 の実測値。⛔ 書き換えない。"""
    assert "0.01412905946857866" in _text(JUDGE)


def test_judge_targets_swallow_for_d3():
    """★ D3 は「同じモデルの別 GGUF」の比較。対象は swallow31-8b に事前固定。"""
    assert re.search(r'^D3_MODEL\s*=\s*"swallow31-8b"', _text(JUDGE), re.M)


# --------------------------------------------------------------------------
# 測定条件 —— ⛔ df1 と同じ経路であること
# --------------------------------------------------------------------------

def test_production_does_not_pass_alpha():
    """⛔ contamlab run に --alpha を渡さない(既定 0.05)。

    2026-08-22 に実測で確認した —— 72:104 の --alpha は contamlab power にだけ
    渡っており、contamlab run には渡っていない。本ランも同じ経路を通す。
    """
    body = _text(PROD)
    m = re.search(r"contamlab run \\\n(.*?)\n\n", body, re.S)
    assert m, "contamlab run の呼び出しが読めない"
    assert "--alpha" not in m.group(1)


def test_production_uses_dev_split_only():
    """⛔ HOLDOUT は開けない(停止条件 4)。"""
    body = _text(PROD)
    assert "--split dev" in body
    assert "--split holdout" not in body


def test_production_uses_frozen_perturbator_and_k():
    """⛔ 摂動器と K は凍結値(shuffle_choices / K=1)。"""
    body = _text(PROD)
    assert "--perturbator shuffle_choices" in body
    assert "--k 1" in body


def test_production_reads_survivors_not_full_roster():
    """★ 本番は生存モデルだけを測る。人が手で選ぶ余地を残さない。"""
    assert "pf1_model_flags" in _text(PROD)
    assert "pf1_all_model_flags" not in _text(PROD)


def test_pilot_uses_separate_cache():
    """⛔ パイロットのキャッシュを本番と混ぜない。

    混ぜると n=150 の応答が n=4,742 の測定に再生され、実際には呼ばれていない
    問題が「測った」ことになる(runner.py:223 の短絡)。
    """
    assert "pf1_pilot_cache_path" in _text(PILOT)


def test_pilot_does_not_use_contamlab_run():
    """⛔ n=150 では検出力ゲートが止める。パイロットは内部 API を直接呼ぶ。"""
    assert "-m contamlab run" not in _text(PILOT)


def test_no_training_in_this_run():
    """⛔ 停止条件 6: 本ランは学習を1回もしない。"""
    for p in (PULL, PILOT, PROD, ORCH):
        assert "train_lora" not in _text(p), p.name


# --------------------------------------------------------------------------
# ⛔ 既存ランのスクリプトに触っていないこと
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "70-positive-control.sh", "72-detector-firstlight.sh",
    "10-bootstrap.sh", "df1-orchestrate.sh",
])
def test_existing_scripts_untouched_by_this_run(name):
    """⛔ 本ランのために既存スクリプトを書き換えていない。

    ここで見るのは **本ランの名前や定数が紛れ込んでいないこと**。
    """
    body = _text(REPO / "scripts" / name)
    assert "perturbation-floor" not in body, f"{name} に本ランの名前が入っている"
    assert "PF1_" not in body, f"{name} が PF1 の定数を参照している"


# --------------------------------------------------------------------------
# 判定スクリプトの挙動
# --------------------------------------------------------------------------

def _run_judge(payload: dict, tmp_path: Path) -> str:
    f = tmp_path / "r.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(JUDGE), str(f)],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _model(name: str, drop: float, lcb: float, ci: tuple[float, float]) -> dict:
    return {
        "name": name, "table": {}, "accuracy_original": 0.6,
        "accuracy_perturbed": 0.6 - drop,
        "drop": drop, "drop_se": 0.006, "ci_low": ci[0], "ci_high": ci[1],
        "lcb": lcb, "deflation": 0.0, "adjusted_lcb": lcb,
        "p_value": 0.01, "p_holm": 0.02, "p_bh": 0.01,
        "unparsed_original": 5, "unparsed_perturbed": 6, "detected": True,
    }


def _payload(models: list[dict]) -> dict:
    return {
        "design": {"perturbator": "shuffle_choices", "alpha": 0.05},
        "sample": {"n_items": 4742, "observed_discordant_rate": 0.4,
                   "observed_power": 0.9, "required_n": 1270,
                   "min_detectable_effect": 0.026},
        "models": models, "heterogeneity": None, "warnings": [],
        "prompt_format": "C",
    }


def test_d1_passes_when_all_lcb_positive(tmp_path):
    out = _run_judge(_payload([
        _model("a", 0.02, 0.005, (0.01, 0.03)),
        _model("b", 0.015, 0.002, (0.005, 0.025)),
        _model("c", 0.018, 0.004, (0.008, 0.028)),
    ]), tmp_path)
    assert "D1: ✅ 通過" in out


def test_d1_fails_when_one_lcb_is_not_positive(tmp_path):
    out = _run_judge(_payload([
        _model("a", 0.02, 0.005, (0.01, 0.03)),
        _model("b", 0.001, -0.004, (-0.009, 0.011)),
        _model("c", 0.018, 0.004, (0.008, 0.028)),
    ]), tmp_path)
    assert "D1: ❌ 不通過" in out


def test_d2_reads_the_band_of_drops(tmp_path):
    """df1 の素のベース +1.4129pt が帯の内側かどうか。"""
    inside = _run_judge(_payload([
        _model("a", 0.005, 0.001, (0.0, 0.01)),
        _model("b", 0.030, 0.020, (0.02, 0.04)),
        _model("c", 0.018, 0.004, (0.008, 0.028)),
    ]), tmp_path)
    assert "D2: ✅ 通過" in inside

    outside = _run_judge(_payload([
        _model("a", 0.020, 0.010, (0.01, 0.03)),
        _model("b", 0.030, 0.020, (0.02, 0.04)),
        _model("c", 0.025, 0.015, (0.015, 0.035)),
    ]), tmp_path)
    assert "D2: ❌ 不通過" in outside


def test_d3_is_undecidable_when_swallow_dropped_out(tmp_path):
    """⛔ swallow31-8b が居なければ「判定不能」。「不通過」と書かない。"""
    out = _run_judge(_payload([
        _model("elyza3-8b", 0.02, 0.005, (0.01, 0.03)),
        _model("plamo2-8b", 0.015, 0.002, (0.005, 0.025)),
        _model("llmjp3-13b", 0.018, 0.004, (0.008, 0.028)),
    ]), tmp_path)
    assert "D3: — 判定不能" in out
    assert "D3: ❌" not in out


def test_d3_passes_when_ci_contains_the_selfbuilt_drop(tmp_path):
    out = _run_judge(_payload([
        _model("swallow31-8b", 0.016, 0.006, (0.008, 0.024)),
        _model("elyza3-8b", 0.02, 0.005, (0.01, 0.03)),
        _model("plamo2-8b", 0.015, 0.002, (0.005, 0.025)),
    ]), tmp_path)
    assert "D3: ✅ 通過" in out


def test_d3_fails_and_names_the_fourth_suspect(tmp_path):
    """★ D3 不通過は「失敗」ではなく最も重い結果。文言で明示する。"""
    out = _run_judge(_payload([
        _model("swallow31-8b", 0.040, 0.030, (0.032, 0.048)),
        _model("elyza3-8b", 0.02, 0.005, (0.01, 0.03)),
        _model("plamo2-8b", 0.015, 0.002, (0.005, 0.025)),
    ]), tmp_path)
    assert "D3: ❌ 不通過" in out
    assert "第4の容疑者" in out


def test_judge_never_rereads_df1_verdict(tmp_path):
    """⛔ 本ランの結果に関わらず df1 の判定 A・B は凍結されたまま、と必ず印字する。"""
    out = _run_judge(_payload([
        _model("swallow31-8b", 0.016, 0.006, (0.008, 0.024)),
        _model("elyza3-8b", 0.02, 0.005, (0.01, 0.03)),
        _model("plamo2-8b", 0.015, 0.002, (0.005, 0.025)),
    ]), tmp_path)
    assert "判定 A・B は、本ランの結果に関わらず凍結されたままである" in out
