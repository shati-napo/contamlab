"""検出力 — 手計算値を固定する。

ここの2つの数字がこのツールの看板になる。
    ψ=0.20 で 5pp を検出するには 493 問要る
    100 問では 11pp 未満は見えない
"""
from __future__ import annotations

import pytest

from contamlab.stats.power import (
    min_detectable_effect,
    plan,
    power_at_n,
    required_n,
)


class TestRequiredN:
    def test_5ポイントの低下には493問要る(self) -> None:
        """Connor (1987) の式を手計算すると 492.5 → 切り上げ 493。"""
        assert required_n(0.05, 0.20, alpha=0.05, power=0.80, one_sided=True) == 493

    def test_効果量が小さいほど多く要る(self) -> None:
        sizes = [required_n(d, 0.20) for d in (0.15, 0.10, 0.05, 0.02)]
        assert sizes == sorted(sizes)

    def test_検出力を上げるほど多く要る(self) -> None:
        assert required_n(0.05, 0.20, power=0.90) > required_n(0.05, 0.20, power=0.80)

    def test_両側のほうが多く要る(self) -> None:
        assert required_n(0.05, 0.20, one_sided=False) > required_n(0.05, 0.20, one_sided=True)

    def test_効果量が不一致率を超えたら弾く(self) -> None:
        """d = ψ(2π−1) なので |d| <= ψ。超える入力は設計の誤り。"""
        with pytest.raises(ValueError, match="不一致率"):
            required_n(0.30, 0.20)

    def test_不正な入力を弾く(self) -> None:
        with pytest.raises(ValueError):
            required_n(0.0, 0.20)
        with pytest.raises(ValueError):
            required_n(0.05, 0.0)
        with pytest.raises(ValueError):
            required_n(0.05, 0.20, power=1.0)
        with pytest.raises(ValueError):
            required_n(0.05, 0.20, alpha=0.5)


class TestPowerAtN:
    def test_必要問題数でちょうど目標検出力に届く(self) -> None:
        assert power_at_n(493, 0.05, 0.20) == pytest.approx(0.80, abs=0.005)

    def test_100問で5ポイントはほぼ見えない(self) -> None:
        """★ このツールを作る理由。多くの汚染研究がこの領域で実験している。"""
        assert power_at_n(100, 0.05, 0.20) < 0.30

    def test_問題数が増えると検出力が上がる(self) -> None:
        powers = [power_at_n(n, 0.05, 0.20) for n in (50, 100, 200, 500, 1000)]
        assert powers == sorted(powers)

    def test_検出力は0と1の間(self) -> None:
        assert 0.0 <= power_at_n(10, 0.05, 0.20) <= 1.0
        assert 0.0 <= power_at_n(100000, 0.05, 0.20) <= 1.0


class TestMinDetectableEffect:
    def test_100問での検出可能な最小効果量は約11ポイント(self) -> None:
        """★ 看板の数字。"""
        mde = min_detectable_effect(100, 0.20, alpha=0.05, power=0.80, one_sided=True)
        assert mde == pytest.approx(0.110, abs=1e-3)

    def test_MDEと必要問題数は整合する(self) -> None:
        """MDE を required_n に入れ直すと、元の n 付近に戻る。"""
        mde = min_detectable_effect(400, 0.25)
        assert required_n(mde, 0.25) == pytest.approx(400, abs=2)

    def test_問題数が増えるとMDEは下がる(self) -> None:
        mdes = [min_detectable_effect(n, 0.20) for n in (100, 200, 500, 1000)]
        assert mdes == sorted(mdes, reverse=True)

    def test_MDEでの検出力はちょうど目標値(self) -> None:
        mde = min_detectable_effect(250, 0.30, power=0.80)
        assert power_at_n(250, mde, 0.30) == pytest.approx(0.80, abs=1e-3)

    def test_原理的に届かない設計は明示的に落とす(self) -> None:
        """不一致率が低すぎると、どんな効果量でも目標検出力に届かない。"""
        with pytest.raises(ValueError, match="届かない"):
            min_detectable_effect(10, 0.02, power=0.80)


class TestPlan:
    def test_足りている設計(self) -> None:
        p = plan(n=600, discordant_rate=0.20, target_effect=0.05)

        assert p.required_n_for_target == 493
        assert p.adequate

    def test_足りない設計(self) -> None:
        p = plan(n=100, discordant_rate=0.20, target_effect=0.05)

        assert not p.adequate
        assert p.min_detectable > 0.05, "狙う効果量より MDE が大きい = 見えない"

    def test_目標効果量が無ければ判定しない(self) -> None:
        p = plan(n=100, discordant_rate=0.20)

        assert p.target_effect is None
        assert p.required_n_for_target is None
        assert p.adequate

    def test_何も検出できない設計でも例外にせず不足として返す(self) -> None:
        """★ 例外にすると生トレースが出る。これは異常ではなく正当な設計結果。"""
        p = plan(n=8, discordant_rate=0.30, target_effect=0.10)

        assert p.min_detectable is None
        assert not p.adequate
        assert "届かない" in p.describe_min_detectable()
        assert p.required_n_for_target is not None, "必要問題数は計算できる"

    def test_検出できない設計のサマリ(self) -> None:
        text = plan(n=8, discordant_rate=0.30, target_effect=0.10).summary()
        detectable_line = next(
            line for line in text.splitlines() if line.startswith("検出可能な最小")
        )

        assert "届かない" in detectable_line
        assert "ポイント" not in detectable_line, "0ポイントまで見える、と誤読させない"
        assert "★ 不足" in text

    def test_サマリに必要な数字が全部出る(self) -> None:
        text = plan(n=100, discordant_rate=0.20, target_effect=0.05).summary()

        assert "100" in text
        assert "不足" in text
        assert "ポイント" in text
