"""④ 固定評価系 — 検出力の門番、多重比較、そして測定装置自身の検査。"""
from __future__ import annotations

import pytest

from contamlab.benchmark import Item
from contamlab.harness import (
    Design,
    UnderpoweredError,
    _synthetic_items,
    run,
    self_check,
)
from contamlab.perturb import Identity, ShuffleChoices
from contamlab.report import format_result, format_self_check, result_to_dict
from contamlab.runner import FakeModel


def _design(**overrides) -> Design:
    base = dict(
        perturbator_name=ShuffleChoices.name,
        seed="test-seed",
        target_effect=0.05,
        expected_discordant_rate=0.30,
    )
    base.update(overrides)
    return Design(**base)


def _contaminated(items, name: str = "contaminated") -> FakeModel:
    return FakeModel(name, items, base_accuracy=0.25, memorized_ids=[i.id for i in items])


def _clean(items, name: str = "clean", accuracy: float = 0.60) -> FakeModel:
    return FakeModel(name, items, base_accuracy=accuracy)


class TestPowerGate:
    """★ このツールの存在理由。走らせる前に止める。"""

    def test_検出力が足りなければ実行を拒否する(self) -> None:
        items = _synthetic_items(100)

        with pytest.raises(UnderpoweredError, match="検出力が足りない"):
            run(items, [_clean(items)], _design(expected_discordant_rate=0.20))

    def test_拒否メッセージに必要問題数と検出可能な最小値が出る(self) -> None:
        items = _synthetic_items(100)

        with pytest.raises(UnderpoweredError) as exc:
            run(items, [_clean(items)], _design(expected_discordant_rate=0.20))

        message = str(exc.value)
        assert "493" in message, "5ptをψ=0.20で見るには493問要る"
        assert "100 問" in message

    def test_何も検出できない問題数でも綺麗に拒否する(self) -> None:
        """★ 生の ValueError を出さない。利用者が直せるメッセージにする。"""
        items = _synthetic_items(8)

        with pytest.raises(UnderpoweredError, match="検出力が足りない"):
            run(items, [_clean(items)], _design(target_effect=0.10))

    def test_強行できるが警告が残る(self) -> None:
        items = _synthetic_items(100)

        result = run(
            items,
            [_clean(items)],
            _design(expected_discordant_rate=0.20),
            force_underpowered=True,
        )

        assert any("検出力不足" in w for w in result.warnings)

    def test_足りていれば通る(self) -> None:
        items = _synthetic_items(800)

        result = run(items, [_clean(items)], _design())

        assert result.prior_plan.adequate
        assert result.n_items == 800


class TestDetection:
    def test_汚染ありを検出する(self) -> None:
        items = _synthetic_items(800)

        result = run(items, [_contaminated(items)], _design())
        model = result.models[0]

        assert model.detected
        assert model.mcnemar.drop > 0.5
        assert model.adjusted_lcb > 0.0

    def test_汚染なしを検出しない(self) -> None:
        """★ 偽陽性を出さないこと。"""
        items = _synthetic_items(800)

        result = run(items, [_clean(items)], _design())
        model = result.models[0]

        assert not model.detected
        assert model.adjusted_lcb <= 0.0

    def test_下限は点推定より小さい(self) -> None:
        items = _synthetic_items(800)

        # K=1 では割引が 0 なので adjusted_lcb == lcb。狭義に小さくなるのは K>1 のとき
        # (TestDeflation を参照)。
        model = run(items, [_contaminated(items)], _design()).models[0]

        assert model.adjusted_lcb == pytest.approx(model.mcnemar.lcb)
        assert model.mcnemar.lcb < model.mcnemar.drop


class TestDeflation:
    def test_試行回数Kが増えると判定が厳しくなる(self) -> None:
        """★ 摂動器を何種類も試して一番落ちたものを報告するのは p-hacking。"""
        items = _synthetic_items(800)
        model = _clean(items)

        k1 = run(items, [model], _design(n_perturbators_tried=1)).models[0]
        k30 = run(items, [model], _design(n_perturbators_tried=30)).models[0]

        assert k30.adjusted_lcb < k1.adjusted_lcb
        assert k1.deflation == 0.0
        assert k30.deflation > 0.0

    def test_K1なら割引なし(self) -> None:
        items = _synthetic_items(800)

        model = run(items, [_clean(items)], _design(n_perturbators_tried=1)).models[0]

        assert model.adjusted_lcb == pytest.approx(model.mcnemar.lcb)


class TestMultiplicity:
    def test_モデル本数の多重比較が判定に反映される(self) -> None:
        """★ 実際に踏んだ偽陽性の回帰テスト。

        汚染なしのモデルを3本並べると、そのうち1本の**素の** p 値が 0.05 を切る
        ことがある。Holm 補正後は非有意なので、名指ししてはいけない。
        効果量の下限だけを見る判定はここで偽陽性を出した。
        """
        items = _synthetic_items(800)
        models = [
            _contaminated(items, "dirty"),
            _clean(items, "clean-a", 0.60),
            _clean(items, "clean-b", 0.55),
        ]

        result = run(items, models, _design(seed="s1"))
        by_name = {m.model_name: m for m in result.models}

        assert by_name["dirty"].detected
        assert not by_name["clean-a"].detected

        suspect = by_name["clean-b"]
        assert suspect.adjusted_lcb > 0.0, "効果量の下限だけなら陽性に見える"
        assert suspect.p_holm >= 0.05, "Holm 補正後は非有意"
        assert not suspect.detected, "★補正後が非有意なら汚染と呼ばない"

    def test_モデルを並べると補正がかかる(self) -> None:
        items = _synthetic_items(800)
        models = [_clean(items, f"clean{i}", 0.5 + i * 0.05) for i in range(4)]

        result = run(items, models, _design())

        for model in result.models:
            assert model.p_holm >= model.mcnemar.p_value
            assert model.p_bh >= model.mcnemar.p_value

    def test_HolmはBHより厳しい(self) -> None:
        items = _synthetic_items(800)
        models = [_contaminated(items, "dirty")] + [
            _clean(items, f"clean{i}", 0.5 + i * 0.05) for i in range(3)
        ]

        result = run(items, models, _design())

        assert all(m.p_holm >= m.p_bh for m in result.models)


class TestHeterogeneity:
    def test_モデルが1本なら判定しない(self) -> None:
        items = _synthetic_items(800)

        result = run(items, [_clean(items)], _design())

        assert result.heterogeneity is None

    def test_汚染ありと汚染なしを並べると不均一さが出る(self) -> None:
        """★ 差の差。一部だけ落ちる = 汚染。"""
        items = _synthetic_items(800)
        models = [_contaminated(items, "dirty"), _clean(items, "clean")]

        result = run(items, models, _design())

        assert result.heterogeneity is not None
        assert result.heterogeneity.heterogeneous

    def test_全部汚染なしなら不均一さは出ない(self) -> None:
        items = _synthetic_items(800)
        models = [_clean(items, f"clean{i}", 0.6) for i in range(3)]

        result = run(items, models, _design())

        assert not result.heterogeneity.heterogeneous


class TestPairing:
    def test_対応がずれていたら落とす(self) -> None:
        """id がずれた状態で検定すると、意味のない数字が静かに出る。"""
        items = _synthetic_items(800)

        class ShiftIds:
            name = "shift_ids"

            def apply(self, item: Item, seed: str) -> Item:
                return Item(
                    id=item.id + "-shifted",
                    question=item.question,
                    answer=item.answer,
                    choices=item.choices,
                )

        with pytest.raises(ValueError, match="対応がずれている"):
            run(items, [_clean(items)], _design(), perturbator=ShiftIds())

    def test_摂動が正解を変えたら落とす(self) -> None:
        items = _synthetic_items(800)

        class BreakAnswer:
            name = "break_answer"

            def apply(self, item: Item, seed: str) -> Item:
                return Item(
                    id=item.id,
                    question=item.question,
                    answer=item.choices[1],
                    choices=item.choices,
                )

        with pytest.raises(ValueError, match="正解を変えている"):
            run(items, [_clean(items)], _design(), perturbator=BreakAnswer())


class TestIdentityPerturbator:
    def test_何も変えなければ不一致ペアはゼロ(self) -> None:
        """★ 測定装置の健全性チェック。ここが0でなければモデルが非決定的。"""
        items = _synthetic_items(800)

        result = run(
            items,
            [_clean(items)],
            _design(perturbator_name=Identity.name),
            perturbator=Identity(),
        )
        model = result.models[0]

        assert model.table.n_discordant == 0
        assert model.mcnemar.drop == 0.0
        assert not model.detected


class TestObservedPower:
    def test_実測の不一致率と達成検出力が出る(self) -> None:
        items = _synthetic_items(800)

        result = run(items, [_clean(items)], _design())

        assert 0.0 < result.observed_discordant_rate < 1.0
        assert result.observed_power is not None

    def test_不一致率がゼロなら検出力は算出不能(self) -> None:
        items = _synthetic_items(800)

        result = run(
            items,
            [_clean(items)],
            _design(perturbator_name=Identity.name),
            perturbator=Identity(),
        )

        assert result.observed_discordant_rate == 0.0
        assert result.observed_power is None


class TestGuards:
    def test_問題が空なら落とす(self) -> None:
        with pytest.raises(ValueError, match="問題が1件も無い"):
            run([], [_clean(_synthetic_items(1))], _design())

    def test_モデルが空なら落とす(self) -> None:
        items = _synthetic_items(800)

        with pytest.raises(ValueError, match="モデルが1本も無い"):
            run(items, [], _design())


class TestSelfCheck:
    def test_3項目すべて通過する(self) -> None:
        """★ これが落ちたら実験を止める。測定装置が壊れていれば全実験が無価値。"""
        checks = self_check()

        assert len(checks) == 3
        for check in checks:
            assert check.passed, f"{check.name}: {check.detail}"

    def test_項目名(self) -> None:
        names = [c.name for c in self_check()]

        assert names == [
            "汚染ありを検出する",
            "汚染なしを検出しない",
            "何も変えなければ差はちょうど0",
        ]


class TestReport:
    def test_テキスト出力に必要な要素が全部入る(self) -> None:
        items = _synthetic_items(800)
        models = [_contaminated(items, "dirty"), _clean(items, "clean")]

        text = format_result(run(items, models, _design()))

        assert "設計(事前確約)" in text
        assert "標本と検出力" in text
        assert "不均一さ" in text
        assert "dirty" in text and "clean" in text
        assert "★汚染" in text

    def test_JSONに再現に必要な設計値が入る(self) -> None:
        items = _synthetic_items(800)

        payload = result_to_dict(run(items, [_contaminated(items)], _design()))

        assert payload["design"]["seed"] == "test-seed"
        assert payload["design"]["perturbator"] == "shuffle_choices"
        assert payload["sample"]["n_items"] == 800
        assert payload["models"][0]["detected"] is True
        assert "adjusted_lcb" in payload["models"][0]

    def test_健全性チェックの整形(self) -> None:
        text = format_self_check(self_check())

        assert "OK" in text
        assert "実験を進めてよい" in text
