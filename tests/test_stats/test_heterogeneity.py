"""差の差(DiD)— 「摂動が難しくなっただけ」と「汚染」を分けられること。"""
from __future__ import annotations

import pytest

from contamlab.stats.heterogeneity import cochran_q, drop_standard_error
from contamlab.stats.mcnemar import PairedTable


def _table(both_correct: int, only_original: int, only_perturbed: int, both_wrong: int):
    return PairedTable(both_correct, only_original, only_perturbed, both_wrong)


class TestDropStandardError:
    def test_標本が増えると標準誤差が下がる(self) -> None:
        small = drop_standard_error(_table(70, 15, 5, 10))
        large = drop_standard_error(_table(700, 150, 50, 100))

        assert large < small

    def test_不一致ペアが0件なら0を返す(self) -> None:
        assert drop_standard_error(_table(80, 0, 0, 20)) == 0.0


class TestCochranQ:
    def test_全モデルが同じだけ落ちたら不均一さは検出されない(self) -> None:
        """★ これは「摂動が難しくなっただけ」の可能性を示す。汚染とは結論できない。"""
        tables = {name: _table(70, 15, 5, 10) for name in ("A", "B", "C")}

        result = cochran_q(tables)

        assert result.q_statistic == pytest.approx(0.0, abs=1e-12)
        assert result.p_value == pytest.approx(1.0)
        assert not result.heterogeneous
        assert result.pooled_drop == pytest.approx(0.10)
        assert result.i_squared == 0.0

    def test_均一な低下では結論を保留する(self) -> None:
        tables = {name: _table(70, 15, 5, 10) for name in ("A", "B", "C")}

        text = cochran_q(tables).interpretation()

        assert "区別できない" in text
        assert "対照モデル" in text

    def test_一部のモデルだけ落ちたら不均一さを検出する(self) -> None:
        """★ これが汚染の向き。"""
        tables = {
            "contaminated": _table(50, 35, 2, 13),
            "clean_a": _table(60, 6, 5, 29),
            "clean_b": _table(58, 5, 7, 30),
        }

        result = cochran_q(tables)

        assert result.heterogeneous
        assert result.p_value < 0.05
        assert result.df == 2
        assert result.i_squared > 0.5

    def test_低下も不均一さも無ければそう言う(self) -> None:
        tables = {
            "a": _table(45, 5, 5, 45),
            "b": _table(45, 5, 5, 45),
        }

        result = cochran_q(tables)

        assert not result.heterogeneous
        assert result.pooled_drop == pytest.approx(0.0)
        assert "有意な低下も不均一さも無い" in result.interpretation()

    def test_不一致ペアが0件のモデルは除外して記録する(self) -> None:
        """黙って落とさない。除外したことが結果に残る。"""
        tables = {
            "no_discordant": _table(80, 0, 0, 20),
            "a": _table(60, 20, 5, 15),
            "b": _table(60, 8, 12, 20),
        }

        result = cochran_q(tables)

        assert result.excluded == ["no_discordant"]
        assert result.model_names == ["a", "b"]
        assert result.df == 1

    def test_有効なモデルが2本未満なら検定できない(self) -> None:
        tables = {
            "only_one": _table(60, 20, 5, 15),
            "no_discordant": _table(80, 0, 0, 20),
        }

        with pytest.raises(ValueError, match="2モデル以上"):
            cochran_q(tables)

    def test_alphaの範囲を弾く(self) -> None:
        tables = {"a": _table(60, 20, 5, 15), "b": _table(60, 8, 12, 20)}

        with pytest.raises(ValueError):
            cochran_q(tables, alpha=0.6)
