"""統計層 — このツールの中心。

外部依存を持たない。式をそのまま書き、既知の表値への回帰テスト(`tests/test_stats/`)で
固定している。scipy に隠れた実装を信用する代わりに、全部読める状態にしてある。
"""

from .distributions import (
    binomial_sf,
    binomial_sf_half,
    chi2_sf,
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    normal_cdf,
    normal_quantile,
    wilson_interval,
)
from .heterogeneity import HeterogeneityResult, cochran_q, drop_standard_error
from .mcnemar import McNemarResult, PairedTable, mcnemar_test, table_from_outcomes
from .multiplicity import (
    benjamini_hochberg,
    deflated_threshold,
    expected_max_of_k,
    holm,
)
from .power import (
    PowerPlan,
    min_detectable_effect,
    plan,
    power_at_n,
    required_n,
)

__all__ = [
    # distributions
    "binomial_sf",
    "binomial_sf_half",
    "chi2_sf",
    "clopper_pearson_interval",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "normal_cdf",
    "normal_quantile",
    "wilson_interval",
    # mcnemar
    "McNemarResult",
    "PairedTable",
    "mcnemar_test",
    "table_from_outcomes",
    # power
    "PowerPlan",
    "min_detectable_effect",
    "plan",
    "power_at_n",
    "required_n",
    # multiplicity
    "benjamini_hochberg",
    "deflated_threshold",
    "expected_max_of_k",
    "holm",
    # heterogeneity
    "HeterogeneityResult",
    "cochran_q",
    "drop_standard_error",
]
