"""Regression coverage for sparse and short temporal histories."""

import pandas as pd
import pytest

from secom.config import FoldPlanName
from secom.cv import choose_outer_fold_plan


@pytest.mark.parametrize(
    ("last_week", "expected"),
    [(11, FoldPlanName.PRIMARY_3FOLD), (9, FoldPlanName.PRIMARY_3FOLD), (8, FoldPlanName.PRIMARY_3FOLD), (3, None)],
)
def test_short_histories_use_fixed_calendar_fractions_or_no_plan(last_week, expected):
    dev = pd.DataFrame(
        {
            "week_label": [week for week in range(1, last_week + 1) for _ in range(2)],
            "y_bin": [0, 1] * last_week,
            "timestamp": pd.date_range("2008-01-01", periods=2 * last_week, freq="D"),
        }
    )
    plan = choose_outer_fold_plan(dev)
    assert (plan.plan_name if plan else None) == expected


def test_empty_development_history_has_no_temporal_plan():
    dev = pd.DataFrame(columns=["week_label", "y_bin", "timestamp"])
    assert choose_outer_fold_plan(dev) is None
