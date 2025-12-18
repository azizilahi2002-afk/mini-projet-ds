from __future__ import annotations
import pandas as pd
import pytest

from src.preprocessing import clean_data, handle_missing, scale_features


class TestPreprocessing:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [25, 30, None, 45, 50],
                "salary": [30000, 45000, 50000, 60000, 70000],
                "city": ["Paris", "Lyon", "Paris", None, "Lyon"],
            }
        )

    def test_handle_missing_fills_nulls(self, sample_data: pd.DataFrame) -> None:
        result = handle_missing(sample_data)
        assert result.isna().sum().sum() == 0
        assert len(result) == len(sample_data)

    def test_scale_features_output_range(self, sample_data: pd.DataFrame) -> None:
        numeric_cols = ["age", "salary"]
        result = scale_features(sample_data[numeric_cols])
        for col in numeric_cols:
            assert 0.0 <= result[col].min() <= 1.0
            assert 0.0 <= result[col].max() <= 1.0

    def test_clean_data_removes_out_of_range_rows(self) -> None:
        data = pd.DataFrame(
            {
                "age": [25, -5, 30, 200],
                "salary": [30000, 40000, 50000, 60000],
            }
        )
        result = clean_data(data, age_min=0, age_max=120)
        assert len(result) == 2
        assert result["age"].between(0, 120).all()
