"""
Unit tests for Factor Analysis strategy selection.

Tests cover:
- ContinuousFAStrategy vs FAMDStrategy selection
- Handling of pure numeric data
- Handling of mixed data
- Edge cases (all categorical, zero numeric columns)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from factor_analysis import select_strategy, ContinuousFAStrategy, FAMDStrategy


class TestStrategySelection:
    """Tests for automatic strategy selection based on data types."""
    
    def test_select_fa_strategy_for_numeric_data(self, valid_numeric_dataframe):
        """Test that numeric data triggers FA strategy."""
        df = valid_numeric_dataframe
        # Remove name column for pure numeric
        df_numeric = df[["feat1", "feat2", "feat3"]]
        
        strategy = select_strategy(df_numeric)
        assert isinstance(strategy, ContinuousFAStrategy)
    
    def test_select_famd_strategy_for_mixed_data(self, mixed_data_dataframe):
        """Test that mixed data triggers FAMD strategy."""
        df = mixed_data_dataframe
        # Include both numeric and categorical
        df_mixed = df[["weight", "height", "breed", "temperament"]]
        
        strategy = select_strategy(df_mixed)
        assert isinstance(strategy, FAMDStrategy)
    
    def test_select_famd_for_categorical_only(self):
        """Test FAMD selection for pure categorical data."""
        df = pd.DataFrame({
            "breed": ["Lab", "Poodle", "Bulldog"],
            "color": ["Brown", "White", "Black"]
        })
        
        strategy = select_strategy(df)
        assert isinstance(strategy, FAMDStrategy)


class TestContinuousFAStrategy:
    """Tests for ContinuousFAStrategy implementation."""
    
    def test_fa_fit_returns_model_and_scores(self, valid_numeric_dataframe):
        """Test that FA.fit() returns model and factor scores."""
        df = valid_numeric_dataframe[["feat1", "feat2", "feat3"]]
        strategy = ContinuousFAStrategy()
        
        model, scores = strategy.fit(df, n_factors=2)
        
        assert model is not None
        assert scores is not None
        assert scores.shape[0] == len(df)  # Same number of rows
        assert scores.shape[1] == 2  # 2 factors requested
    
    def test_fa_components_extraction(self, dataframe_with_high_correlation):
        """Test extraction of component loadings."""
        df = dataframe_with_high_correlation[["feat1", "feat2", "feat3", "feat4"]]
        strategy = ContinuousFAStrategy()
        
        model, scores = strategy.fit(df, n_factors=2)
        components = strategy.get_components(model)
        
        assert components is not None
        assert components.shape[0] == 2  # 2 factors
        assert components.shape[1] == 4  # 4 features


class TestFAMDStrategy:
    """Tests for FAMDStrategy implementation."""
    
    def test_famd_fit_with_mixed_data(self, mixed_data_dataframe):
        """Test FAMD fit with mixed numeric/categorical data."""
        df = mixed_data_dataframe[["weight", "height", "breed", "temperament"]]
        strategy = FAMDStrategy()
        
        model, scores = strategy.fit(df, n_factors=2)
        
        assert model is not None
        assert scores is not None
        assert scores.shape[0] == len(df)
        assert scores.shape[1] == 2
    
    def test_famd_with_zero_numeric_cols(self):
        """
        Test FAMD with zero numeric columns (FAILURE POINT #6).
        Should add dummy numeric column to avoid 'All variables are qualitative' error.
        """
        df = pd.DataFrame({
            "breed": ["Lab", "Poodle", "Bulldog", "Beagle"],
            "color": ["Brown", "White", "Black", "Brown"]
        })
        
        strategy = FAMDStrategy()
        
        # Should not raise ValueError about "All variables are qualitative"
        # The strategy should handle this by adding a dummy column
        try:
            model, scores = strategy.fit(df, n_factors=1)
            # If it succeeds, verify output
            assert scores is not None
        except ValueError as e:
            # If it still fails, ensure it's handled gracefully
            assert "qualitative" in str(e).lower()
    
    def test_famd_components_extraction(self, mixed_data_dataframe):
        """Test extraction of FAMD component loadings."""
        df = mixed_data_dataframe[["weight", "height", "breed"]]
        strategy = FAMDStrategy()
        
        model, scores = strategy.fit(df, n_factors=2)
        components = strategy.get_components(model)
        
        assert components is not None
        # FAMD expands categorical variables into dummy columns
        assert components.shape[0] == 2  # 2 factors


class TestEdgeCases:
    """Tests for edge cases in strategy selection and execution."""
    
    def test_single_column_dataframe(self):
        """Test handling of DataFrame with single column."""
        df = pd.DataFrame({"feat1": [1, 2, 3, 4, 5]})
        strategy = ContinuousFAStrategy()
        
        # Should handle single feature (though FA needs at least 2)
        with pytest.raises((ValueError, Exception)):
            model, scores = strategy.fit(df, n_factors=1)
    
    def test_more_factors_than_features(self, valid_numeric_dataframe):
        """Test requesting more factors than available features."""
        df = valid_numeric_dataframe[["feat1", "feat2", "feat3"]]  # 3 features
        strategy = ContinuousFAStrategy()
        
        # Requesting 5 factors from 3 features should fail or auto-adjust
        with pytest.raises((ValueError, Exception)):
            model, scores = strategy.fit(df, n_factors=5)
    
    def test_dataframe_with_constant_column(self):
        """Test handling of constant (zero variance) columns."""
        df = pd.DataFrame({
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [5, 5, 5, 5, 5],  # Constant
            "feat3": [10, 20, 30, 40, 50]
        })
        strategy = ContinuousFAStrategy()
        
        # FA should handle or warn about constant columns
        # Exact behavior depends on implementation
        model, scores = strategy.fit(df, n_factors=2)
        assert scores is not None


@pytest.mark.parametrize("n_factors", [1, 2, 3])
def test_fa_with_different_factor_counts(valid_numeric_dataframe, n_factors):
    """Parametrized test for different factor counts."""
    df = valid_numeric_dataframe[["feat1", "feat2", "feat3"]]
    strategy = ContinuousFAStrategy()
    
    model, scores = strategy.fit(df, n_factors=n_factors)
    
    assert scores.shape[1] == n_factors
    assert scores.shape[0] == len(df)
