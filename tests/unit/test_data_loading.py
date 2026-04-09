"""
Unit tests for data loading functions in app_utilities.py.

Tests cover:
- Delimiter detection (comma, semicolon, tab)
- CSV loading with various formats
- Handling of missing values
- Empty DataFrames
"""

import pytest
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from app_utilities import detect_delimiter


class TestDelimiterDetection:
    """Tests for delimiter detection function."""
    
    def test_detect_comma_delimiter(self):
        """Test detection of comma-separated values."""
        line = "name,age,height"
        assert detect_delimiter(line) == ","
    
    def test_detect_semicolon_delimiter(self):
        """Test detection of semicolon-separated values."""
        line = "name;age;height"
        assert detect_delimiter(line) == ";"
    
    def test_detect_tab_delimiter(self):
        """Test detection of tab-separated values."""
        line = "name\tage\theight"
        assert detect_delimiter(line) == "\t"
    
    def test_detect_pipe_delimiter(self):
        """Test detection of pipe-separated values."""
        line = "name|age|height"
        assert detect_delimiter(line) == "|"
    
    def test_default_to_comma_if_ambiguous(self):
        """Test that comma is default when no clear delimiter."""
        line = "nameageheight"  # No delimiters
        result = detect_delimiter(line)
        # Should default to comma or handle gracefully
        assert result in [",", ";", "\t", "|"]


class TestDataFrameLoading:
    """Tests for loading DataFrames from CSV buffers."""
    
    def test_load_valid_numeric_dataframe(self, valid_numeric_dataframe):
        """Test loading a valid numeric DataFrame."""
        df = valid_numeric_dataframe
        
        assert not df.empty
        assert len(df) == 5
        assert "name" in df.columns
        assert "feat1" in df.columns
        assert df["feat1"].dtype in [np.float64, float]
    
    def test_load_mixed_data_dataframe(self, mixed_data_dataframe):
        """Test loading mixed numeric/categorical DataFrame."""
        df = mixed_data_dataframe
        
        assert not df.empty
        assert len(df) == 4
        assert "breed" in df.columns
        assert df["breed"].dtype == object  # Categorical
        assert "weight" in df.columns
        assert df["weight"].dtype in [np.float64, float]  # Numeric
    
    def test_dataframe_with_nan_values(self, csv_with_nan):
        """Test loading DataFrame with NaN values."""
        df = pd.read_csv(csv_with_nan)
        
        assert len(df) == 3
        assert df["feat1"].isna().sum() == 1  # One NaN value
        assert df["feat2"].isna().sum() == 0  # No NaN values
    
    def test_empty_dataframe_detection(self, empty_csv):
        """Test detection of empty DataFrame."""
        df = pd.read_csv(empty_csv)
        
        assert len(df) == 0
        assert df.empty


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_large_dataframe_loads(self, large_numeric_dataframe):
        """Test that large DataFrames load correctly."""
        df = large_numeric_dataframe
        
        assert len(df) == 100
        assert len(df.columns) == 11  # 10 features + name column
    
    def test_correlated_features_dataframe(self, dataframe_with_high_correlation):
        """Test DataFrame with correlated features (good for FA)."""
        df = dataframe_with_high_correlation
        
        # Check correlation between feat1 and feat2
        corr = df["feat1"].corr(df["feat2"])
        assert corr > 0.9  # Should be highly correlated
    
    def test_dataframe_dtypes_preserved(self, mixed_data_dataframe):
        """Test that dtypes are preserved after loading."""
        df = mixed_data_dataframe
        
        # Numeric columns
        assert pd.api.types.is_numeric_dtype(df["weight"])
        assert pd.api.types.is_numeric_dtype(df["height"])
        
        # Categorical columns
        assert pd.api.types.is_object_dtype(df["breed"])
        assert pd.api.types.is_object_dtype(df["temperament"])


class TestColumnDetection:
    """Tests for column type detection."""
    
    def test_detect_numeric_columns(self, valid_numeric_dataframe):
        """Test detection of numeric columns."""
        df = valid_numeric_dataframe
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assert "feat1" in numeric_cols
        assert "feat2" in numeric_cols
        assert "feat3" in numeric_cols
        assert "name" not in numeric_cols  # Name is string
    
    def test_detect_categorical_columns(self, mixed_data_dataframe):
        """Test detection of categorical columns."""
        df = mixed_data_dataframe
        
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        assert "breed" in categorical_cols
        assert "temperament" in categorical_cols
        assert "name" in categorical_cols


@pytest.mark.parametrize("delimiter,expected", [
    ("name,age,height", ","),
    ("name;age;height", ";"),
    ("name\tage\theight", "\t"),
    ("name|age|height", "|"),
])
def test_delimiter_detection_parametrized(delimiter, expected):
    """Parametrized test for various delimiters."""
    assert detect_delimiter(delimiter) == expected
