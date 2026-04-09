"""
Unit tests for utility functions.

Tests cover:
- split_qualities (splitting "x vs y" labels)
- choose_article (a/an selection)
- Other helper functions
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from wordalisation import ClusterWordalisation


class TestSplitQualities:
    """Tests for split_qualities function (critical for label parsing)."""
    
    def test_split_qualities_valid_label(self):
        """Test splitting a valid 'x vs y' label."""
        label = "calm and passive vs energetic and active"
        left, right = ClusterWordalisation.split_qualities(label)
        
        assert left == "calm and passive"
        assert right == "energetic and active"
    
    def test_split_qualities_case_insensitive(self):
        """Test that 'vs' matching is case-insensitive."""
        label = "strong VS weak"
        left, right = ClusterWordalisation.split_qualities(label)
        
        assert left == "strong"
        assert right == "weak"
    
    def test_split_qualities_with_period(self):
        """Test splitting with 'vs.' (period after vs)."""
        label = "small vs. large"
        left, right = ClusterWordalisation.split_qualities(label)
        
        assert left == "small"
        assert right == "large"
    
    def test_split_qualities_with_extra_whitespace(self):
        """Test handling of extra whitespace around 'vs'."""
        label = "low   vs   high"
        left, right = ClusterWordalisation.split_qualities(label)
        
        assert left == "low"
        assert right == "high"
    
    def test_split_qualities_missing_vs_raises_error(self):
        """Test that missing 'vs' raises ValueError (FAILURE POINT #3)."""
        label = "just one side without separator"
        
        with pytest.raises(ValueError, match="must contain 'vs'"):
            ClusterWordalisation.split_qualities(label)
    
    def test_split_qualities_multiple_vs_takes_first(self):
        """Test handling of multiple 'vs' in label (takes first split)."""
        label = "small vs medium vs large"
        left, right = ClusterWordalisation.split_qualities(label)
        
        # Should split on first 'vs' only
        assert left == "small"
        assert "medium" in right or "medium vs large" in right


class TestDescribeLevel:
    """Tests for describe_level and threshold-based descriptions."""
    
    def test_describe_level_extreme_low(self):
        """Test description for extremely low values (< -2)."""
        from wordalisation import CreateWordalisation
        result = CreateWordalisation.describe(
            thresholds=[-2, -1, -0.5, 0.5, 1, 2],
            words=[
                "is extremely low on ",
                "is very low on ",
                "is quite low on ",
                "is average on ",
                "is quite high on ",
                "is very high on ",
                "is extremely high on "
            ],
            value=-2.5
        )
        assert result == "is extremely low on "
    
    def test_describe_level_average(self):
        """Test description for average values (-0.5 to 0.5)."""
        from wordalisation import CreateWordalisation
        result = CreateWordalisation.describe(
            thresholds=[-2, -1, -0.5, 0.5, 1, 2],
            words=[
                "is extremely low on ",
                "is very low on ",
                "is quite low on ",
                "is average on ",
                "is quite high on ",
                "is very high on ",
                "is extremely high on "
            ],
            value=0.0
        )
        assert result == "is average on "
    
    def test_describe_level_extreme_high(self):
        """Test description for extremely high values (> 2)."""
        from wordalisation import CreateWordalisation
        result = CreateWordalisation.describe(
            thresholds=[-2, -1, -0.5, 0.5, 1, 2],
            words=[
                "is extremely low on ",
                "is very low on ",
                "is quite low on ",
                "is average on ",
                "is quite high on ",
                "is very high on ",
                "is extremely high on "
            ],
            value=2.5
        )
        assert result == "is extremely high on "


@pytest.mark.parametrize("label,expected_left,expected_right", [
    ("calm vs active", "calm", "active"),
    ("small and light vs large and heavy", "small and light", "large and heavy"),
    ("low VS high", "low", "high"),
    ("weak vs. strong", "weak", "strong"),
])
def test_split_qualities_parametrized(label, expected_left, expected_right):
    """Parametrized test for various valid label formats."""
    left, right = ClusterWordalisation.split_qualities(label)
    assert left == expected_left
    assert right == expected_right


@pytest.mark.parametrize("invalid_label", [
    "no separator here",
    "missing the word",
    "only-one-side",
    "",
])
def test_split_qualities_invalid_labels(invalid_label):
    """Parametrized test for invalid labels (should raise ValueError)."""
    with pytest.raises(ValueError, match="must contain 'vs'"):
        ClusterWordalisation.split_qualities(invalid_label)
