"""
Test dataset generators for ADA pipeline testing.
"""

import pytest
import pandas as pd
import numpy as np
import io
import json


@pytest.fixture
def valid_numeric_csv():
    """Pure numerical dataset for standard FA (5 entities, 3 features)."""
    data = {
        "name": ["Entity_A", "Entity_B", "Entity_C", "Entity_D", "Entity_E"],
        "feat1": [1.2, 2.3, 3.1, 4.5, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feat3": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer


@pytest.fixture
def valid_numeric_dataframe():
    """Returns DataFrame directly instead of CSV buffer."""
    data = {
        "name": ["Entity_A", "Entity_B", "Entity_C", "Entity_D", "Entity_E"],
        "feat1": [1.2, 2.3, 3.1, 4.5, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feat3": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mixed_data_csv():
    """Mixed numerical + categorical for FAMD (4 entities)."""
    data = {
        "name": ["Dog1", "Dog2", "Dog3", "Dog4"],
        "weight": [10.5, 20.2, 15.3, 12.0],
        "height": [30.0, 40.0, 35.0, 32.0],
        "breed": ["Labrador", "Bulldog", "Labrador", "Poodle"],
        "temperament": ["Friendly", "Calm", "Friendly", "Energetic"],
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer


@pytest.fixture
def mixed_data_dataframe():
    """Returns mixed data DataFrame directly."""
    data = {
        "name": ["Dog1", "Dog2", "Dog3", "Dog4"],
        "weight": [10.5, 20.2, 15.3, 12.0],
        "height": [30.0, 40.0, 35.0, 32.0],
        "breed": ["Labrador", "Bulldog", "Labrador", "Poodle"],
        "temperament": ["Friendly", "Calm", "Friendly", "Energetic"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def csv_with_nan():
    """Dataset with missing values (3 entities, 1 has NaN)."""
    data = {
        "name": ["A", "B", "C"],
        "feat1": [1.0, np.nan, 3.0],
        "feat2": [10.0, 20.0, 30.0],
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer


@pytest.fixture
def empty_csv():
    """Empty CSV file with headers only."""
    return io.StringIO("col1,col2\n")


@pytest.fixture
def invalid_delimiter_csv():
    """CSV with semicolon delimiter instead of comma."""
    content = "name;feat1;feat2\nA;1.2;10\nB;2.3;20\nC;3.1;30\n"
    return io.StringIO(content)


@pytest.fixture
def valid_json_mapping(tmp_path):
    """Valid JSON column mapping file."""
    mapping = {"feat1": "Feature One", "feat2": "Feature Two", "feat3": "Feature Three"}
    file_path = tmp_path / "map.json"
    with open(file_path, "w") as f:
        json.dump(mapping, f)
    return str(file_path)


@pytest.fixture
def valid_excel_mapping(tmp_path):
    """Valid Excel column mapping with Key/Value columns."""
    df = pd.DataFrame({
        "Key": ["feat1", "feat2", "feat3"],
        "Value": ["Feature One", "Feature Two", "Feature Three"]
    })
    file_path = tmp_path / "map.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture
def malformed_excel_mapping(tmp_path):
    """Excel mapping missing 'Key' column (should fail)."""
    df = pd.DataFrame({
        "Column": ["feat1"],
        "Mapping": ["Feature One"]
    })
    file_path = tmp_path / "bad_map.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture
def large_numeric_dataframe():
    """Large dataset with 100 rows and 10 features for performance testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    data = {"name": [f"Entity_{i}" for i in range(n_samples)]}
    for i in range(n_features):
        data[f"feat{i}"] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def dataframe_with_high_correlation():
    """
    Dataset where features are highly correlated (good for FA).
    feat2 = 2 * feat1, feat3 = feat1 + noise
    """
    np.random.seed(42)
    n_samples = 50
    
    feat1 = np.random.randn(n_samples)
    feat2 = 2 * feat1 + np.random.randn(n_samples) * 0.1
    feat3 = feat1 + np.random.randn(n_samples) * 0.2
    feat4 = np.random.randn(n_samples)  # Independent
    
    data = {
        "name": [f"Entity_{i}" for i in range(n_samples)],
        "feat1": feat1,
        "feat2": feat2,
        "feat3": feat3,
        "feat4": feat4,
    }
    
    return pd.DataFrame(data)
