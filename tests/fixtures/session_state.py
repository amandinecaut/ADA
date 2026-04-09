"""
Mock Streamlit session_state for testing without running Streamlit UI.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def mock_session_state():
    """
    Simulates Streamlit session_state without running Streamlit.
    Returns a dict with all default values matching app_utilities.py defaults.
    """
    state = {
        # Data storage
        "df_full": pd.DataFrame(),
        "df_filtered": pd.DataFrame(),
        "df_FA": pd.DataFrame(),
        "df_original": pd.DataFrame(),
        "df_famd": pd.DataFrame(),
        
        # Feature tracking
        "features": [],
        "features_FA": [],
        "features_famd": [],
        
        # Column mapping
        "col_mapping": {},
        "ind_col_map": None,
        
        # Entity configuration
        "entity_col": "Index",
        "entity_id": "entity",
        "article": "an",
        "selected_entity": "",
        "indice": "",
        
        # Factor Analysis
        "FA_component_dict": {},
        "FA_done": False,
        "factor_nb": 5,
        "threshold": 0.3,
        "strategy_name": "FA",
        
        # Clustering
        "num_clusters": 4,
        "u_labels": np.array([]),
        "centroids": None,
        "list_cluster_name": [],
        "list_description_cluster": [],
        
        # File uploads
        "file": None,
        "map": None,
        
        # Debug
        "debug_prompts_fa": [],
        "debug_prompts_clustering": [],
        "debug_prompts_view": [],
        "show_gpt_calls": False,
        "current_debug_context": "fa",
    }
    return state


@pytest.fixture
def mock_streamlit_module(mock_session_state, monkeypatch):
    """
    Creates a mock streamlit module with session_state.
    Use this with monkeypatch to replace 'import streamlit as st'.
    """
    from unittest.mock import MagicMock
    
    # Create mock streamlit module
    mock_st = MagicMock()
    mock_st.session_state = mock_session_state
    
    # Add common streamlit functions as mocks
    mock_st.spinner = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.info = MagicMock()
    mock_st.success = MagicMock()
    mock_st.write = MagicMock()
    
    return mock_st
