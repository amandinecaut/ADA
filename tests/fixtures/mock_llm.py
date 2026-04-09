"""
Mock LLM responses for testing without hitting real APIs.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def mock_fa_labels():
    """Pre-defined valid factor labels in 'x vs y' format."""
    return {
        "Factor 1": "calm and passive vs energetic and active",
        "Factor 2": "small and light vs large and heavy",
        "Factor 3": "independent and aloof vs friendly and social",
    }


@pytest.fixture
def mock_invalid_fa_label():
    """Label missing 'vs' separator (failure case)."""
    return "high energy levels without vs"


@pytest.fixture
def mock_cluster_names():
    """Valid cluster names (short, 1-2 words)."""
    return ["Active Companions", "Calm Protectors", "Small Energetic"]


@pytest.fixture
def mock_cluster_descriptions():
    """Valid cluster descriptions (3 sentences each)."""
    return [
        "This cluster represents highly active entities. They excel in energy and engagement. However, they may lack calmness.",
        "This cluster shows calm and protective traits. Members are stable and reliable. They tend to be less energetic.",
        "This cluster consists of small but energetic entities. They balance size with activity. They are moderately social.",
    ]


@pytest.fixture
def mock_qanda_dataframe():
    """Valid Q&A dataset for few-shot learning."""
    return pd.DataFrame({
        "User": [
            "What does friendly mean?",
            "What is energetic?",
            "What does calm indicate?"
        ],
        "Assistant": [
            "Friendly entities are social and welcoming.",
            "Energetic entities are active and lively.",
            "Calm entities are peaceful and composed."
        ]
    })


class MockModelHandler:
    """
    Replaces wordalisation.ModelHandler for testing without real API calls.
    Returns predefined responses based on message content patterns.
    """
    
    def __init__(self, mock_responses=None):
        """
        Args:
            mock_responses: Dict mapping response types to mock text
        """
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.call_history = []
    
    def get_generate(self, msgs, max_output_token):
        """
        Return predefined response based on message content.
        Inspects the messages to determine what type of response is expected.
        """
        self.call_count += 1
        self.call_history.append(msgs)
        
        # Convert messages to string for pattern matching
        if isinstance(msgs, dict):
            msgs_str = str(msgs.get("content", "")) + str(msgs.get("history", ""))
        else:
            msgs_str = str(msgs)
        
        msgs_str_lower = msgs_str.lower()
        
        # Determine response type based on content
        if "name the factor" in msgs_str_lower or "x vs y" in msgs_str_lower:
            return self.mock_responses.get("fa_label", "calm and passive vs energetic and active")
        
        elif "describe cluster" in msgs_str_lower or "provide a description of a cluster" in msgs_str_lower:
            return self.mock_responses.get(
                "cluster_desc",
                "This cluster represents entities with high energy. They are active and engaged. They tend to be social."
            )
        
        elif "label for the cluster" in msgs_str_lower or "cluster label must be short" in msgs_str_lower:
            return self.mock_responses.get("cluster_name", "Energetic Group")
        
        elif "what does it mean" in msgs_str_lower:
            return self.mock_responses.get(
                "qanda",
                "This trait indicates a particular characteristic of the entity."
            )
        
        elif "summary" in msgs_str_lower or "describe" in msgs_str_lower:
            return self.mock_responses.get(
                "entity_summary",
                "This entity shows strong characteristics. It has notable strengths. Overall, it performs well."
            )
        
        else:
            return self.mock_responses.get("default", "Mock LLM response")
    
    def get_model(self):
        """Return mock model tuple."""
        mock_model = MagicMock()
        return [(mock_model, "gemini")]


@pytest.fixture
def mock_model_handler():
    """
    Returns a MockModelHandler instance with default responses.
    Use with monkeypatch to replace the real ModelHandler.
    """
    return MockModelHandler({
        "fa_label": "small and calm vs large and energetic",
        "cluster_desc": "Cluster members show balanced traits. They have moderate energy. They are adaptable.",
        "cluster_name": "Balanced",
        "qanda": "This indicates a moderate level of the trait.",
        "entity_summary": "The entity displays well-rounded characteristics. It has clear strengths. It shows potential for growth."
    })


@pytest.fixture
def mock_model_handler_with_invalid_label():
    """
    Returns a MockModelHandler that returns invalid labels (missing 'vs').
    Used to test error handling.
    """
    return MockModelHandler({
        "fa_label": "just one side without separator",  # Invalid: no "vs"
        "cluster_desc": "Valid description here.",
        "cluster_name": "ValidName",
    })


@pytest.fixture
def mock_model_handler_raises_exception():
    """
    Returns a MockModelHandler that raises exceptions (simulates API failures).
    """
    handler = MockModelHandler()
    
    def failing_get_generate(*args, **kwargs):
        raise Exception("API timeout - ResourceExhausted")
    
    handler.get_generate = failing_get_generate
    return handler
