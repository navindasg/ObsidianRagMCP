from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Yields a temporary directory for config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yield config_dir


@pytest.fixture
def valid_config_dict(tmp_path):
    """Returns a minimal valid config dict with a real vault directory."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return {"vaults": [{"name": "test", "path": str(vault_dir)}]}


@pytest.fixture
def mock_ollama():
    """Patches ollama.Client to return a mock with nomic-embed-text:latest available."""
    mock_model = MagicMock()
    mock_model.model = "nomic-embed-text:latest"

    mock_response = MagicMock()
    mock_response.models = [mock_model]

    mock_client_instance = MagicMock()
    mock_client_instance.list.return_value = mock_response

    with patch("ollama.Client", return_value=mock_client_instance) as mock_cls:
        yield mock_cls
