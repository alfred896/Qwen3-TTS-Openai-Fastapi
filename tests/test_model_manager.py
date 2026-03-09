import pytest
from pathlib import Path
from api.backends.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Create a ModelManager instance for testing."""
    return ModelManager(
        models_config={
            "CustomVoice": "Qwen/Qwen3-TTS-CustomVoice",
            "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-Voice-Design",
            "Base": "Qwen/Qwen3-TTS",
        },
        cache_dir=Path(".cache_test"),
        voice_library_dir=Path("./voice_library_test"),
    )


def test_model_manager_init(model_manager):
    """Test ModelManager initializes correctly."""
    assert model_manager is not None
    assert model_manager.get_current_model() is None  # No model loaded yet
    assert model_manager.get_available_models() == ["CustomVoice", "VoiceDesign", "Base"]


def test_get_available_models(model_manager):
    """Test getting list of available models."""
    models = model_manager.get_available_models()
    assert isinstance(models, list)
    assert len(models) == 3
    assert "CustomVoice" in models
    assert "VoiceDesign" in models
    assert "Base" in models


def test_model_manager_state(model_manager):
    """Test model manager tracks current model state."""
    assert model_manager.get_current_model() is None
    assert model_manager.is_loading() == False
