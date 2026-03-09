import pytest
from pathlib import Path
from api.backends.model_manager import ModelManager


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance for testing."""
    return ModelManager(
        models_config={
            "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        },
        cache_dir=tmp_path / "cache",
        voice_library_dir=tmp_path / "voice_library",
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


@pytest.mark.asyncio
async def test_load_model_success(model_manager):
    """Test loading a model."""
    success = await model_manager.load_model("CustomVoice")
    assert success is True
    assert model_manager.get_current_model() == "CustomVoice"
    assert model_manager.is_loading() is False


@pytest.mark.asyncio
async def test_load_model_unknown(model_manager):
    """Test loading unknown model fails."""
    success = await model_manager.load_model("UnknownModel")
    assert success is False
    assert model_manager.get_current_model() is None


@pytest.mark.asyncio
async def test_unload_model(model_manager):
    """Test unloading a model."""
    await model_manager.load_model("CustomVoice")
    success = await model_manager.unload_model()
    assert success is True
    assert model_manager.get_current_model() is None


@pytest.mark.asyncio
async def test_load_different_model_unloads_previous(model_manager):
    """Test loading a different model unloads the previous one."""
    await model_manager.load_model("CustomVoice")
    assert model_manager.get_current_model() == "CustomVoice"

    await model_manager.load_model("Base")
    assert model_manager.get_current_model() == "Base"


@pytest.mark.asyncio
async def test_download_all_models(model_manager, monkeypatch):
    """Test downloading all models from HuggingFace."""
    # Mock the HF snapshot_download to avoid actual downloads
    def mock_snapshot_download(*args, **kwargs):
        return "mocked_path"

    # Patch huggingface_hub.snapshot_download
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        mock_snapshot_download,
    )

    results = await model_manager.download_all_models()
    assert isinstance(results, dict)
    assert len(results) == 3
    assert all(results.values())  # All should be True


@pytest.mark.asyncio
async def test_load_model_switches_env(model_manager, monkeypatch):
    """Test that load_model sets TTS_MODEL_ID env var and uses backend factory."""
    # Mock get_backend to track calls
    mock_backend = type('obj', (object,), {'get_backend_name': lambda: 'test_backend'})()
    get_backend_called = []

    def mock_get_backend():
        get_backend_called.append(True)
        return mock_backend

    monkeypatch.setattr("api.backends.factory.get_backend", mock_get_backend)

    success = await model_manager.load_model("CustomVoice")
    assert success is True
    assert get_backend_called  # get_backend deve essere chiamato
    assert model_manager.get_current_model() == "CustomVoice"
    assert model_manager.is_loading() is False
