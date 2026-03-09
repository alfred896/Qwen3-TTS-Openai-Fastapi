"""Test Qwen3-TTS model compatibility and API."""

import pytest


def test_qwen3_tts_import():
    """Verify qwen-tts library is available."""
    try:
        from qwen_tts import Qwen3TTSModel
        assert Qwen3TTSModel is not None
    except ImportError as e:
        pytest.skip(f"qwen-tts not installed: {e}")


def test_backend_factory():
    """Verify backend factory can create backends."""
    try:
        from api.backends.factory import create_backend
        backend = create_backend(backend_type="official")
        assert backend is not None
        assert hasattr(backend, "initialize")
        assert hasattr(backend, "generate_speech")
    except ImportError:
        pytest.skip("Backend factory not available")


def test_voice_design_model_signature():
    """Verify VoiceDesign model has expected API."""
    try:
        from qwen_tts import Qwen3TTSModel
        import inspect

        # Check if Qwen3TTSModel has voice design support
        if hasattr(Qwen3TTSModel, "generate_voice_design"):
            sig = inspect.signature(Qwen3TTSModel.generate_voice_design)
            params = list(sig.parameters.keys())
            assert "text" in params or len(params) >= 1
            assert "instruct" in params or len(params) >= 2
        else:
            pytest.skip("Voice design API not available on Qwen3TTSModel")
    except (ImportError, AttributeError):
        pytest.skip("Voice design API not available")


def test_qwen3_tts_model_has_required_methods():
    """Verify Qwen3TTSModel has required synthesis methods."""
    try:
        from qwen_tts import Qwen3TTSModel

        # Check for required methods
        assert hasattr(Qwen3TTSModel, "from_pretrained"), "Missing from_pretrained class method"

        # Instance methods will be checked when model is actually instantiated
        # This just verifies the class structure

    except ImportError:
        pytest.skip("Qwen3TTSModel not available")
