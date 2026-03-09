import pytest
import httpx
from pathlib import Path


@pytest.fixture
def api_client():
    """Create HTTP client for API testing."""
    return httpx.Client(base_url="http://localhost:8880", timeout=30)


def test_model_switching_workflow(api_client):
    """
    E2E test: Model switching and voice management workflow.

    Steps:
    1. Get available models
    2. Switch to each model
    3. Load voices list
    4. Verify each step
    """
    # Step 1: Get available models
    response = api_client.get("/v1/models/available")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 3
    assert set(data["models"]) == {"CustomVoice", "VoiceDesign", "Base"}
    assert data["current"] is not None  # One model should be loaded
    assert data["loading"] is False

    # Step 2: Get initial model
    initial_model = data["current"]
    assert initial_model in ["CustomVoice", "VoiceDesign", "Base"]

    # Step 3: Switch to each model
    for target_model in ["CustomVoice", "VoiceDesign", "Base"]:
        response = api_client.post(f"/v1/models/switch?task_type={target_model}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["current"] == target_model
        assert data["loading"] is False

    # Step 4: Get current model status
    response = api_client.get("/v1/models/current")
    assert response.status_code == 200
    data = response.json()
    assert data["current"] == "Base"  # Last model we switched to
    assert data["loading"] is False

    # Step 5: Get voices list
    response = api_client.get("/v1/models/voices/saved")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert "count" in data
    assert isinstance(data["voices"], list)
    assert isinstance(data["count"], int)
    assert data["count"] == len(data["voices"])


def test_invalid_model_switch(api_client):
    """Test that switching to invalid model returns error."""
    response = api_client.post("/v1/models/switch?task_type=InvalidModel")
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_model_status_endpoint(api_client):
    """Test getting detailed model status."""
    response = api_client.get("/v1/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "current" in data
    assert "loading" in data
    assert isinstance(data["available"], list)
    assert len(data["available"]) == 3
