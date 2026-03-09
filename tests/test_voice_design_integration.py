"""Integration tests for voice design functionality."""

import pytest
from pathlib import Path


def test_voice_design_profile_lifecycle():
    """Test creating, retrieving, and listing voice design profiles."""
    from api.services.voice_profile_manager import VoiceProfileManager

    manager = VoiceProfileManager()

    # Create
    meta = manager.create_voice_design_profile(
        design_name="Professional Female",
        instruct="清晰女声，温和的语气",
        language="Chinese",
    )
    assert meta["task_type"] == "VoiceDesign"
    assert meta["name"] == "Professional Female"
    assert meta["instruct"] == "清晰女声，温和的语气"

    # Retrieve
    retrieved = manager.get_voice_design_profile("Professional Female")
    assert retrieved is not None
    assert retrieved["name"] == "Professional Female"
    assert retrieved["task_type"] == "VoiceDesign"

    # List
    designs = manager.list_voice_design_profiles()
    assert len(designs) > 0
    assert any(d["name"] == "Professional Female" for d in designs)

    # Cleanup
    manager.delete_profile("Professional Female")
    assert manager.get_voice_design_profile("Professional Female") is None


def test_voice_design_profile_persistence():
    """Test that voice design profiles are persisted on disk."""
    from api.services.voice_profile_manager import VoiceProfileManager
    import json
    from pathlib import Path

    manager = VoiceProfileManager()

    # Create profile
    design_name = "Test Design Persistence"
    meta = manager.create_voice_design_profile(
        design_name=design_name,
        instruct="Test instruction for persistence",
        language="English",
    )

    profile_id = meta["profile_id"]
    profile_dir = manager.profiles_dir / profile_id
    meta_file = profile_dir / "meta.json"

    # Verify file exists
    assert meta_file.exists(), f"Meta file not found at {meta_file}"

    # Verify content
    with open(meta_file, "r") as f:
        saved_meta = json.load(f)

    assert saved_meta["name"] == design_name
    assert saved_meta["instruct"] == "Test instruction for persistence"
    assert saved_meta["task_type"] == "VoiceDesign"

    # Cleanup
    manager.delete_profile(design_name)


def test_voice_clone_profile_with_audio():
    """Test creating and retrieving voice clone profiles with audio."""
    from api.services.voice_profile_manager import VoiceProfileManager
    import tempfile
    import os

    manager = VoiceProfileManager()

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF")
        f.write(b"\x24\x00\x00\x00")
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(b"\x10\x00\x00\x00")
        f.write(b"\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00")
        f.write(b"data")
        f.write(b"\x00\x00\x00\x00")
        temp_path = f.name

    try:
        # Create profile
        meta = manager.create_voice_clone_profile(
            voice_name="Integration Test Voice",
            reference_audio_path=temp_path,
            reference_text="This is a test voice",
            language="English",
        )

        assert meta["task_type"] == "VoiceCloning"
        assert meta["name"] == "Integration Test Voice"
        assert meta["ref_text"] == "This is a test voice"

        # Retrieve
        retrieved = manager.get_voice_profile("Integration Test Voice")
        assert retrieved is not None
        assert retrieved["task_type"] == "VoiceCloning"

        # Get reference audio path
        audio_path = manager.get_reference_audio_path("Integration Test Voice")
        assert audio_path is not None
        assert Path(audio_path).exists()

        # Cleanup
        manager.delete_profile("Integration Test Voice")

    finally:
        os.unlink(temp_path)


def test_mixed_profile_listing():
    """Test that voice design and clone profiles are correctly categorized."""
    from api.services.voice_profile_manager import VoiceProfileManager
    import tempfile
    import os

    manager = VoiceProfileManager()

    # Create a design profile
    design_meta = manager.create_voice_design_profile(
        design_name="Mixed Test Design",
        instruct="Test design",
    )

    # Create a clone profile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF\x24\x00\x00\x00WAVE")
        temp_path = f.name

    try:
        clone_meta = manager.create_voice_clone_profile(
            voice_name="Mixed Test Clone",
            reference_audio_path=temp_path,
            language="English",
        )

        # List all profiles
        all_profiles = manager.list_voice_profiles()
        designs = manager.list_voice_design_profiles()
        clones = manager.list_voice_clone_profiles()

        # Verify counts
        design_names = [d["name"] for d in designs]
        clone_names = [c["name"] for c in clones]

        assert "Mixed Test Design" in design_names
        assert "Mixed Test Clone" in clone_names

        # Cleanup
        manager.delete_profile("Mixed Test Design")
        manager.delete_profile("Mixed Test Clone")

    finally:
        os.unlink(temp_path)


def test_profile_id_generation_consistency():
    """Test that profile IDs are generated consistently for the same name."""
    from api.services.voice_profile_manager import VoiceProfileManager

    manager = VoiceProfileManager()

    name1 = "Consistent Name Test"
    id1 = manager._generate_profile_id(name1)
    id2 = manager._generate_profile_id(name1)

    assert id1 == id2, "Profile IDs should be consistent for the same name"

    # Different names should produce different IDs
    name2 = "Different Name Test"
    id3 = manager._generate_profile_id(name2)

    assert id1 != id3, "Different names should produce different profile IDs"
