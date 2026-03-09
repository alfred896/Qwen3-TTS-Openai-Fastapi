"""Manage voice profiles: storage, loading, and lifecycle."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import pickle
import hashlib


class VoiceProfileManager:
    """Manage voice cloning and design profiles with pkl storage."""

    def __init__(self, voice_library_dir: str = "./voice_library"):
        self.voice_library_dir = Path(voice_library_dir)
        self.profiles_dir = self.voice_library_dir / "profiles"
        self.pkl_dir = self.voice_library_dir / "pkl_profiles"

        # Create directories if needed
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.pkl_dir.mkdir(parents=True, exist_ok=True)

    def create_voice_clone_profile(
        self,
        voice_name: str,
        reference_audio_path: str,
        reference_text: Optional[str] = None,
        language: str = "English",
        use_icl_mode: bool = True,
        x_vector: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Create and save a voice cloning profile.

        Args:
            voice_name: User-friendly name for the voice
            reference_audio_path: Path to reference audio file
            reference_text: Optional transcript (required for ICL mode)
            language: Language of the voice
            use_icl_mode: Use ICL (in-context learning) mode if transcript provided
            x_vector: Optional pre-computed speaker embedding (pkl bytes)

        Returns:
            Profile metadata dict
        """
        profile_id = self._generate_profile_id(voice_name)
        profile_dir = self.profiles_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Save reference audio
        import shutil

        ref_audio_dest = profile_dir / "reference.wav"
        shutil.copy(reference_audio_path, ref_audio_dest)

        # Save x-vector if provided
        if x_vector:
            x_vec_path = profile_dir / "x_vector.pkl"
            with open(x_vec_path, "wb") as f:
                pickle.dump(x_vector, f)

        # Save metadata
        meta = {
            "name": voice_name,
            "profile_id": profile_id,
            "ref_audio_filename": "reference.wav",
            "ref_text": reference_text,
            "x_vector_only_mode": not reference_text,
            "use_icl_mode": use_icl_mode and reference_text is not None,
            "language": language,
            "task_type": "VoiceCloning",
            "created_at": datetime.utcnow().isoformat(),
        }

        meta_path = profile_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def create_voice_design_profile(
        self,
        design_name: str,
        instruct: str,
        language: str = "English",
    ) -> Dict[str, Any]:
        """Create a saved voice design profile.

        Args:
            design_name: User-friendly name for this design
            instruct: Voice instruction (e.g., "清晰女声，温和语气")
            language: Language of the design

        Returns:
            Profile metadata
        """
        profile_id = self._generate_profile_id(design_name)
        profile_dir = self.profiles_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Save design metadata (no audio file)
        meta = {
            "name": design_name,
            "profile_id": profile_id,
            "instruct": instruct,
            "language": language,
            "task_type": "VoiceDesign",
            "created_at": datetime.utcnow().isoformat(),
        }

        meta_path = profile_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def get_voice_profile(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Load voice profile metadata by name."""
        profile_id = self._generate_profile_id(voice_name)
        profile_dir = self.profiles_dir / profile_id

        if not profile_dir.exists():
            return None

        meta_path = profile_dir / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return meta

    def get_voice_design_profile(self, design_name: str) -> Optional[Dict[str, Any]]:
        """Load voice design profile by name."""
        profile_id = self._generate_profile_id(design_name)
        profile_dir = self.profiles_dir / profile_id

        if not profile_dir.exists():
            return None

        meta_path = profile_dir / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Verify it's a VoiceDesign profile
        if meta.get("task_type") != "VoiceDesign":
            return None

        return meta

    def get_reference_audio_path(self, voice_name: str) -> Optional[str]:
        """Get path to reference audio for a voice."""
        profile = self.get_voice_profile(voice_name)
        if not profile:
            return None

        profile_id = profile["profile_id"]
        audio_path = self.profiles_dir / profile_id / profile["ref_audio_filename"]

        return str(audio_path) if audio_path.exists() else None

    def get_x_vector(self, voice_name: str) -> Optional[any]:
        """Load saved x-vector (speaker embedding) for a voice."""
        profile = self.get_voice_profile(voice_name)
        if not profile:
            return None

        profile_id = profile["profile_id"]
        x_vec_path = self.profiles_dir / profile_id / "x_vector.pkl"

        if not x_vec_path.exists():
            return None

        with open(x_vec_path, "rb") as f:
            x_vector = pickle.load(f)

        return x_vector

    def list_voice_profiles(self) -> list:
        """List all available voice profiles."""
        profiles = []
        if not self.profiles_dir.exists():
            return profiles

        for profile_dir in self.profiles_dir.iterdir():
            if profile_dir.is_dir():
                meta_path = profile_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    profiles.append(meta)

        return profiles

    def list_voice_design_profiles(self) -> list:
        """List all voice design profiles."""
        all_profiles = self.list_voice_profiles()
        return [p for p in all_profiles if p.get("task_type") == "VoiceDesign"]

    def list_voice_clone_profiles(self) -> list:
        """List all voice clone profiles."""
        all_profiles = self.list_voice_profiles()
        return [p for p in all_profiles if p.get("task_type") == "VoiceCloning"]

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile by name.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if deleted, False if not found
        """
        profile_id = self._generate_profile_id(profile_name)
        profile_dir = self.profiles_dir / profile_id

        if not profile_dir.exists():
            return False

        import shutil

        shutil.rmtree(profile_dir)
        return True

    def _generate_profile_id(self, voice_name: str) -> str:
        """Generate a unique profile ID from voice name."""
        # Simple slug from name + hash for uniqueness
        slug = voice_name.lower().replace(" ", "_").replace("/", "_")
        hash_suffix = hashlib.md5(voice_name.encode()).hexdigest()[:8]
        return f"{slug}_{hash_suffix}"
