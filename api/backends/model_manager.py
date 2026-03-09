# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Model Manager for Qwen3-TTS.

Handles model lifecycle: download, load, unload with GPU memory management.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of multiple TTS models.

    Features:
    - Download all models from HuggingFace at initialization
    - Load/unload models with GPU memory management
    - Track currently loaded model
    - Expose model state via properties
    """

    def __init__(
        self,
        models_config: Dict[str, str],
        cache_dir: Path = Path.home() / ".cache" / "huggingface" / "hub",
        voice_library_dir: Path = Path("./voice_library"),
    ):
        """
        Initialize ModelManager.

        Args:
            models_config: Dict mapping task_type -> model_id (e.g., {"CustomVoice": "Qwen/..."})
            cache_dir: HuggingFace cache directory
            voice_library_dir: Directory for storing voice profiles (pkl files)
        """
        self.models_config = models_config
        self.cache_dir = Path(cache_dir).resolve()
        self.voice_library_dir = Path(voice_library_dir).resolve()

        self._current_model: Optional[str] = None
        self._is_loading: bool = False
        self._model_instances: Dict[str, Any] = {}
        self._load_lock = asyncio.Lock()  # Lock per async safety

        logger.info(f"ModelManager initialized with {len(models_config)} models")
        logger.info(f"Cache dir: {self.cache_dir}")
        logger.info(f"Voice library dir: {self.voice_library_dir}")

    def get_available_models(self) -> List[str]:
        """Return list of available model names."""
        return list(self.models_config.keys())

    def get_current_model(self) -> Optional[str]:
        """Return currently loaded model name, or None if no model loaded."""
        return self._current_model

    def is_loading(self) -> bool:
        """Return whether a model is currently loading."""
        return self._is_loading

    async def download_all_models(self) -> Dict[str, bool]:
        """
        Download all configured models from HuggingFace.

        Returns:
            Dict mapping task_type -> success (bool)
        """
        results = {}
        logger.info(f"Starting download of {len(self.models_config)} models...")

        # Ensure cache dir exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for task_type, model_id in self.models_config.items():
            try:
                logger.info(f"Downloading {task_type} ({model_id})...")

                # Use HuggingFace API to download model
                from huggingface_hub import snapshot_download

                # Run download in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    snapshot_download,
                    model_id,
                    str(self.cache_dir),  # cache_dir
                )

                results[task_type] = True
                logger.info(f"Downloaded {task_type}")
            except (OSError, IOError, Exception) as e:
                logger.error(f"Failed to download {task_type}: {e}")
                results[task_type] = False

        # Log summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Download complete: {successful}/{len(results)} models")

        return results

    async def load_model(self, task_type: str) -> bool:
        """
        Load specified model into memory.

        If a different model is already loaded, unload it first.

        Args:
            task_type: Model name to load (e.g., "CustomVoice")

        Returns:
            True if successful, False otherwise
        """
        if task_type not in self.models_config:
            logger.error(f"Unknown model: {task_type}")
            return False

        async with self._load_lock:  # Protection with lock
            if self._current_model == task_type:
                logger.info(f"Model {task_type} already loaded")
                return True

            self._is_loading = True

        try:
            # Unload current model if different
            if self._current_model is not None:
                await self.unload_model()

            logger.info(f"Loading model: {task_type}")
            # Placeholder: actual load logic will be implemented
            self._current_model = task_type
            logger.info(f"Loaded {task_type}")
            return True
        except (OSError, IOError, Exception) as e:
            logger.error(f"Failed to load {task_type}: {e}")
            return False
        finally:
            self._is_loading = False

    async def unload_model(self) -> bool:
        """
        Unload current model and release GPU memory.

        Returns:
            True if successful
        """
        if self._current_model is None:
            return True

        try:
            logger.info(f"Unloading model: {self._current_model}")
            # Placeholder: actual unload logic will be implemented

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

            return True
        except (OSError, IOError, Exception) as e:
            logger.error(f"Failed to unload model: {e}")
            return False
        finally:
            # Reset state AFTER try-except block
            self._current_model = None
            logger.info("Model unloaded")

    def get_saved_voices(self) -> List[Dict]:
        """
        Get list of saved voice profiles from voice_library.

        Returns:
            List of dicts with voice metadata: {"id": "...", "name": "...", ...}
        """
        voices = []

        if not self.voice_library_dir.exists():
            logger.warning(f"Voice library dir does not exist: {self.voice_library_dir}")
            return voices

        # Find all .pkl files and their .json metadata
        for pkl_file in self.voice_library_dir.glob("*.pkl"):
            json_file = pkl_file.with_suffix(".json")

            # Build voice entry
            voice_entry = {
                "id": pkl_file.stem,
                "pkl_path": str(pkl_file),
            }

            # Try to load metadata from .json sidecar
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        metadata = json.load(f)
                    voice_entry.update(metadata)
                except (OSError, IOError, Exception) as e:
                    logger.warning(f"Failed to load metadata for {pkl_file.stem}: {e}")

            voices.append(voice_entry)

        logger.info(f"Found {len(voices)} saved voice profiles")
        return voices
