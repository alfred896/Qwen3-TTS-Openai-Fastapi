# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
FastAPI router for model management endpoints.
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Global ModelManager instance (will be set during app lifespan)
_model_manager = None


def set_model_manager(manager):
    """Set the global ModelManager instance."""
    global _model_manager
    _model_manager = manager


def get_model_manager():
    """Get the global ModelManager instance."""
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="ModelManager not initialized")
    return _model_manager


@router.get("/available")
async def get_available_models() -> Dict:
    """Get list of available models."""
    manager = get_model_manager()
    return {
        "models": manager.get_available_models(),
        "current": manager.get_current_model(),
        "loading": manager.is_loading(),
    }


@router.get("/current")
async def get_current_model() -> Dict:
    """Get currently loaded model."""
    manager = get_model_manager()
    return {
        "current": manager.get_current_model(),
        "loading": manager.is_loading(),
    }


@router.post("/switch")
async def switch_model(task_type: str) -> Dict:
    """
    Switch to a different model.

    Query params:
        task_type: Model name to load (CustomVoice, VoiceDesign, Base)

    Returns:
        {"success": bool, "current": str, "loading": bool}
    """
    manager = get_model_manager()

    if task_type not in manager.get_available_models():
        raise HTTPException(status_code=400, detail=f"Unknown model: {task_type}")

    success = await manager.load_model(task_type)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {task_type}")

    return {
        "success": True,
        "current": manager.get_current_model(),
        "loading": manager.is_loading(),
    }


@router.get("/status")
async def get_models_status() -> Dict:
    """Get detailed status of all models."""
    manager = get_model_manager()
    return {
        "available": manager.get_available_models(),
        "current": manager.get_current_model(),
        "loading": manager.is_loading(),
    }


@router.get("/voices/saved")
async def get_saved_voices() -> Dict:
    """
    Get list of saved voice profiles from voice_library.

    Returns:
        {"voices": [{"id": "...", "name": "...", "created": "...", "task_type": "..."}, ...]}
    """
    manager = get_model_manager()
    voices = manager.get_saved_voices()

    return {
        "voices": voices,
        "count": len(voices),
    }
