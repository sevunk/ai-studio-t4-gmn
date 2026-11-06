# -*- coding: utf-8 -*-
"""
Auto-Registry semua class Engine (`engines/__init__.py`)
Menyediakan akses terpusat ke 8 Engine Utama + 1 Spandrel Tiled Upscaler Engine.
"""

from pathlib import Path
from typing import Dict, List, Type

from config.catalog import COMFY_DIR
from engines.anima_engine import AnimaEngine
from engines.base import BaseEngine
from engines.chroma_engine import ChromaEngine
from engines.flux_fusion_engine import FluxFusionEngine
from engines.krea2_engine import Krea2Engine
from engines.moss_engine import MossEngine
from engines.qwen21_engine import Qwen21Engine
from engines.upscaler_engine import UpscalerEngine
from engines.wan21_engine import QUICK_MOTION_TAGS, WAN21_FPS_CHOICES, Wan21Engine
from engines.z_image_engine import ZImageEngine

ENGINE_CLASSES: Dict[str, Type[BaseEngine]] = {
    "anima": AnimaEngine,
    "moss": MossEngine,
    "qwen21": Qwen21Engine,
    "krea2": Krea2Engine,
    "chroma": ChromaEngine,
    "z_image": ZImageEngine,
    "flux_fusion": FluxFusionEngine,
    "wan21": Wan21Engine,
    "upscaler": UpscalerEngine,
}

MAIN_ENGINE_IDS: List[str] = [
    "anima",
    "moss",
    "qwen21",
    "krea2",
    "chroma",
    "z_image",
    "flux_fusion",
    "wan21",
]


def create_engine_registry(comfy_dir: Path = COMFY_DIR) -> Dict[str, BaseEngine]:
    """Menginisialisasi seluruh instance Engine dengan `comfy_dir` yang diberikan."""
    return {eng_id: cls(comfy_dir=comfy_dir) for eng_id, cls in ENGINE_CLASSES.items()}


def resolve_engine_id(id_or_display_name: str) -> str:
    """Menerjemahkan `display_name` atau `engine_id` menjadi `engine_id` kanonik."""
    key = str(id_or_display_name).strip()
    if key in ENGINE_CLASSES:
        return key
    lower = key.lower()
    for eng_id, cls in ENGINE_CLASSES.items():
        if eng_id in lower or cls.display_name.lower() == lower:
            return eng_id
    if "anima" in lower:
        return "anima"
    if "moss" in lower:
        return "moss"
    if "qwen" in lower:
        return "qwen21"
    if "krea" in lower:
        return "krea2"
    if "chroma" in lower:
        return "chroma"
    if "z-image" in lower or "z_image" in lower or "tongyi" in lower:
        return "z_image"
    if "flux" in lower or "fusion" in lower or "anibaaal" in lower:
        return "flux_fusion"
    if "wan" in lower:
        return "wan21"
    if "upscale" in lower:
        return "upscaler"
    return "anima"


__all__ = [
    "BaseEngine",
    "AnimaEngine",
    "MossEngine",
    "Qwen21Engine",
    "Krea2Engine",
    "ChromaEngine",
    "ZImageEngine",
    "FluxFusionEngine",
    "Wan21Engine",
    "UpscalerEngine",
    "ENGINE_CLASSES",
    "MAIN_ENGINE_IDS",
    "WAN21_FPS_CHOICES",
    "QUICK_MOTION_TAGS",
    "create_engine_registry",
    "resolve_engine_id",
]
