# -*- coding: utf-8 -*-
"""
7. Memory-Efficient Spandrel Tiled Upscaler (`engines/upscaler_engine.py`)
Mendukung:
- Mode Standalone (2–4 detik di T4, <350 MB VRAM)
- Mode Inline (ditempelkan otomatis setelah `VAEDecode` pada ke-5 engine gambar)
Model yang didukung:
- `4x-UltraSharp.safetensors` (Default Foto/Realisme)
- `4x-AnimeSharp.safetensors` (Default Anime/Ilustrasi Detail)
- `RealESRGAN_x4plus_anime_6B.pth` (Ultra-Light 17.9 MB 6-Block RRDB)
Faktor Skala Akhir (Lanczos downscale/preserve dari 4x): `1.5x`, `2.0x`, `4.0x`.
"""

from typing import Any, Dict, List

from config.catalog import UPSCALER_CATALOG
from engines.base import BaseEngine


class UpscalerEngine(BaseEngine):
    """Engine Spandrel Tiled Upscaler untuk mode Standalone maupun Inline Post-Render."""

    engine_id = "upscaler"
    display_name = "🔍 Spandrel Tiled Upscaler (4x-UltraSharp · 4x-AnimeSharp · RealESRGAN 6B)"
    is_video = False
    supports_ref_images = True
    supports_negative_prompt = False

    SUPPORTED_SCALES = (1.5, 2.0, 4.0)

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "upscaler_model": "4x-UltraSharp.safetensors",
            "upscale_factor": 2.0,
            "upscale_choices": list(UPSCALER_CATALOG.keys()),
            "scale_choices": ["1.5x", "2.0x", "4.0x"],
            "steps": 1,
            "cfg": 1.0,
            "shift": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "aspect_ratios": {"Original": (1024, 1024)},
            "default_aspect": "Original",
            "presets": {},
            "placeholder": "Standalone Tiled Upscaler tidak memerlukan prompt teks.",
        }

    @staticmethod
    def parse_scale_factor(raw_scale: Any) -> float:
        """Menerjemahkan string `'1.5x'` / `'2.0x'` / `'4.0x'` atau float menjadi angka valid."""
        if isinstance(raw_scale, str):
            cleaned = raw_scale.lower().replace("x", "").strip()
            try:
                val = float(cleaned)
            except ValueError:
                val = 2.0
        else:
            try:
                val = float(raw_scale)
            except (TypeError, ValueError):
                val = 2.0

        if val <= 1.6:
            return 1.5
        if val <= 2.5:
            return 2.0
        return 4.0

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(params)
        model_name = str(clean.get("upscaler_model") or "4x-UltraSharp.safetensors")
        if model_name not in UPSCALER_CATALOG:
            model_name = "4x-UltraSharp.safetensors"
        clean["upscaler_model"] = model_name
        clean["upscale_factor"] = self.parse_scale_factor(clean.get("upscale_factor", 2.0))
        return clean

    @classmethod
    def append_inline_nodes(
        cls,
        nodes: Dict[str, Any],
        image_ref: List[Any],
        save_node_id: str = "11",
        upscaler_model: str = "4x-UltraSharp.safetensors",
        scale_factor: float = 2.0,
        filename_prefix: str = "Studio_Upscaled",
    ) -> Dict[str, Any]:
        """
        Menyisipkan rantai node Tiled Upscaler setelah `VAEDecode`:
        `UpscaleModelLoader` -> `ImageUpscaleWithModel` (512x512 tiled) ->
        `ImageScaleBy` (`upscale_method="lanczos"`, `scale_by = target_scale / 4.0`) -> `SaveImage`.
        """
        target_scale = cls.parse_scale_factor(scale_factor)
        scale_by_ratio = round(target_scale / 4.0, 4)

        nodes["90"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": upscaler_model,
            },
        }
        nodes["91"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["90", 0],
                "image": image_ref,
            },
        }
        nodes["92"] = {
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["91", 0],
                "upscale_method": "lanczos",
                "scale_by": scale_by_ratio,
            },
        }
        nodes[save_node_id] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["92", 0],
            },
        }
        return nodes

    def build_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Menghasilkan workflow ComfyUI untuk Standalone Tiled Upscaler."""
        clean = self.sanitize_params(params)
        input_image = clean.get("input_image")
        if not input_image:
            raise ValueError("Pilih atau unggah gambar yang ingin di-upscale terlebih dahulu.")

        staged_name = self.stage_input_image(input_image, prefix="upscale_src")
        nodes: Dict[str, Any] = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": staged_name,
                },
            }
        }
        self.append_inline_nodes(
            nodes=nodes,
            image_ref=["1", 0],
            save_node_id="11",
            upscaler_model=clean["upscaler_model"],
            scale_factor=clean["upscale_factor"],
            filename_prefix="Standalone_Upscale",
        )
        return nodes
