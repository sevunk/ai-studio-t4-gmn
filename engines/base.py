# -*- coding: utf-8 -*-
"""
Abstract BaseEngine (`engines/base.py`)
Mendefinisikan kontrak standar untuk ke-6 Engine Utama + Spandrel Tiled Upscaler.
"""

import shutil
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.catalog import COMFY_DIR, ENGINE_CATALOG, MODELS_DIR


class BaseEngine(ABC):
    """Kelas dasar abstrak untuk seluruh engine workflow ComfyUI di AI Studio T4."""

    engine_id: str = "base"
    display_name: str = "Base Engine"
    is_video: bool = False
    supports_ref_images: bool = False
    supports_negative_prompt: bool = True

    def __init__(self, comfy_dir: Path = COMFY_DIR) -> None:
        self.comfy_dir = comfy_dir
        self.models_dir = comfy_dir / "models"
        self.input_dir = comfy_dir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """
        Mengembalikan dict berisi:
        - `steps`, `cfg`, `shift`, `sampler`, `scheduler`
        - `aspect_ratios`: dict `{label: (width, height)}`
        - `default_aspect`: string key default
        - `presets`: dict `{preset_label: {steps, cfg, ...}}`
        - `placeholder`: contoh prompt yang sesuai dengan karakter engine
        """
        raise NotImplementedError

    @abstractmethod
    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Memvalidasi kelipatan piksel (16 atau 32), jumlah frame ((4*k)+1),
        dan mereset parameter yang tidak kompatibel.
        """
        raise NotImplementedError

    @abstractmethod
    def build_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Menghasilkan payload JSON API ComfyUI yang siap dikirim ke `POST /prompt`."""
        raise NotImplementedError

    def get_catalog_defaults(self) -> Dict[str, Any]:
        """Mengambil nama file default untuk engine ini dari ENGINE_CATALOG."""
        meta = ENGINE_CATALOG.get(self.engine_id, {})
        return dict(meta.get("defaults", {}))  # type: ignore[arg-type]

    def stage_input_image(self, image_path_or_pil: Any, prefix: str = "input") -> str:
        """
        Menyalin file gambar lokal atau objek PIL.Image ke dalam `ComfyUI/input/`
        dan mengembalikan nama file relatif untuk node `LoadImage`.
        """
        self.input_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        uid = uuid.uuid4().hex[:6]

        if isinstance(image_path_or_pil, (str, Path)):
            src = Path(image_path_or_pil)
            if not src.is_file():
                raise FileNotFoundError(f"File gambar referensi tidak ditemukan: {src}")
            ext = src.suffix.lower() if src.suffix else ".png"
            dest_name = f"{prefix}_{ts}_{uid}{ext}"
            shutil.copy2(src, self.input_dir / dest_name)
            return dest_name

        # Jika objek PIL.Image
        from PIL import Image

        if isinstance(image_path_or_pil, Image.Image):
            dest_name = f"{prefix}_{ts}_{uid}.png"
            image_path_or_pil.convert("RGB").save(self.input_dir / dest_name)
            return dest_name

        raise TypeError(f"Format input gambar tidak didukung: {type(image_path_or_pil)}")

    @staticmethod
    def snap_multiple(value: int, mult: int = 16, min_val: int = 256, max_val: int = 4096) -> int:
        """Membulatkan ukuran piksel ke kelipatan `mult` terdekat dalam batas aman."""
        clamped = max(min_val, min(max_val, int(value)))
        return max(min_val, (clamped // mult) * mult)

    @staticmethod
    def filter_custom_loras(
        lora_slots: List[Tuple[str, float]],
        exclude_names: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Memfilter slot Custom LoRA yang aktif dan valid."""
        excluded = set(exclude_names or []) | {"", "(None)", "None"}
        valid: List[Tuple[str, float]] = []
        for name, strength in lora_slots or []:
            if name and str(name) not in excluded and abs(float(strength)) > 1e-4:
                valid.append((str(name), float(strength)))
        return valid

    @staticmethod
    def attach_final_save_or_upscale(
        nodes: Dict[str, Any],
        decoded_image_ref: List[Any],
        params: Dict[str, Any],
        filename_prefix: str,
        save_node_id: str = "11",
    ) -> Dict[str, Any]:
        """
        Menghubungkan output `VAEDecode` ke `SaveImage` secara langsung, ATAU
        menyisipkan node Spandrel Tiled Upscaler (`UpscaleModelLoader` ->
        `ImageUpscaleWithModel` -> `ImageScaleBy` Lanczos) jika `auto_upscale=True`.
        """
        if params.get("auto_upscale", False):
            from engines.upscaler_engine import UpscalerEngine

            upscaler_model = str(params.get("upscaler_model") or "4x-UltraSharp.safetensors")
            upscale_factor = float(params.get("upscale_factor") or 2.0)
            UpscalerEngine.append_inline_nodes(
                nodes=nodes,
                image_ref=decoded_image_ref,
                save_node_id=save_node_id,
                upscaler_model=upscaler_model,
                scale_factor=upscale_factor,
                filename_prefix=f"{filename_prefix}_Upscaled",
            )
        else:
            nodes[save_node_id] = {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": filename_prefix,
                    "images": decoded_image_ref,
                },
            }
        return nodes
