# -*- coding: utf-8 -*-
"""
3. Qwen-Image 2.1 Turbo 7B Engine (`engines/qwen21_engine.py`)
Spesifikasi Resmi (Comfy-Org/Qwen-Image-2.1):
- Resolusi WAJIB kelipatan 32. Default: steps=8, cfg=1.0, sampler="euler", scheduler="simple".
- Mendukung 1–3 gambar referensi (`ref_img1`, `ref_img2`, `ref_img3`).
- Jika ada gambar referensi, salin ke `ComfyUI/input/`, buat node `LoadImage` untuk masing-masing,
  dan hubungkan ke `TextEncodeQwenImage21` (`images.image_1`, dst., dengan `resolution=992` agar
  bebas bug noise #16435; jika tanpa gambar referensi set `resolution=0`).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

QWEN21_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square T4 Recommended (768 × 768)": (768, 768),
    "1:1 Square HD (1024 × 1024)": (1024, 1024),
    "3:4 Portrait (768 × 1024)": (768, 1024),
    "4:3 Landscape (1024 × 768)": (1024, 768),
    "9:16 Vertical (768 × 1344)": (768, 1344),
    "16:9 Widescreen (1344 × 768)": (1344, 768),
}

QWEN21_DEFAULT_NEGATIVE = "worst quality, low quality, blurry, deformed, watermark, text artifacts"


class Qwen21Engine(BaseEngine):
    """Workflow Engine untuk Qwen-Image 2.1 Turbo 7B (INT8 ConvRot + W4A8 + 3-Ref Multi-Image Edit)."""

    engine_id = "qwen21"
    display_name = str(ENGINE_CATALOG["qwen21"]["display_name"])
    is_video = False
    supports_ref_images = True
    supports_negative_prompt = True

    @property
    def default_params(self) -> Dict[str, Any]:
        defaults = self.get_catalog_defaults()
        return {
            "unet_name": defaults["unet"],
            "clip_name": defaults["text_encoder"],
            "vae_name": defaults["vae"],
            "default_preset": "⚡ Extreme Turbo (8-Step / 4-Step LoRA)",
            "steps": 8,
            "cfg": 1.0,
            "shift": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "width": 768,
            "height": 768,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "lock_ref_size": True,
            "ref_resolution": 992,
            "negative_prompt": QWEN21_DEFAULT_NEGATIVE,
            "aspect_ratios": QWEN21_ASPECT_RATIOS,
            "default_aspect": "1:1 Square T4 Recommended (768 × 768)",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 14,
                    "cfg": 1.5,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 24,
                    "cfg": 2.5,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "Untuk T2I: deskripsikan adegan detail. Untuk Multi-Image Edit: unggah gambar di tab "
                "Input Image lalu gunakan tag <image1>, <image2>, atau <image3> di dalam kalimat prompt."
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        # Qwen-Image 2.1 WAJIB kelipatan 32 piksel
        clean["width"] = self.snap_multiple(int(clean.get("width", 768)), mult=32)
        clean["height"] = self.snap_multiple(int(clean.get("height", 768)), mult=32)
        clean["steps"] = max(1, min(100, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(10.0, float(clean.get("cfg", 1.0))))
        clean["ref_resolution"] = self.snap_multiple(
            int(clean.get("ref_resolution", 992)), mult=32, min_val=0, max_val=1536
        )
        clean["seed"] = int(clean.get("seed", 42))
        return clean

    def build_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = self.sanitize_params(params)
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt tidak boleh kosong.")
        negative = str(p.get("negative_prompt", "")).strip()

        unet_name = str(p["unet_name"])
        clip_name = str(p["clip_name"])
        vae_name = str(p["vae_name"])

        is_gguf = unet_name.lower().endswith(".gguf")
        unet_node = (
            {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}}
            if is_gguf
            else {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
        )

        nodes: Dict[str, Any] = {
            "1": unet_node,
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "qwen_image",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_name,
                },
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": p["width"],
                    "height": p["height"],
                    "batch_size": 1,
                },
            },
        }

        model_ref: List[Any] = ["1", 0]
        clip_ref: List[Any] = ["2", 0]

        custom_loras = self.filter_custom_loras(p.get("lora_slots", []))
        for idx, (lora_name, strength) in enumerate(custom_loras):
            node_id = str(20 + idx)
            nodes[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": float(strength),
                    "strength_clip": float(strength),
                    "model": model_ref,
                    "clip": clip_ref,
                },
            }
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]

        raw_refs = p.get("ref_images") or [
            p.get("ref_img1"),
            p.get("ref_img2"),
            p.get("ref_img3"),
        ]
        valid_refs = [
            r
            for r in raw_refs
            if r is not None and (not isinstance(r, (str, Path)) or Path(r).is_file())
        ]

        ref_res = int(p.get("ref_resolution", 992)) if valid_refs else 0
        if valid_refs and ref_res <= 0:
            ref_res = 992

        encode_inputs: Dict[str, Any] = {
            "clip": clip_ref,
            "prompt": prompt,
            "negative_prompt": negative,
            "resolution": ref_res,
        }

        if valid_refs:
            encode_inputs["vae"] = ["3", 0]
            for idx, ref_item in enumerate(valid_refs[:3], start=1):
                load_node_id = str(30 + idx)
                staged_name = self.stage_input_image(ref_item, prefix=f"qwen21_ref{idx}")
                nodes[load_node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": staged_name,
                    },
                }
                encode_inputs[f"images.image_{idx}"] = [load_node_id, 0]

        nodes["4"] = {
            "class_type": "TextEncodeQwenImage21",
            "inputs": encode_inputs,
        }

        sampler_model_ref = model_ref
        if p.get("has_qwen_cache", True):
            nodes["15"] = {
                "class_type": "QwenImage21Cache",
                "inputs": {
                    "model": sampler_model_ref,
                    "device": "auto",
                    "dtype": "int8" if valid_refs else "default",
                },
            }
            sampler_model_ref = ["15", 0]

        if p.get("use_lazycache", False) and int(p["steps"]) >= 12 and p.get("has_lazycache", True):
            nodes["12"] = {
                "class_type": "LazyCache",
                "inputs": {
                    "model": sampler_model_ref,
                    "reuse_threshold": float(p.get("cache_threshold", 0.20)),
                    "start_percent": 0.15,
                    "end_percent": 0.95,
                    "verbose": False,
                },
            }
            sampler_model_ref = ["12", 0]

        latent_source = ["4", 2] if (valid_refs and p.get("lock_ref_size", True)) else ["6", 0]
        nodes["9"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(p["seed"]),
                "steps": int(p["steps"]),
                "cfg": float(p["cfg"]),
                "sampler_name": str(p["sampler"]),
                "scheduler": str(p["scheduler"]),
                "denoise": 1.0,
                "model": sampler_model_ref,
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": latent_source,
            },
        }
        nodes["10"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0],
            },
        }
        return self.attach_final_save_or_upscale(
            nodes=nodes,
            decoded_image_ref=["10", 0],
            params=p,
            filename_prefix="Qwen21_Studio",
            save_node_id="11",
        )
