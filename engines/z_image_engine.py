# -*- coding: utf-8 -*-
"""
6. Tongyi-MAI Z-Image-Turbo 6B Engine (`engines/z_image_engine.py`)
Spesifikasi Resmi (Tongyi-MAI/Z-Image-Turbo & Comfy-Org/Z-Image-Turbo):
- Single-Stream Diffusion Transformer (S3-DiT) 6B yang dioptimalkan untuk 8 NFEs:
  - Keunggulan: Fotorealisme tajam, perenderan teks dwibahasa (Inggris & Mandarin) akurat, dan kepatuhan instruksi tinggi.
- File Resmi Comfy-Org untuk Tesla T4 (15 GiB VRAM):
  - Diffusion: `z_image_turbo_int8_convrot.safetensors` (`6.20 GB`, dimuat via `UNETLoader`)
  - Text Encoder: `qwen_3_4b_fp8_mixed.safetensors` (`5.63 GB`, dimuat via `CLIPLoader` dengan `type="lumina2"`)
  - VAE: `ae.safetensors` (`335 MB` FLUX VAE — dibagi bersama Chroma1-HD)
- Nodes Workflow Resmi Comfy-Org:
  - `UNETLoader` -> rantai `LoraLoader` (3 slot) -> `ModelSamplingAuraFlow(shift=3.0)` -> `LazyCache` (opsional)
  - `CLIPLoader(type="lumina2")` -> `CLIPTextEncode` (positive) & `ConditioningZeroOut` (jika `cfg <= 1.0`)
  - `EmptySD3LatentImage` (kelipatan 16, default `1024x1024`)
  - `KSampler(steps=8, cfg=1.0, sampler_name="res_multistep", scheduler="simple")`
  - `VAEDecode` -> `SaveImage` / Inline Spandrel Tiled Upscaler.
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

Z_IMAGE_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square HD (1024 × 1024) — Default Z-Image-Turbo": (1024, 1024),
    "16:9 Widescreen Official (1280 × 720)": (1280, 720),
    "9:16 Vertical Official (720 × 1280)": (720, 1280),
    "3:4 Editorial Portrait (896 × 1152)": (896, 1152),
    "4:3 Editorial Landscape (1152 × 896)": (1152, 896),
    "2:3 Poster Portrait (832 × 1216)": (832, 1216),
    "3:2 Cinema Wide (1216 × 832)": (1216, 832),
}

Z_IMAGE_DEFAULT_NEGATIVE = (
    "worst quality, low quality, blurry, out of focus, distorted text, deformed hands, "
    "bad anatomy, watermark, oversaturated, jpeg artifacts"
)


class ZImageEngine(BaseEngine):
    """Workflow Engine untuk Tongyi-MAI Z-Image-Turbo 6B (INT8 ConvRot + Qwen3 4B FP8 Lumina2 + AuraFlow Shift 3.0)."""

    engine_id = "z_image"
    display_name = str(ENGINE_CATALOG["z_image"]["display_name"])
    is_video = False
    supports_ref_images = False
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
            "shift": 3.0,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "width": 1024,
            "height": 1024,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": Z_IMAGE_DEFAULT_NEGATIVE,
            "aspect_ratios": Z_IMAGE_ASPECT_RATIOS,
            "default_aspect": "1:1 Square HD (1024 × 1024) — Default Z-Image-Turbo",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 3.0,
                    "sampler": "res_multistep",
                    "scheduler": "simple",
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 12,
                    "cfg": 1.0,
                    "shift": 3.0,
                    "sampler": "res_multistep",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 16,
                    "cfg": 1.5,
                    "shift": 3.0,
                    "sampler": "res_multistep",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "A wide-shot, eye-level photograph captures a lone Apollo astronaut standing beside a Lunar Roving "
                "Vehicle on the desolate moonscape, gold-tinted reflective visor, sharp Hasselblad 70mm clarity, "
                "clear stencil sign reading 'Z-IMAGE TURBO'"
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 1024)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 1024)), mult=16)
        clean["steps"] = max(1, min(60, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(8.0, float(clean.get("cfg", 1.0))))
        clean["shift"] = max(0.5, min(10.0, float(clean.get("shift", 3.0))))
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

        if clip_name.lower().endswith(".gguf"):
            clip_node = {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "lumina2",
                },
            }
        else:
            clip_node = {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "lumina2",
                    "device": "default",
                },
            }

        nodes: Dict[str, Any] = {
            "1": unet_node,
            "2": clip_node,
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_name,
                },
            },
            "6": {
                "class_type": "EmptySD3LatentImage",
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

        # Official Z-Image-Turbo ModelSamplingAuraFlow (shift=3.0)
        nodes["7"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": model_ref,
                "shift": float(p["shift"]),
            },
        }

        sampler_model_ref: List[Any] = ["7", 0]
        if p.get("use_lazycache", False) and int(p["steps"]) >= 8 and p.get("has_lazycache", True):
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

        nodes["4"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": clip_ref,
            },
        }

        if float(p["cfg"]) <= 1.0:
            nodes["5"] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["4", 0],
                },
            }
        else:
            nodes["5"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative,
                    "clip": clip_ref,
                },
            }

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
                "negative": ["5", 0],
                "latent_image": ["6", 0],
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
            filename_prefix="ZImage_Turbo_Studio",
            save_node_id="11",
        )
