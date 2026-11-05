# -*- coding: utf-8 -*-
"""
4. Krea 2 Turbo 12B GGUF Engine (`engines/krea2_engine.py`)
Spesifikasi Resmi (byteshape/Krea-2-Turbo-GGUF & Comfy-Org/Krea-2):
- Resolusi kelipatan 16. Default: steps=8, cfg=1.0, sampler="euler", scheduler="simple".
- Dokumentasi resmi byteshape/Krea-2-Turbo-GGUF: gunakan CFG=1.0 dan scheduler="simple" pada
  checkpoint Krea-2-Turbo (ConditioningZeroOut pada cabang negatif saat CFG=1.0).
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

KREA2_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square HD (1024 × 1024) — Default Krea 2": (1024, 1024),
    "1:1 Square Fast (768 × 768)": (768, 768),
    "3:4 Editorial Portrait (896 × 1152)": (896, 1152),
    "4:3 Editorial Landscape (1152 × 896)": (1152, 896),
    "2:3 Fashion Studio (832 × 1216)": (832, 1216),
    "3:2 Cinema Wide (1216 × 832)": (1216, 832),
}

KREA2_DEFAULT_NEGATIVE = "worst quality, low quality, blurry, watermark, deformed, plastic skin"


class Krea2Engine(BaseEngine):
    """Workflow Engine untuk Krea 2 Turbo 12B (GGUF Q4_K_M + Qwen3-VL 4B FP8)."""

    engine_id = "krea2"
    display_name = str(ENGINE_CATALOG["krea2"]["display_name"])
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
            "shift": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "width": 1024,
            "height": 1024,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": KREA2_DEFAULT_NEGATIVE,
            "aspect_ratios": KREA2_ASPECT_RATIOS,
            "default_aspect": "1:1 Square HD (1024 × 1024) — Default Krea 2",
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
                    "steps": 12,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 16,
                    "cfg": 1.5,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": True,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "Ultra-detailed analog 35mm film portrait of an astronaut sitting in a retro 1970s diner "
                "at dawn, golden sunlight streaming through blinds, Kodak Portra 400 grain, natural skin texture"
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 1024)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 1024)), mult=16)
        clean["steps"] = max(1, min(100, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(10.0, float(clean.get("cfg", 1.0))))
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
                    "type": "krea2",
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
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": float(strength),
                    "model": model_ref,
                },
            }
            model_ref = [node_id, 0]

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

        sampler_model_ref = model_ref
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
            filename_prefix="Krea2_Studio",
            save_node_id="11",
        )
