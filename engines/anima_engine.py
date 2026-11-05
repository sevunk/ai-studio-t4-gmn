# -*- coding: utf-8 -*-
"""
1. CircleStone Labs Anima 2B Engine (`engines/anima_engine.py`)
Spesifikasi Resmi (circlestone-labs/Anima README.md):
- Resolusi wajib kelipatan 16 (512^2 s/d 1536^2, Default: 1024x1024)
- Preset Extreme Turbo (Anima-Turbo LoRA v0.2): steps=8, cfg=1.0, sampler="euler", scheduler="simple"
- Preset Speed (Anima-Turbo LoRA v0.2 + 12 steps): steps=12, cfg=1.0, sampler="euler", scheduler="simple"
- Preset High Quality (Anima-Base tanpa Turbo LoRA): steps=35 (rentang resmi 30-50), cfg=4.5 (rentang resmi 4-5),
  sampler="er_sde" (rekomendasi utama pembuat Anima untuk garis tajam & warna akurat), scheduler="simple".
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

ANIMA_TURBO_LORA = "anima-turbo-lora-v0.2.safetensors"

ANIMA_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square HD (1024 × 1024) — Default Anima": (1024, 1024),
    "3:4 Anime Portrait (896 × 1152)": (896, 1152),
    "4:3 Anime Landscape (1152 × 896)": (1152, 896),
    "2:3 Illustration Vertical (832 × 1216)": (832, 1216),
    "3:2 Illustration Wide (1216 × 832)": (1216, 832),
    "9:16 Wallpaper Vertical (768 × 1344)": (768, 1344),
    "16:9 Cinematic Anime (1344 × 768)": (1344, 768),
}

ANIMA_DEFAULT_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, "
    "blurry, jpeg artifacts, chromatic aberration"
)


class AnimaEngine(BaseEngine):
    """Workflow Engine untuk CircleStone Labs Anima 2B (Base v1.0 + Official Turbo LoRA v0.2)."""

    engine_id = "anima"
    display_name = str(ENGINE_CATALOG["anima"]["display_name"])
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
            "negative_prompt": ANIMA_DEFAULT_NEGATIVE,
            "aspect_ratios": ANIMA_ASPECT_RATIOS,
            "default_aspect": "1:1 Square HD (1024 × 1024) — Default Anima",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_turbo_lora": True,
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 12,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_turbo_lora": True,
                    "use_lazycache": True,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 35,
                    "cfg": 4.5,
                    "shift": 1.0,
                    "sampler": "er_sde",
                    "scheduler": "simple",
                    "use_turbo_lora": False,
                    "use_lazycache": True,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "masterpiece, best quality, score_7, safe, 1girl, silver hair, glowing blue eyes, "
                "standing on a floating crystal balcony above a night city, dramatic lighting, vibrant colors"
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 1024)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 1024)), mult=16)
        clean["steps"] = max(1, min(100, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(12.0, float(clean.get("cfg", 1.0))))
        clean["seed"] = int(clean.get("seed", 42))

        preset_mode = str(clean.get("preset_mode", "⚡ Extreme Turbo"))
        if "High Quality" in preset_mode or clean["steps"] >= 20 or clean["cfg"] >= 2.5:
            clean["use_turbo_lora"] = False
        else:
            clean["use_turbo_lora"] = bool(clean.get("use_turbo_lora", True))
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
                    "type": "stable_diffusion",
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

        lora_chain: List[Tuple[str, float]] = []
        turbo_file_exists = (self.models_dir / "loras" / ANIMA_TURBO_LORA).is_file()
        if p.get("use_turbo_lora", True) and (turbo_file_exists or p.get("assume_turbo_lora", False)):
            lora_chain.append((ANIMA_TURBO_LORA, 1.0))

        custom_loras = self.filter_custom_loras(
            p.get("lora_slots", []),
            exclude_names=[ANIMA_TURBO_LORA],
        )
        lora_chain.extend(custom_loras)

        for idx, (lora_name, strength) in enumerate(lora_chain):
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
            filename_prefix="Anima_Studio",
            save_node_id="11",
        )
