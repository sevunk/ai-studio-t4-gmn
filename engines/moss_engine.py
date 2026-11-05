# -*- coding: utf-8 -*-
"""
2. OpenMOSS MOSS-Image-SFT 5B Engine (`engines/moss_engine.py`)
Spesifikasi Resmi (OpenMOSS/MOSS-Image & lodestones/MOSS-Image-SFT-8-steps-dmd):
- Resolusi kelipatan 16.
- Default 8-Step Turbo (DMD LoRA): steps=8, cfg=1.0, shift=5.0, sampler="euler", scheduler="sgm_uniform",
  otomatis memasang `moss_sft_8step_dmd.safetensors` (strength 1.0), use_slg=False.
- Default Speed (Balanced + LazyCache): steps=20, cfg=2.5, shift=5.0, sampler="euler", scheduler="sgm_uniform",
  use_slg=True (layer 22), use_lazycache=True.
- Default High Quality (SFT Base): steps=28, cfg=3.0, shift=5.0, sampler="euler", scheduler="sgm_uniform",
  use_slg=True (layer 22), use_lazycache=False.
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

MOSS_DMD_LORA = "moss_sft_8step_dmd.safetensors"

MOSS_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square T4 Recommended (768 × 768)": (768, 768),
    "1:1 Square HD (1024 × 1024)": (1024, 1024),
    "3:4 Portrait (768 × 1024)": (768, 1024),
    "4:3 Landscape (1024 × 768)": (1024, 768),
    "2:3 Poster Portrait (1024 × 1536)": (1024, 1536),
    "3:2 Wide Landscape (1536 × 1024)": (1536, 1024),
}

MOSS_DEFAULT_NEGATIVE = (
    "worst quality, low quality, blurry, distorted, deformed anatomy, watermark, oversaturated, banding"
)


class MossEngine(BaseEngine):
    """Workflow Engine untuk OpenMOSS MOSS-Image-SFT 5B (INT8 ConvRot + Qwen3.5 9B W4A8 + 8-Step DMD LoRA)."""

    engine_id = "moss"
    display_name = str(ENGINE_CATALOG["moss"]["display_name"])
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
            "shift": 5.0,
            "sampler": "euler",
            "scheduler": "sgm_uniform",
            "width": 768,
            "height": 768,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": MOSS_DEFAULT_NEGATIVE,
            "aspect_ratios": MOSS_ASPECT_RATIOS,
            "default_aspect": "1:1 Square T4 Recommended (768 × 768)",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "sampler": "euler",
                    "scheduler": "sgm_uniform",
                    "use_dmd_lora": True,
                    "use_slg": False,
                    "skip_layer": 22,
                    "use_lazycache": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 20,
                    "cfg": 2.5,
                    "shift": 5.0,
                    "sampler": "euler",
                    "scheduler": "sgm_uniform",
                    "use_dmd_lora": False,
                    "use_slg": True,
                    "skip_layer": 22,
                    "use_lazycache": True,
                },
                "🎯 High Quality": {
                    "steps": 28,
                    "cfg": 3.0,
                    "shift": 5.0,
                    "sampler": "euler",
                    "scheduler": "sgm_uniform",
                    "use_dmd_lora": False,
                    "use_slg": True,
                    "skip_layer": 22,
                    "use_lazycache": False,
                },
            },
            "placeholder": (
                "A cinematic editorial photograph of a small independent bookstore on a rainy evening, "
                "warm amber light inside, wet pavement reflecting the shop windows, realistic materials, fine detail."
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 768)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 768)), mult=16)
        clean["steps"] = max(1, min(100, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(10.0, float(clean.get("cfg", 1.0))))
        clean["shift"] = max(1.0, min(10.0, float(clean.get("shift", 5.0))))
        clean["skip_layer"] = max(0, min(41, int(clean.get("skip_layer", 22))))
        clean["seed"] = int(clean.get("seed", 42))

        preset_mode = str(clean.get("preset_mode", "⚡ Extreme Turbo"))
        if "Turbo" in preset_mode and clean["steps"] <= 10 and clean["cfg"] <= 1.2:
            clean["use_dmd_lora"] = bool(clean.get("use_dmd_lora", True))
        else:
            clean["use_dmd_lora"] = False
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
            else {"class_type": "MOSSImageUNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
        )

        nodes: Dict[str, Any] = {
            "1": unet_node,
            "2": {
                "class_type": "MOSSImageCLIPLoader",
                "inputs": {
                    "clip_name": clip_name,
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
                "class_type": "EmptyFlux2LatentImage",
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
        dmd_exists = (self.models_dir / "loras" / MOSS_DMD_LORA).is_file()
        if p.get("use_dmd_lora", True) and (dmd_exists or p.get("assume_turbo_lora", False)):
            lora_chain.append((MOSS_DMD_LORA, 1.0))

        custom_loras = self.filter_custom_loras(
            p.get("lora_slots", []),
            exclude_names=[MOSS_DMD_LORA],
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
        nodes["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": clip_ref,
            },
        }
        nodes["7"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "shift": float(p["shift"]),
                "model": model_ref,
            },
        }

        single_layers_val = (
            str(int(p["skip_layer"]))
            if (p.get("use_slg", False) and float(p["cfg"]) > 1.0)
            else ""
        )
        nodes["8"] = {
            "class_type": "SkipLayerGuidanceDiTSimple",
            "inputs": {
                "double_layers": "",
                "single_layers": single_layers_val,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "model": ["7", 0],
            },
        }

        sampler_model_ref: List[Any] = ["8", 0]
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
            filename_prefix="MOSS_Studio",
            save_node_id="11",
        )
