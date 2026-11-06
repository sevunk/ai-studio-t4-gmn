# -*- coding: utf-8 -*-
"""
7. Flux-Fusion V2 12B 4-Step Merge GGUF Engine (`engines/flux_fusion_engine.py`)
Spesifikasi Resmi (Anibaaal/Flux-Fusion-V2-4step-merge-gguf-nf4 & comfyanonymous/flux_text_encoders):
- Arsitektur: FLUX.1 12B Merge dari `FLUX.1-schnell`, finetuned `FLUX.1-dev`, dan `Hyper-FLUX`.
- Direkomendasikan 4–8 steps ("Recommended 4-8 steps. Greatly improved quality at 4 steps compared to V1").
- File Resmi untuk Tesla T4 (15 GiB VRAM):
  - Diffusion: `FluxFusionV2-Q4_K_M.gguf` (`6.92 GB`, dimuat via `UnetLoaderGGUF`)
  - Dual Text Encoder (`DualCLIPLoader` dengan `type="flux"`):
    1. `t5xxl_fp8_e4m3fn_scaled.safetensors` (`5.16 GB` — dibagi otomatis dengan `Chroma1-HD`)
    2. `clip_l.safetensors` (`246 MB`)
  - VAE: `ae.safetensors` (`335 MB` — dibagi otomatis dengan `Chroma1-HD` & `Z-Image-Turbo`)
- Nodes Workflow Resmi ComfyUI FLUX GGUF:
  - `UnetLoaderGGUF` -> rantai `LoraLoader` (3 slot) -> `LazyCache` (opsional jika `steps >= 8`)
  - `DualCLIPLoader(clip_name1=t5xxl, clip_name2=clip_l, type="flux")` -> `CLIPTextEncode` -> `FluxGuidance(guidance=shift)`
  - `ConditioningZeroOut` pada cabang negatif saat `cfg <= 1.0` (1 forward pass per step = 4 NFE total di mode 4-Step!)
  - `EmptySD3LatentImage` (kelipatan 16, default `1024x1024`)
  - `KSampler(steps=4, cfg=1.0, sampler_name="euler", scheduler="simple")`
  - `VAEDecode` -> `SaveImage` / Inline Spandrel Tiled Upscaler.
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

FLUX_FUSION_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square HD (1024 × 1024) — Default Flux-Fusion V2": (1024, 1024),
    "3:4 Portrait Editorial (896 × 1152)": (896, 1152),
    "4:3 Landscape Editorial (1152 × 896)": (1152, 896),
    "2:3 Cinema Portrait (832 × 1216)": (832, 1216),
    "3:2 Cinema Wide (1216 × 832)": (1216, 832),
    "16:9 Widescreen 720p+ (1344 × 768)": (1344, 768),
    "9:16 Vertical Story (768 × 1344)": (768, 1344),
}

FLUX_FUSION_DEFAULT_NEGATIVE = (
    "worst quality, low quality, blurry, out of focus, watermark, deformed hands, "
    "bad anatomy, oversaturated, plastic skin, jpeg artifacts"
)


class FluxFusionEngine(BaseEngine):
    """Workflow Engine untuk Anibaaal Flux-Fusion V2 12B (4-Step Merge GGUF Q4_K_M + DualCLIP T5-XXL/CLIP-L)."""

    engine_id = "flux_fusion"
    display_name = str(ENGINE_CATALOG["flux_fusion"]["display_name"])
    is_video = False
    supports_ref_images = False
    supports_negative_prompt = True

    @property
    def default_params(self) -> Dict[str, Any]:
        defaults = self.get_catalog_defaults()
        return {
            "unet_name": defaults["unet"],
            "clip_name": defaults["text_encoder"],
            "clip_l_name": defaults.get("clip_l", "clip_l.safetensors"),
            "vae_name": defaults["vae"],
            "default_preset": "⚡ Extreme Turbo (8-Step / 4-Step LoRA)",
            "steps": 4,
            "cfg": 1.0,
            "shift": 3.5,  # Digunakan sebagai FluxGuidance (default 3.5 untuk FLUX.1)
            "sampler": "euler",
            "scheduler": "simple",
            "width": 1024,
            "height": 1024,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": FLUX_FUSION_DEFAULT_NEGATIVE,
            "aspect_ratios": FLUX_FUSION_ASPECT_RATIOS,
            "default_aspect": "1:1 Square HD (1024 × 1024) — Default Flux-Fusion V2",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 4,
                    "cfg": 1.0,
                    "shift": 3.5,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 6,
                    "cfg": 1.0,
                    "shift": 3.5,
                    "sampler": "euler",
                    "scheduler": "sgm_uniform",
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 3.5,
                    "sampler": "dpmpp_2m",
                    "scheduler": "sgm_uniform",
                    "use_lazycache": True,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "Astronaut in a bioluminescent jungle at night, cold color palette, muted cinematic colors, "
                "intricate suit reflections, volumetric fog, shot on ARRI Alexa 65, 8k photorealistic"
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 1024)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 1024)), mult=16)
        clean["steps"] = max(1, min(50, int(clean.get("steps", 4))))
        clean["cfg"] = max(1.0, min(8.0, float(clean.get("cfg", 1.0))))
        clean["shift"] = max(0.5, min(10.0, float(clean.get("shift", 3.5))))
        clean["seed"] = int(clean.get("seed", 42))
        return clean

    def build_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = self.sanitize_params(params)
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt tidak boleh kosong.")
        negative = str(p.get("negative_prompt", "")).strip()

        unet_name = str(p["unet_name"])
        t5_name = str(p["clip_name"])
        clip_l_name = str(p.get("clip_l_name", "clip_l.safetensors"))
        if t5_name == "clip_l.safetensors":
            t5_name = "t5xxl_fp8_e4m3fn_scaled.safetensors"
        vae_name = str(p["vae_name"])

        is_gguf = unet_name.lower().endswith(".gguf")
        unet_node = (
            {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}}
            if is_gguf
            else {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
        )

        if t5_name.lower().endswith(".gguf") or clip_l_name.lower().endswith(".gguf"):
            clip_node = {
                "class_type": "DualCLIPLoaderGGUF",
                "inputs": {
                    "clip_name1": t5_name,
                    "clip_name2": clip_l_name,
                    "type": "flux",
                },
            }
        else:
            clip_node = {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": t5_name,
                    "clip_name2": clip_l_name,
                    "type": "flux",
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

        sampler_model_ref: List[Any] = model_ref
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
        nodes["7"] = {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["4", 0],
                "guidance": float(p["shift"]),
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
                "positive": ["7", 0],
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
            filename_prefix="FluxFusionV2_Studio",
            save_node_id="11",
        )
