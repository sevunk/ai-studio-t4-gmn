# -*- coding: utf-8 -*-
"""
5. Chroma1-HD 8.9B De-Distilled FLUX Engine (`engines/chroma_engine.py`)
Spesifikasi Resmi (lodestones/Chroma1-HD & silveroxides/Chroma1-HD-GGUF):
- Resolusi kelipatan 16. Default: 1024x1024 (atau 1152x1152), steps=26, cfg=3.8,
  sampler="euler", scheduler="beta", flow_shift=1.0, min_padding=1, beta_alpha=0.45, beta_beta=0.45.
- Nodes:
  - UnetLoaderGGUF (`Chroma1-HD-Q5_K_M.gguf`)
  - CLIPLoader (`t5xxl_fp8_e4m3fn_scaled.safetensors`, `type="chroma"`, `device="default"` —
    JANGAN gunakan CLIPLoaderGGUF untuk file `.safetensors`!)
  - VAELoader (`ae.safetensors`)
  - rantai LoraLoader (3 slot)
  - T5TokenizerOptions (`min_padding=1, min_length=0`)
  - ModelSamplingAuraFlow (`shift=1.0`)
  - CLIPTextEncode (pos & neg)
  - CFGGuider (`cfg`)
  - KSamplerSelect (`sampler_name`)
  - RandomNoise (`noise_seed`)
  - EmptySD3LatentImage
  - BetaSamplingScheduler (`steps, alpha, beta`)
  - SamplerCustomAdvanced
  - VAEDecode
  - SaveImage / Inline Upscaler.
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

CHROMA_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square Standard (1024 × 1024) — Cepat di T4": (1024, 1024),
    "1:1 Square HD Resmi (1152 × 1152) — Default Lodestones": (1152, 1152),
    "3:2 Landscape Photo (1216 × 832)": (1216, 832),
    "2:3 Portrait Studio (832 × 1216)": (832, 1216),
    "16:9 Widescreen Cinema (1344 × 768)": (1344, 768),
    "9:16 Vertical Story (768 × 1344)": (768, 1344),
}

CHROMA_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, out of focus, watermark, text, signature, "
    "bad anatomy, deformed hands, oversaturated, plastic skin, jpeg artifacts"
)


class ChromaEngine(BaseEngine):
    """Workflow Engine untuk Chroma1-HD 8.9B (GGUF Q5_K_M + T5-XXL FP8 + Real CFG + BetaScheduler)."""

    engine_id = "chroma"
    display_name = str(ENGINE_CATALOG["chroma"]["display_name"])
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
            "default_preset": "🎯 High Quality",
            "steps": 26,
            "cfg": 3.8,
            "shift": 1.0,
            "flow_shift": 1.0,
            "min_padding": 1,
            "beta_alpha": 0.45,
            "beta_beta": 0.45,
            "sampler": "euler",
            "scheduler": "beta",
            "width": 1024,
            "height": 1024,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": CHROMA_DEFAULT_NEGATIVE,
            "aspect_ratios": CHROMA_ASPECT_RATIOS,
            "default_aspect": "1:1 Square Standard (1024 × 1024) — Cepat di T4",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 16,
                    "cfg": 3.2,
                    "shift": 1.0,
                    "sampler": "heun",
                    "scheduler": "beta",
                    "use_lazycache": False,
                    "use_slg": False,
                    "beta_alpha": 0.45,
                    "beta_beta": 0.45,
                    "min_padding": 1,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 22,
                    "cfg": 3.5,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "beta",
                    "use_lazycache": False,
                    "use_slg": False,
                    "beta_alpha": 0.45,
                    "beta_beta": 0.45,
                    "min_padding": 1,
                },
                "🎯 High Quality": {
                    "steps": 26,
                    "cfg": 3.8,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "beta",
                    "use_lazycache": False,
                    "use_slg": False,
                    "beta_alpha": 0.45,
                    "beta_beta": 0.45,
                    "min_padding": 1,
                },
            },
            "placeholder": (
                "This is a close-up photograph from a nature documentary capturing the right side of the face "
                "of a tiger, intricate fur detail, natural lighting, sharp focus, 3D text saying 'CHROMA1-HD'"
            ),
        }

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 1024)), mult=16)
        clean["height"] = self.snap_multiple(int(clean.get("height", 1024)), mult=16)
        clean["steps"] = max(8, min(60, int(clean.get("steps", 26))))
        clean["cfg"] = max(1.0, min(8.0, float(clean.get("cfg", 3.8))))
        flow_shift = float(clean.get("flow_shift", clean.get("shift", 1.0)))
        clean["flow_shift"] = max(0.2, min(5.0, flow_shift))
        clean["shift"] = clean["flow_shift"]
        clean["min_padding"] = max(0, min(16, int(clean.get("min_padding", 1))))
        clean["beta_alpha"] = max(0.05, min(1.0, float(clean.get("beta_alpha", 0.45))))
        clean["beta_beta"] = max(0.05, min(1.0, float(clean.get("beta_beta", 0.45))))
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

        is_gguf_unet = unet_name.lower().endswith(".gguf")
        unet_node = (
            {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}}
            if is_gguf_unet
            else {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
        )

        # JANGAN gunakan CLIPLoaderGGUF untuk file .safetensors!
        if clip_name.lower().endswith(".gguf"):
            clip_node = {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "chroma",
                },
            }
        else:
            clip_node = {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "chroma",
                    "device": "default",
                },
            }

        nodes: Dict[str, Any] = {
            "710": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_name,
                },
            },
            "731": unet_node,
            "733": clip_node,
        }

        model_ref: List[Any] = ["731", 0]
        clip_ref: List[Any] = ["733", 0]

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

        nodes["741"] = {
            "class_type": "T5TokenizerOptions",
            "inputs": {
                "clip": clip_ref,
                "min_padding": int(p["min_padding"]),
                "min_length": 0,
            },
        }
        nodes["701"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": model_ref,
                "shift": float(p["flow_shift"]),
            },
        }
        nodes["748"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["741", 0],
                "text": prompt,
            },
        }
        nodes["749"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["741", 0],
                "text": negative,
            },
        }
        nodes["694"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["701", 0],
                "positive": ["748", 0],
                "negative": ["749", 0],
                "cfg": float(p["cfg"]),
            },
        }
        nodes["700"] = {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": str(p["sampler"]),
            },
        }
        nodes["718"] = {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(p["seed"]),
            },
        }
        nodes["737"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": int(p["width"]),
                "height": int(p["height"]),
                "batch_size": 1,
            },
        }
        nodes["751"] = {
            "class_type": "BetaSamplingScheduler",
            "inputs": {
                "model": ["701", 0],
                "steps": int(p["steps"]),
                "alpha": float(p["beta_alpha"]),
                "beta": float(p["beta_beta"]),
            },
        }
        nodes["747"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["718", 0],
                "guider": ["694", 0],
                "sampler": ["700", 0],
                "sigmas": ["751", 0],
                "latent_image": ["737", 0],
            },
        }
        nodes["298"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["747", 0],
                "vae": ["710", 0],
            },
        }
        return self.attach_final_save_or_upscale(
            nodes=nodes,
            decoded_image_ref=["298", 0],
            params=p,
            filename_prefix="Chroma1_HD_Studio",
            save_node_id="740",
        )
