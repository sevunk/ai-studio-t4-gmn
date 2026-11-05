# -*- coding: utf-8 -*-
"""
5. Chroma1-HD 8.9B De-Distilled FLUX Engine (`engines/chroma_engine.py`)
Spesifikasi Resmi (lodestones/Chroma1-HD, silveroxides/Chroma1-HD-GGUF, & silveroxides/Chroma-LoRAs):
- Resolusi kelipatan 16.
- Dilengkapi akselerator `Chroma1-Flash_r12-fp32.safetensors` (CFG=1.0 Single-Pass) + `LazyCache`:
  1. `⚡ Extreme Turbo (8-Step / 4-Step LoRA)`:
     - Memasang `Chroma1-Flash_r12-fp32.safetensors` (strength 1.0)
     - `steps=8, cfg=1.0, sampler="euler", scheduler="beta"`
     - Karena `cfg=1.0` menggunakan `ConditioningZeroOut`, GPU hanya mengeksekusi **1 forward pass per step**
       (total 8 evaluasi jaringan, memangkas waktu render di Tesla T4 dari ~320 detik menjadi ~45–52 detik!).
  2. `🚀 Speed (Balanced + LazyCache)`:
     - Memasang `Chroma1-Flash_r12-fp32.safetensors` (strength 1.0) + `LazyCache`
     - `steps=12, cfg=1.0, sampler="euler", scheduler="beta", use_lazycache=True`.
  3. `🎯 High Quality` (De-Distilled Real CFG Murni + LazyCache):
     - Tanpa Flash LoRA (`use_flash_lora=False`), menggunakan Real CFG (`steps=22, cfg=3.5, sampler="euler", scheduler="beta"`)
       dengan `LazyCache` aktif agar menghemat ~25% waktu sampling.
"""

from typing import Any, Dict, List, Tuple

from config.catalog import ENGINE_CATALOG
from engines.base import BaseEngine

CHROMA_FLASH_LORA = "Chroma1-Flash_r12-fp32.safetensors"

CHROMA_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 Square Standard (1024 × 1024) — Cepat di T4": (1024, 1024),
    "3:4 Portrait Fast (768 × 1024) — Sangat Cepat di T4": (768, 1024),
    "4:3 Landscape Fast (1024 × 768)": (1024, 768),
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
    """Workflow Engine untuk Chroma1-HD 8.9B (GGUF Q5_K_M + Chroma1-Flash 8-Step LoRA + Real CFG + LazyCache)."""

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
            "default_preset": "⚡ Extreme Turbo (8-Step / 4-Step LoRA)",
            "steps": 8,
            "cfg": 1.0,
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
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "beta",
                    "use_flash_lora": True,
                    "use_lazycache": False,
                    "use_slg": False,
                    "beta_alpha": 0.45,
                    "beta_beta": 0.45,
                    "min_padding": 1,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 12,
                    "cfg": 1.0,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "beta",
                    "use_flash_lora": True,
                    "use_lazycache": True,
                    "use_slg": False,
                    "beta_alpha": 0.45,
                    "beta_beta": 0.45,
                    "min_padding": 1,
                },
                "🎯 High Quality": {
                    "steps": 22,
                    "cfg": 3.5,
                    "shift": 1.0,
                    "sampler": "euler",
                    "scheduler": "beta",
                    "use_flash_lora": False,
                    "use_lazycache": True,
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
        clean["steps"] = max(4, min(60, int(clean.get("steps", 8))))
        clean["cfg"] = max(1.0, min(8.0, float(clean.get("cfg", 1.0))))
        flow_shift = float(clean.get("flow_shift", clean.get("shift", 1.0)))
        clean["flow_shift"] = max(0.2, min(5.0, flow_shift))
        clean["shift"] = clean["flow_shift"]
        clean["min_padding"] = max(0, min(16, int(clean.get("min_padding", 1))))
        clean["beta_alpha"] = max(0.05, min(1.0, float(clean.get("beta_alpha", 0.45))))
        clean["beta_beta"] = max(0.05, min(1.0, float(clean.get("beta_beta", 0.45))))
        clean["seed"] = int(clean.get("seed", 42))

        preset_mode = str(clean.get("preset_mode", "⚡ Extreme Turbo"))
        if "High Quality" in preset_mode or clean["cfg"] > 1.5:
            clean["use_flash_lora"] = False
        else:
            clean["use_flash_lora"] = bool(clean.get("use_flash_lora", True))
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

        # Pasang Chroma1-Flash r12 Turbo LoRA (UNet-only) saat mode Turbo/Speed (CFG <= 1.5)
        flash_exists = (self.models_dir / "loras" / CHROMA_FLASH_LORA).is_file()
        if p.get("use_flash_lora", True) and (flash_exists or p.get("assume_turbo_lora", False)):
            nodes["19"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_ref,
                    "lora_name": CHROMA_FLASH_LORA,
                    "strength_model": 1.0,
                },
            }
            model_ref = ["19", 0]

        custom_loras = self.filter_custom_loras(
            p.get("lora_slots", []),
            exclude_names=[CHROMA_FLASH_LORA],
        )
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

        sampler_model_ref: List[Any] = ["701", 0]
        if p.get("use_lazycache", False) and int(p["steps"]) >= 8 and p.get("has_lazycache", True):
            nodes["712"] = {
                "class_type": "LazyCache",
                "inputs": {
                    "model": sampler_model_ref,
                    "reuse_threshold": float(p.get("cache_threshold", 0.20)),
                    "start_percent": 0.15,
                    "end_percent": 0.95,
                    "verbose": False,
                },
            }
            sampler_model_ref = ["712", 0]

        nodes["748"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["741", 0],
                "text": prompt,
            },
        }

        # Saat CFG <= 1.0 (Mode Flash Turbo), gunakan ConditioningZeroOut agar tidak melakukan 2x forward pass!
        if float(p["cfg"]) <= 1.0:
            nodes["749"] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["748", 0],
                },
            }
        else:
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
                "model": sampler_model_ref,
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
                "model": sampler_model_ref,
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
