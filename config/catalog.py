# -*- coding: utf-8 -*-
"""
Single Source of Truth: Katalog 7 Engine Utama + 3 Model Upscaler + Shared Files + Validasi Ukuran Byte.
Seluruh URL dan `min_bytes` telah diaudit langsung terhadap ukuran file aktual di Hugging Face.
Target Perangkat Keras: Google Colab Tesla T4 GPU (15 GiB VRAM, ~12.7 GiB System RAM, ~70 GiB Disk).
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

BASE_DIR = Path(os.environ.get("STUDIO_BASE_DIR", "/content" if Path("/content").exists() else str(Path.cwd())))
COMFY_DIR = Path(os.environ.get("STUDIO_COMFY_DIR", str(BASE_DIR / "ComfyUI")))
MODELS_DIR = COMFY_DIR / "models"
GALLERY_DIR = Path(os.environ.get("STUDIO_GALLERY_DIR", str(BASE_DIR / "Studio_Gallery")))

MODEL_SUBDIRS: Tuple[str, ...] = (
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "vae",
    "loras",
    "clip_vision",
    "upscale_models",
)


@dataclass(frozen=True)
class ModelFileSpec:
    """Spesifikasi satu file bobot model beserta ukuran minimal validasi anti-korup."""

    role: str  # "diffusion", "text_encoder", "vae", "lora", "clip_vision", "upscaler"
    subdir: str  # Folder di bawah ComfyUI/models/
    filename: str  # Nama file target di disk lokal
    url: str  # Direct resolve URL
    min_bytes: int  # Ukuran byte minimum agar dianggap valid
    shared_with: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def rel_path(self) -> str:
        return f"{self.subdir}/{self.filename}"

    def resolve_local_path(self, models_root: Path = MODELS_DIR) -> Path:
        """Mengembalikan path utama file di dalam models_root."""
        return models_root / self.subdir / self.filename

    def candidate_paths(self, models_root: Path = MODELS_DIR) -> List[Path]:
        """Mendukung pembacaan silang diffusion_models <-> unet dan text_encoders <-> clip."""
        paths = [models_root / self.subdir / self.filename]
        if self.subdir == "diffusion_models":
            paths.append(models_root / "unet" / self.filename)
        elif self.subdir == "unet":
            paths.append(models_root / "diffusion_models" / self.filename)
        elif self.subdir == "text_encoders":
            paths.append(models_root / "clip" / self.filename)
        elif self.subdir == "clip":
            paths.append(models_root / "text_encoders" / self.filename)
        return paths


# ==============================================================================
# A. 7 ENGINE UTAMA (RESMI & TERVALIDASI UKURAN AKTUAL HUGGING FACE)
# ==============================================================================
ENGINE_CATALOG: Dict[str, Dict[str, object]] = {
    "anima": {
        "engine_id": "anima",
        "display_name": "1. Anima 2B (CircleStone Anime/Illustration · Base + Turbo LoRA)",
        "short_name": "Anima 2B",
        "is_video": False,
        "supports_ref_images": False,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 5.77,
        "defaults": {
            "unet": "anima-base-v1.0.safetensors",
            "text_encoder": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
            "turbo_lora": "anima-turbo-lora-v0.2.safetensors",
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="anima-base-v1.0.safetensors",
                url="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
                min_bytes=3_600_000_000,  # Aktual HF: 4.18 GB
                description="Anima 2B Base v1.0 Diffusion (Aktual HF: 4.18 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="qwen_3_06b_base.safetensors",
                url="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
                min_bytes=1_000_000_000,  # Aktual HF: 1.19 GB
                description="Qwen3 0.6B Base Text Encoder (Aktual HF: 1.19 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="qwen_image_vae.safetensors",
                url="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                min_bytes=200_000_000,  # Aktual HF: 254 MB
                shared_with=("anima", "krea2"),
                description="Qwen Image VAE (Aktual HF: 254 MB — Shared dengan Krea 2 Turbo)",
            ),
            ModelFileSpec(
                role="lora",
                subdir="loras",
                filename="anima-turbo-lora-v0.2.safetensors",
                url="https://huggingface.co/circlestone-labs/Anima-Official-LoRAs/resolve/main/anima-turbo-lora-v0.2.safetensors",
                min_bytes=120_000_000,  # Aktual HF: 149 MB
                description="Anima Official Turbo LoRA v0.2 (Aktual HF: 149 MB)",
            ),
        ],
    },
    "moss": {
        "engine_id": "moss",
        "display_name": "2. MOSS-Image-SFT 5B (OpenMOSS INT8 ConvRot + Qwen3.5 9B W4A8)",
        "short_name": "MOSS-Image-SFT 5B",
        "is_video": False,
        "supports_ref_images": False,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 12.16,
        "defaults": {
            "unet": "moss_image_sft_int8_convrot.safetensors",
            "text_encoder": "qwen3.5_9b_moss_image_sft_w4a8_gptq.safetensors",
            "vae": "moss_image_vae_bf16.safetensors",
            "turbo_lora": "moss_sft_8step_dmd.safetensors",
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="moss_image_sft_int8_convrot.safetensors",
                url="https://huggingface.co/nif0/MOSS-Image-SFT-ComfyUI/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors",
                min_bytes=5_000_000_000,  # Aktual HF: 5.41 GB
                description="MOSS-Image-SFT 5B INT8 ConvRot (Aktual HF: 5.41 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="qwen3.5_9b_moss_image_sft_w4a8_gptq.safetensors",
                url="https://huggingface.co/nif0/MOSS-Image-SFT-ComfyUI/resolve/main/text_encoders/qwen3.5_9b_moss_image_sft_w4a8_gptq.safetensors",
                min_bytes=5_000_000_000,  # Aktual HF: 5.43 GB
                description="Qwen3.5 9B MOSS W4A8 GPTQ Text Encoder (Aktual HF: 5.43 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="moss_image_vae_bf16.safetensors",
                url="https://huggingface.co/nif0/MOSS-Image-SFT-ComfyUI/resolve/main/vae/moss_image_vae_bf16.safetensors",
                min_bytes=100_000_000,  # Aktual HF: 125 MB
                description="MOSS Image VAE BF16 (Aktual HF: 125 MB)",
            ),
            ModelFileSpec(
                role="lora",
                subdir="loras",
                filename="moss_sft_8step_dmd.safetensors",
                url="https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora.safetensors",
                min_bytes=50_000_000,  # Aktual HF: 1.19 GB
                description="MOSS 8-Step DMD Turbo LoRA (Aktual HF: 1.19 GB)",
            ),
        ],
    },
    "qwen21": {
        "engine_id": "qwen21",
        "display_name": "3. Qwen-Image 2.1 Turbo 7B (INT8 + W4A8 · T2I & 3-Ref Image Edit)",
        "short_name": "Qwen-Image 2.1 Turbo",
        "is_video": False,
        "supports_ref_images": True,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 14.25,
        "defaults": {
            "unet": "qwen_image_2.1_turbo_int8_convrot.safetensors",
            "text_encoder": "qwen3vl_8b_w4a8.safetensors",
            "vae": "qwen_image_2.1_vae_bf16.safetensors",
            "turbo_lora": None,
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="qwen_image_2.1_turbo_int8_convrot.safetensors",
                url="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                min_bytes=7_000_000_000,  # Aktual HF: 7.26 GB
                description="Qwen-Image 2.1 Turbo 7B INT8 ConvRot (Aktual HF: 7.26 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="qwen3vl_8b_w4a8.safetensors",
                url="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors",
                min_bytes=6_000_000_000,  # Aktual HF: 6.31 GB
                description="Qwen3-VL 8B W4A8 Text & Vision Encoder (Aktual HF: 6.31 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="qwen_image_2.1_vae_bf16.safetensors",
                url="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
                min_bytes=600_000_000,  # Aktual HF: 676 MB
                description="Qwen-Image 2.1 RGBA VAE BF16 (Aktual HF: 676 MB)",
            ),
        ],
    },
    "krea2": {
        "engine_id": "krea2",
        "display_name": "4. Krea 2 Turbo 12B (GGUF Q4_K_M + Qwen3-VL 4B FP8)",
        "short_name": "Krea 2 Turbo 12B",
        "is_video": False,
        "supports_ref_images": False,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 13.39,
        "defaults": {
            "unet": "Krea-2-Turbo-Q4_K_M-4.93bpw.gguf",
            "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
            "vae": "qwen_image_vae.safetensors",
            "turbo_lora": None,
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="Krea-2-Turbo-Q4_K_M-4.93bpw.gguf",
                url="https://huggingface.co/byteshape/Krea-2-Turbo-GGUF/resolve/main/Krea-2-Turbo-Q4_K_M-4.93bpw.gguf",
                min_bytes=7_000_000_000,  # Aktual HF: 7.90 GB
                description="Krea 2 Turbo 12B GGUF Q4_K_M (Aktual HF: 7.90 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="qwen3vl_4b_fp8_scaled.safetensors",
                url="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                min_bytes=4_800_000_000,  # Aktual HF: 5.24 GB
                description="Qwen3-VL 4B FP8 Scaled Text Encoder (Aktual HF: 5.24 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="qwen_image_vae.safetensors",
                url="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
                min_bytes=200_000_000,  # Aktual HF: 254 MB
                shared_with=("anima", "krea2"),
                description="Qwen Image VAE (Aktual HF: 254 MB — Shared dengan Anima 2B)",
            ),
        ],
    },
    "chroma": {
        "engine_id": "chroma",
        "display_name": "5. Chroma1-HD 8.9B (GGUF Q5_K_M + Flash 8-Step LoRA & Real CFG)",
        "short_name": "Chroma1-HD 8.9B",
        "is_video": False,
        "supports_ref_images": False,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 12.32,
        "defaults": {
            "unet": "Chroma1-HD-Q5_K_M.gguf",
            "text_encoder": "t5xxl_fp8_e4m3fn_scaled.safetensors",
            "vae": "ae.safetensors",
            "turbo_lora": "Chroma1-Flash_r12-fp32.safetensors",
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="Chroma1-HD-Q5_K_M.gguf",
                url="https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q5_K_M.gguf",
                min_bytes=6_200_000_000,  # Aktual HF: 6.65 GB
                description="Chroma1-HD 8.9B GGUF Q5_K_M (Aktual HF: 6.65 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="t5xxl_fp8_e4m3fn_scaled.safetensors",
                url="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
                min_bytes=4_500_000_000,  # Aktual HF: 5.16 GB
                description="T5-XXL FP8 Scaled Text Encoder (Aktual HF: 5.16 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="ae.safetensors",
                url="https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
                min_bytes=300_000_000,  # Aktual HF: 335 MB
                shared_with=("chroma", "z_image"),
                description="FLUX AutoEncoder VAE (Aktual HF: 335 MB — Shared dengan Z-Image-Turbo)",
            ),
            ModelFileSpec(
                role="lora",
                subdir="loras",
                filename="Chroma1-Flash_r12-fp32.safetensors",
                url="https://huggingface.co/silveroxides/Chroma-LoRAs/resolve/main/Chroma1-Flash/Chroma1-Flash_r12-fp32.safetensors",
                min_bytes=150_000_000,  # Aktual HF: 169 MB
                description="Chroma1-Flash r12 Fast CFG=1.0 Turbo LoRA (Aktual HF: 169 MB)",
            ),
        ],
    },
    "z_image": {
        "engine_id": "z_image",
        "display_name": "6. Z-Image-Turbo 6B (Tongyi-MAI INT8 ConvRot + Qwen3 4B FP8 · Bilingual Text & Photo)",
        "short_name": "Z-Image-Turbo 6B",
        "is_video": False,
        "supports_ref_images": False,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_1_speed",
        "approx_size_gb": 12.16,
        "defaults": {
            "unet": "z_image_turbo_int8_convrot.safetensors",
            "text_encoder": "qwen_3_4b_fp8_mixed.safetensors",
            "vae": "ae.safetensors",
            "turbo_lora": None,
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="z_image_turbo_int8_convrot.safetensors",
                url="https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
                min_bytes=5_800_000_000,  # Aktual HF: 6.20 GB
                description="Tongyi Z-Image-Turbo 6B INT8 ConvRot (Aktual HF: 6.20 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="qwen_3_4b_fp8_mixed.safetensors",
                url="https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
                min_bytes=5_200_000_000,  # Aktual HF: 5.63 GB
                description="Qwen3 4B FP8 Mixed Lumina2 Text Encoder (Aktual HF: 5.63 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="ae.safetensors",
                url="https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/vae/ae.safetensors",
                min_bytes=300_000_000,  # Aktual HF: 335 MB
                shared_with=("chroma", "z_image"),
                description="FLUX AutoEncoder VAE (Aktual HF: 335 MB — Shared dengan Chroma1-HD)",
            ),
        ],
    },
    "wan21": {
        "engine_id": "wan21",
        "display_name": "7. Wan 2.1 I2V 14B 480P (Image-to-Video GGUF Q4_K_S + Lightx2v 4-Step)",
        "short_name": "Wan 2.1 I2V 14B",
        "is_video": True,
        "supports_ref_images": True,
        "supports_negative_prompt": True,
        "default_memory_profile": "profile_2_lowram",
        "approx_size_gb": 16.31,
        "defaults": {
            "unet": "wan2.1-i2v-14b-480p-Q4_K_S.gguf",
            "text_encoder": "umt5-xxl-encoder-Q4_K_M.gguf",
            "vae": "wan_2.1_vae.safetensors",
            "clip_vision": "clip_vision_h.safetensors",
            "turbo_lora": "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        },
        "files": [
            ModelFileSpec(
                role="diffusion",
                subdir="diffusion_models",
                filename="wan2.1-i2v-14b-480p-Q4_K_S.gguf",
                url="https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_K_S.gguf",
                min_bytes=9_500_000_000,  # Aktual HF: 10.4 GB
                description="Wan 2.1 I2V 14B 480P GGUF Q4_K_S (Aktual HF: 10.4 GB)",
            ),
            ModelFileSpec(
                role="text_encoder",
                subdir="text_encoders",
                filename="umt5-xxl-encoder-Q4_K_M.gguf",
                url="https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf",
                min_bytes=3_000_000_000,  # Aktual HF: 3.66 GB
                description="UMT5-XXL Encoder GGUF Q4_K_M (Aktual HF: 3.66 GB)",
            ),
            ModelFileSpec(
                role="lora",
                subdir="loras",
                filename="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                url="https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/resolve/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                min_bytes=650_000_000,  # Aktual HF: 739 MB
                description="Lightx2v 4-Step CFG=1.0 Distill LoRA (Aktual HF: 739 MB)",
            ),
            ModelFileSpec(
                role="clip_vision",
                subdir="clip_vision",
                filename="clip_vision_h.safetensors",
                url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
                min_bytes=1_100_000_000,  # Aktual HF: 1.26 GB
                description="CLIP Vision H Image Encoder (Aktual HF: 1.26 GB)",
            ),
            ModelFileSpec(
                role="vae",
                subdir="vae",
                filename="wan_2.1_vae.safetensors",
                url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
                min_bytes=200_000_000,  # Aktual HF: 254 MB
                description="Wan 2.1 3D Causal VAE (Aktual HF: 254 MB)",
            ),
        ],
    },
}

# ==============================================================================
# B. 3 MODEL UPSCALER HEMAT MEMORI (SPANDREL TILED UPSCALER <350MB VRAM)
# ==============================================================================
UPSCALER_CATALOG: Dict[str, ModelFileSpec] = {
    "4x-UltraSharp.safetensors": ModelFileSpec(
        role="upscaler",
        subdir="upscale_models",
        filename="4x-UltraSharp.safetensors",
        url="https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors",
        min_bytes=60_000_000,  # Aktual HF: 66.9 MB
        description="4x-UltraSharp (Default Foto/Realisme · Aktual HF: 66.9 MB)",
    ),
    "4x-AnimeSharp.safetensors": ModelFileSpec(
        role="upscaler",
        subdir="upscale_models",
        filename="4x-AnimeSharp.safetensors",
        url="https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.safetensors",
        min_bytes=30_000_000,  # Aktual HF: 33.5 MB FP16
        description="4x-AnimeSharp (Default Anime/Ilustrasi Detail · Aktual HF: 33.5 MB FP16)",
    ),
    "RealESRGAN_x4plus_anime_6B.pth": ModelFileSpec(
        role="upscaler",
        subdir="upscale_models",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url="https://huggingface.co/amd/realesrgan-x4plus-anime-6b/resolve/main/RealESRGAN_x4plus_anime_6B.pth",
        min_bytes=15_000_000,  # Aktual HF: 17.9 MB
        description="RealESRGAN_x4plus_anime_6B (Ultra-Light 6-Block RRDB · Aktual HF: 17.9 MB)",
    ),
}

# Daftar file yang dibagi antar-engine
SHARED_FILE_KEYS: Set[str] = {
    "vae/qwen_image_vae.safetensors",
    "vae/ae.safetensors",
}


# ==============================================================================
# C. UTILITAS VALIDASI FILE, DISK & DETEKSI ON-DEMAND
# ==============================================================================
def ensure_model_directories(models_root: Path = MODELS_DIR) -> None:
    """Membuat seluruh subfolder model ComfyUI jika belum tersedia."""
    for sub in MODEL_SUBDIRS:
        (models_root / sub).mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def check_file_installed(spec: ModelFileSpec, models_root: Path = MODELS_DIR) -> Tuple[bool, Optional[Path]]:
    """
    Memeriksa apakah file model sudah ada di salah satu candidate_paths
    dan memiliki ukuran >= spec.min_bytes.
    """
    for candidate in spec.candidate_paths(models_root):
        if candidate.is_file() and candidate.stat().st_size >= spec.min_bytes:
            return True, candidate
    return False, None


def get_engine_files(engine_id: str) -> List[ModelFileSpec]:
    """Mengembalikan daftar ModelFileSpec untuk engine tertentu."""
    if engine_id not in ENGINE_CATALOG:
        raise KeyError(f"Engine ID '{engine_id}' tidak terdaftar di ENGINE_CATALOG.")
    return list(ENGINE_CATALOG[engine_id]["files"])  # type: ignore[arg-type]


def check_engine_installed(engine_id: str, models_root: Path = MODELS_DIR) -> Tuple[bool, List[ModelFileSpec]]:
    """
    Mengecek apakah seluruh bobot wajib untuk `engine_id` sudah terinstal di disk.
    Mengembalikan (semua_terinstal, daftar_file_yang_belum_ada).
    """
    specs = get_engine_files(engine_id)
    missing: List[ModelFileSpec] = []
    for spec in specs:
        ok, _ = check_file_installed(spec, models_root)
        if not ok:
            missing.append(spec)
    return len(missing) == 0, missing


def get_other_engine_exclusive_files(keep_engine_id: str, models_root: Path = MODELS_DIR) -> List[Path]:
    """
    Mengembalikan daftar file model di disk yang dimiliki oleh engine LAIN,
    tetapi TIDAK dipakai oleh `keep_engine_id` (aman untuk dihapus demi menghemat disk Colab).
    """
    keep_filenames: Set[str] = {spec.filename for spec in get_engine_files(keep_engine_id)}
    removable: List[Path] = []
    seen_paths: Set[Path] = set()

    for eng_id, meta in ENGINE_CATALOG.items():
        if eng_id == keep_engine_id:
            continue
        for spec in meta["files"]:  # type: ignore[union-attr]
            if spec.filename in keep_filenames:
                continue
            for candidate in spec.candidate_paths(models_root):
                if candidate.is_file() and candidate not in seen_paths:
                    removable.append(candidate)
                    seen_paths.add(candidate)
    return removable


def list_local_models(role_or_subdir: str, models_root: Path = MODELS_DIR, min_bytes: int = 100_000) -> List[str]:
    """
    Memindai isi folder model di disk (mendukung penyatuan diffusion_models<->unet dan text_encoders<->clip).
    """
    if role_or_subdir in ("diffusion", "diffusion_models", "unet"):
        folders = ["diffusion_models", "unet"]
    elif role_or_subdir in ("text_encoder", "text_encoders", "clip"):
        folders = ["text_encoders", "clip"]
    else:
        folders = [role_or_subdir]

    valid_exts = {".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".bin"}
    found: Set[str] = set()
    for sub in folders:
        fdir = models_root / sub
        if not fdir.exists():
            continue
        for p in fdir.iterdir():
            if (
                p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in valid_exts
                and p.stat().st_size >= min_bytes
            ):
                found.add(p.name)
    return sorted(found)


def get_disk_usage_gb(target_path: Path = BASE_DIR) -> Tuple[float, float, float]:
    """Mengembalikan (used_gb, free_gb, total_gb) untuk partisi disk."""
    check_dir = target_path if target_path.exists() else Path.cwd()
    usage = shutil.disk_usage(str(check_dir))
    used_gb = usage.used / (1024**3)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return used_gb, free_gb, total_gb
