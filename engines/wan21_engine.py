# -*- coding: utf-8 -*-
"""
6. Wan 2.1 I2V 14B 480P Video Engine (`engines/wan21_engine.py`)
Spesifikasi:
- Resolusi kelipatan 16 (rentang 320–832, default 640x368).
- Jumlah Base Frame wajib rumus `((4 * k) + 1)` (rentang 17–81, default 49 frame = 3.1 detik).
- Smart Aspect Ratio & Center-Crop (`ImageOps.fit` Lanczos): saat gambar diunggah, hitung
  ukuran 480P optimal sesuai rasio asli gambar agar subjek tidak pernah gepeng/distorsi.
- Nodes: CLIPLoaderGGUF (`umt5-xxl-encoder-Q4_K_M.gguf`, `type="wan"`), CLIPTextEncode (pos),
  ConditioningZeroOut (jika cfg <= 1.0) atau CLIPTextEncode (neg), CLIPVisionLoader (`clip_vision_h.safetensors`),
  CLIPVisionEncode, VAELoader (`wan_2.1_vae.safetensors`), WanImageToVideo,
  UnetLoaderGGUF (`wan2.1-i2v-14b-480p-Q4_K_S.gguf`), LoraLoaderModelOnly untuk `Lightx2v`
  (default strength 1.0, atau 0.85 saat Auto-Tune Custom LoRA) diikuti hingga 3 Custom LoRA,
  ModelSamplingSD3 (`shift=5.0`), KSampler (`steps=4`, `cfg=1.0`, `uni_pc`, `simple`),
  VAEDecodeTiled (`tile_size=256, overlap=64, temporal_size=64, temporal_overlap=8`).
- True-Duration Video Encoding (Anti Speed-Up): simpan frame hasil decode selalu pada
  16 FPS native terlebih dahulu (`durasi = frames / 16.0` detik). Jika user memilih output
  24 FPS atau 32 FPS, jalankan `ffmpeg -vf minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1`
  (dengan fallback `mi_mode=blend`) sehingga gerakan menjadi sangat halus tanpa mempercepat/memotong durasi video!
"""

import io
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image, ImageOps

from config.catalog import BASE_DIR, ENGINE_CATALOG, GALLERY_DIR
from engines.base import BaseEngine

LIGHTX2V_LORA = "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
NATIVE_WAN_FPS = 16

WAN21_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "16:9 Landscape 480P (640 × 368) — Default Wan 2.1": (640, 368),
    "9:16 Vertical 480P (368 × 640)": (368, 640),
    "1:1 Square 480P (480 × 480)": (480, 480),
    "4:3 Classic Landscape (576 × 416)": (576, 416),
    "3:4 Classic Portrait (416 × 576)": (416, 576),
    "16:9 Wide 480P Max (832 × 480)": (832, 480),
}

WAN21_FPS_CHOICES = [
    "16 FPS Native (Instan · 100% Akurat)",
    "24 FPS Smooth (Motion Interpolated · Durasi Tetap)",
    "32 FPS Ultra-Smooth (Motion Interpolated · Durasi Tetap)",
]

QUICK_MOTION_TAGS: Dict[str, str] = {
    "+ Slow Push-In": "slow cinematic camera push-in, subtle depth of field",
    "+ Orbit Shot": "smooth camera orbit movement around the subject, parallax effect",
    "+ Smooth Pan": "smooth horizontal camera pan, fluid motion",
    "+ Wind & Hair": "gentle breeze blowing hair and clothes naturally, lifelike micro-movements",
    "+ Handheld Realism": "natural handheld camera movement, photorealistic lighting and motion",
}

WAN21_DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


class Wan21Engine(BaseEngine):
    """Workflow Engine untuk Wan 2.1 I2V 14B 480P (GGUF Q4_K_S + Lightx2v + True-Duration Minterpolate)."""

    engine_id = "wan21"
    display_name = str(ENGINE_CATALOG["wan21"]["display_name"])
    is_video = True
    supports_ref_images = True
    supports_negative_prompt = True

    @property
    def default_params(self) -> Dict[str, Any]:
        defaults = self.get_catalog_defaults()
        return {
            "unet_name": defaults["unet"],
            "clip_name": defaults["text_encoder"],
            "vae_name": defaults["vae"],
            "clip_vision_name": defaults["clip_vision"],
            "default_preset": "⚡ Extreme Turbo (8-Step / 4-Step LoRA)",
            "steps": 4,
            "cfg": 1.0,
            "shift": 5.0,
            "sampler": "uni_pc",
            "scheduler": "simple",
            "width": 640,
            "height": 368,
            "frames": 49,
            "fps_choice": WAN21_FPS_CHOICES[0],
            "lightx2v_strength": 1.0,
            "use_lazycache": False,
            "use_slg": False,
            "skip_layer": 22,
            "negative_prompt": WAN21_DEFAULT_NEGATIVE,
            "auto_aspect_crop": True,
            "aspect_ratios": WAN21_ASPECT_RATIOS,
            "default_aspect": "16:9 Landscape 480P (640 × 368) — Default Wan 2.1",
            "presets": {
                "⚡ Extreme Turbo (8-Step / 4-Step LoRA)": {
                    "steps": 4,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "sampler": "uni_pc",
                    "scheduler": "simple",
                    "lightx2v_strength": 1.0,
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🚀 Speed (Balanced + LazyCache)": {
                    "steps": 6,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "sampler": "uni_pc",
                    "scheduler": "simple",
                    "lightx2v_strength": 0.85,
                    "use_lazycache": False,
                    "use_slg": False,
                },
                "🎯 High Quality": {
                    "steps": 8,
                    "cfg": 1.5,
                    "shift": 5.0,
                    "sampler": "uni_pc",
                    "scheduler": "simple",
                    "lightx2v_strength": 0.85,
                    "use_lazycache": False,
                    "use_slg": False,
                },
            },
            "placeholder": (
                "Cinematic shot, smooth natural motion, realistic physics, the subject moves gently, "
                "high detail, 4k"
            ),
        }

    @staticmethod
    def snap_wan_frames(frames: int) -> int:
        """Memastikan jumlah Base Frame mengikuti rumus wajib `((4 * k) + 1)` dalam rentang 17..81."""
        clamped = max(17, min(81, int(frames)))
        return int(((clamped - 1) // 4) * 4 + 1)

    @staticmethod
    def parse_target_fps(fps_val: Any) -> int:
        """Mengekstrak angka FPS (16, 24, atau 32) dari pilihan dropdown."""
        if isinstance(fps_val, (int, float)):
            val = int(fps_val)
        else:
            m = re.search(r"(\d+)", str(fps_val))
            val = int(m.group(1)) if m else NATIVE_WAN_FPS
        if val >= 30:
            return 32
        if val >= 20:
            return 24
        return 16

    @staticmethod
    def compute_optimal_480p_size(
        orig_w: int,
        orig_h: int,
        target_pixels: int = 640 * 368,
    ) -> Tuple[int, int]:
        """Menghitung ukuran 480P (kelipatan 16, rentang 320..832) sesuai rasio asli gambar."""
        aspect = orig_w / max(1, orig_h)
        h = int(round(((target_pixels / aspect) ** 0.5) / 16) * 16)
        w = int(round((h * aspect) / 16) * 16)
        w = max(320, min(832, w))
        h = max(320, min(832, h))
        return w, h

    @staticmethod
    def smart_center_crop(pil_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Center-crop menggunakan `ImageOps.fit` Lanczos agar subjek tidak pernah gepeng/distorsi."""
        return ImageOps.fit(
            pil_img.convert("RGB"),
            (int(target_w), int(target_h)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(self.default_params)
        clean.update({k: v for k, v in params.items() if v is not None})

        clean["width"] = self.snap_multiple(int(clean.get("width", 640)), mult=16, min_val=320, max_val=832)
        clean["height"] = self.snap_multiple(int(clean.get("height", 368)), mult=16, min_val=320, max_val=832)
        clean["frames"] = self.snap_wan_frames(int(clean.get("frames", 49)))
        clean["steps"] = max(1, min(30, int(clean.get("steps", 4))))
        clean["cfg"] = max(1.0, min(6.0, float(clean.get("cfg", 1.0))))
        clean["shift"] = max(1.0, min(10.0, float(clean.get("shift", 5.0))))
        clean["lightx2v_strength"] = max(0.0, min(1.5, float(clean.get("lightx2v_strength", 1.0))))
        clean["target_fps"] = self.parse_target_fps(clean.get("fps_choice", WAN21_FPS_CHOICES[0]))
        clean["duration_sec"] = round(clean["frames"] / float(NATIVE_WAN_FPS), 1)
        clean["seed"] = int(clean.get("seed", 42))
        return clean

    def prepare_start_frame(self, start_image_input: Any, width: int, height: int, auto_crop: bool = True) -> Tuple[str, int, int]:
        """
        Memuat gambar frame awal, menghitung rasio 480P optimal jika `auto_crop=True`,
        melakukan center-crop `ImageOps.fit` Lanczos, dan menyimpannya ke `ComfyUI/input/`.
        """
        if start_image_input is None:
            raise ValueError("Wan 2.1 I2V membutuhkan Gambar Frame Awal (Start Frame). Unggah di tab Input Image.")

        if isinstance(start_image_input, (str, Path)):
            pil_img = Image.open(start_image_input).convert("RGB")
        elif isinstance(start_image_input, Image.Image):
            pil_img = start_image_input.convert("RGB")
        else:
            raise TypeError(f"Tipe start_image tidak didukung: {type(start_image_input)}")

        if auto_crop:
            target_w, target_h = self.compute_optimal_480p_size(
                pil_img.width,
                pil_img.height,
                target_pixels=int(width) * int(height),
            )
        else:
            target_w = self.snap_multiple(width, mult=16, min_val=320, max_val=832)
            target_h = self.snap_multiple(height, mult=16, min_val=320, max_val=832)

        cropped_pil = self.smart_center_crop(pil_img, target_w, target_h)
        staged_filename = self.stage_input_image(cropped_pil, prefix="wan21_start")
        return staged_filename, target_w, target_h

    def build_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = self.sanitize_params(params)
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt gerakan (motion prompt) tidak boleh kosong.")
        negative = str(p.get("negative_prompt", "")).strip()

        start_img_input = p.get("start_image") or p.get("input_image")
        staged_start_name, final_w, final_h = self.prepare_start_frame(
            start_img_input,
            width=p["width"],
            height=p["height"],
            auto_crop=bool(p.get("auto_aspect_crop", True)),
        )
        p["width"], p["height"] = final_w, final_h

        unet_name = str(p["unet_name"])
        clip_name = str(p["clip_name"])
        vae_name = str(p["vae_name"])
        clip_vision_name = str(p.get("clip_vision_name", "clip_vision_h.safetensors"))

        nodes: Dict[str, Any] = {
            "1": {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {
                    "clip_name": clip_name,
                    "type": "wan",
                },
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 0],
                    "text": prompt,
                },
            },
        }

        if float(p["cfg"]) <= 1.0:
            nodes["3"] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["2", 0],
                },
            }
        else:
            nodes["3"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 0],
                    "text": negative,
                },
            }

        nodes["4"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": staged_start_name,
            },
        }
        nodes["5"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": clip_vision_name,
            },
        }
        nodes["6"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["5", 0],
                "image": ["4", 0],
                "crop": "none",
            },
        }
        nodes["7"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        }
        nodes["8"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "vae": ["7", 0],
                "width": int(final_w),
                "height": int(final_h),
                "length": int(p["frames"]),
                "batch_size": 1,
                "start_image": ["4", 0],
                "clip_vision_output": ["6", 0],
            },
        }
        nodes["9"] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": unet_name,
            },
        }

        model_ref: List[Any] = ["9", 0]

        # Pasang Lightx2v Accelerator LoRA terlebih dahulu
        lx_strength = float(p.get("lightx2v_strength", 1.0))
        if lx_strength > 0:
            nodes["19"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_ref,
                    "lora_name": LIGHTX2V_LORA,
                    "strength_model": lx_strength,
                },
            }
            model_ref = ["19", 0]

        # Rantai hingga 3 Custom LoRA
        custom_loras = self.filter_custom_loras(
            p.get("lora_slots", []),
            exclude_names=[LIGHTX2V_LORA],
        )
        for idx, (lora_name, strength) in enumerate(custom_loras):
            node_id = str(20 + idx)
            nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_ref,
                    "lora_name": lora_name,
                    "strength_model": float(strength),
                },
            }
            model_ref = [node_id, 0]

        nodes["10"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": model_ref,
                "shift": float(p["shift"]),
            },
        }
        nodes["11"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "seed": int(p["seed"]),
                "steps": int(p["steps"]),
                "cfg": float(p["cfg"]),
                "sampler_name": str(p["sampler"]),
                "scheduler": str(p["scheduler"]),
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2],
                "denoise": 1.0,
            },
        }
        nodes["12"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["7", 0],
                "tile_size": 256,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        }
        nodes["13"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Wan21_Frame",
                "images": ["12", 0],
            },
        }
        return nodes

    @classmethod
    def encode_frames_to_mp4(
        cls,
        frame_bytes_list: List[bytes],
        target_fps: int = 16,
        output_dir: Path = GALLERY_DIR,
        seed: int = 42,
    ) -> Tuple[Path, Path, float]:
        """
        True-Duration Video Encoding (Anti Speed-Up):
        1. Menyimpan seluruh frame hasil decode pada `16 FPS` native terlebih dahulu
           (`durasi = len(frames) / 16.0` detik).
        2. Jika `target_fps` adalah `24` atau `32`, menjalankan:
           `ffmpeg -vf minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1`
           (dengan fallback `mi_mode=blend`) agar gerakan ekstra mulus tanpa mempercepat durasi video.
        Mengembalikan `(final_mp4_path, thumbnail_jpg_path, true_duration_sec)`.
        """
        if not frame_bytes_list:
            raise RuntimeError("Tidak ada frame video yang dihasilkan dari VAEDecodeTiled.")

        output_dir.mkdir(parents=True, exist_ok=True)
        frames_np: List[np.ndarray] = []
        first_pil: Optional[Image.Image] = None

        for raw in frame_bytes_list:
            pil_f = Image.open(io.BytesIO(raw)).convert("RGB")
            if first_pil is None:
                first_pil = pil_f
            frames_np.append(np.array(pil_f, dtype=np.uint8))

        true_duration_sec = round(len(frames_np) / float(NATIVE_WAN_FPS), 2)
        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:5]
        stem = f"wan21_{ts}_{seed}_{uid}"

        raw_16fps_mp4 = output_dir / f"_raw_{stem}.mp4"
        final_mp4 = output_dir / f"{stem}.mp4"
        thumb_jpg = output_dir / f"{stem}.jpg"

        if first_pil is not None:
            first_pil.save(str(thumb_jpg), quality=90)

        # Langkah 1: Selalu encode pada 16 FPS native terlebih dahulu
        imageio.mimsave(str(raw_16fps_mp4), frames_np, fps=NATIVE_WAN_FPS, codec="libx264", quality=8)

        # Langkah 2: Motion Interpolation ke 24 / 32 FPS jika diminta
        clean_fps = cls.parse_target_fps(target_fps)
        if clean_fps <= NATIVE_WAN_FPS or shutil.which("ffmpeg") is None:
            shutil.move(str(raw_16fps_mp4), str(final_mp4))
            return final_mp4, thumb_jpg, true_duration_sec

        mci_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_16fps_mp4),
            "-vf",
            f"minterpolate=fps={clean_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(final_mp4),
        ]
        res = subprocess.run(mci_cmd, capture_output=True, text=True)
        if res.returncode != 0 or not final_mp4.is_file():
            blend_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw_16fps_mp4),
                "-vf",
                f"minterpolate=fps={clean_fps}:mi_mode=blend",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(final_mp4),
            ]
            subprocess.run(blend_cmd, check=False)

        if not final_mp4.is_file():
            shutil.move(str(raw_16fps_mp4), str(final_mp4))
        else:
            raw_16fps_mp4.unlink(missing_ok=True)

        return final_mp4, thumb_jpg, true_duration_sec
