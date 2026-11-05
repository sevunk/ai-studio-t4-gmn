# -*- coding: utf-8 -*-
"""
Unified Session Gallery & Fast Thumbnail Metadata Inspector (`core/gallery.py`)
Mengelola penyimpanan, pembuatan thumbnail ringan (`*.thumb.jpg` ~20 KB agar Galeri
termuat instan tanpa mendownload PNG 5-15 MB per item), pembacaan riwayat,
ekstraksi metadata, penghapusan item, dan pembuatan arsip `.zip` untuk Gambar (`.png`)
maupun Video (`.mp4` + thumbnail `.jpg`).
"""

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from config.catalog import BASE_DIR, GALLERY_DIR

THUMB_MAX_SIZE = (420, 420)
THUMB_JPEG_QUALITY = 84


class GalleryManager:
    """Manajer riwayat terpadu Gambar (.png + fast .thumb.jpg) & Video (.mp4 + .jpg) + Metadata (.json)."""

    def __init__(self, gallery_dir: Path = GALLERY_DIR) -> None:
        self.gallery_dir = gallery_dir
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_image_thumbnail(img_path: Path) -> Path:
        """
        Membuat atau mengambil file thumbnail JPEG ringan (`{stem}.thumb.jpg`, ~20 KB)
        dari file gambar penuh (`{stem}.png`) agar `gr.Gallery` memuat 20x lebih cepat.
        Jika pembuatan thumbnail gagal, otomatis fallback ke `img_path` asli.
        """
        thumb_path = img_path.with_name(f"{img_path.stem}.thumb.jpg")
        if thumb_path.is_file() and thumb_path.stat().st_size > 500:
            return thumb_path
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail(THUMB_MAX_SIZE, Image.Resampling.LANCZOS)
                rgb.save(thumb_path, format="JPEG", quality=THUMB_JPEG_QUALITY, optimize=True)
            return thumb_path
        except Exception:
            return img_path

    def save_image_item(
        self,
        image_bytes: bytes,
        engine_id: str,
        seed: int,
        metadata: Dict[str, Any],
        ext: str = ".png",
    ) -> Tuple[Path, Path]:
        """Menyimpan file gambar penuh (.png), thumbnail cepat (.thumb.jpg), dan metadata (.json)."""
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{engine_id}_{ts}_{seed}_{uuid.uuid4().hex[:5]}"
        img_path = self.gallery_dir / f"{stem}{ext}"
        json_path = self.gallery_dir / f"{stem}.json"

        img_path.write_bytes(image_bytes)
        thumb_path = self._ensure_image_thumbnail(img_path)

        record = dict(metadata)
        record.update(
            {
                "media_type": "image",
                "file_name": img_path.name,
                "media_path": str(img_path),
                "preview_path": str(thumb_path),
                "engine_id": engine_id,
                "seed": seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return img_path, json_path

    def save_video_item(
        self,
        mp4_src_path: Path,
        thumbnail_jpg_src_path: Optional[Path],
        engine_id: str,
        seed: int,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, Optional[Path], Path]:
        """Menyimpan video (.mp4), thumbnail (.jpg), dan metadata (.json) di dalam folder galeri."""
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{engine_id}_{ts}_{seed}_{uuid.uuid4().hex[:5]}"
        mp4_dst = self.gallery_dir / f"{stem}.mp4"
        jpg_dst = self.gallery_dir / f"{stem}.jpg"
        json_path = self.gallery_dir / f"{stem}.json"

        if mp4_src_path.resolve() != mp4_dst.resolve():
            mp4_dst.write_bytes(mp4_src_path.read_bytes())

        if thumbnail_jpg_src_path is not None and thumbnail_jpg_src_path.is_file():
            if thumbnail_jpg_src_path.resolve() != jpg_dst.resolve():
                jpg_dst.write_bytes(thumbnail_jpg_src_path.read_bytes())
        else:
            jpg_dst = None

        record = dict(metadata)
        record.update(
            {
                "media_type": "video",
                "file_name": mp4_dst.name,
                "media_path": str(mp4_dst),
                "preview_path": str(jpg_dst) if jpg_dst else str(mp4_dst),
                "engine_id": engine_id,
                "seed": seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return mp4_dst, jpg_dst, json_path

    def list_items(self) -> Tuple[List[Tuple[str, str]], List[str], str]:
        """
        Mengembalikan:
        - `gallery_tuples`: list `(fast_thumbnail_path, caption)` untuk ditampilkan instan di `gr.Gallery`
        - `media_paths`: list path file utama resolusi penuh (`.png` atau `.mp4`) sesuai urutan indeks
        - `summary_markdown`: ringkasan jumlah gambar & video di galeri
        """
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        json_files = sorted(
            [p for p in self.gallery_dir.glob("*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        gallery_tuples: List[Tuple[str, str]] = []
        media_paths: List[str] = []
        img_count = 0
        vid_count = 0

        for jf in json_files:
            try:
                meta = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue

            mtype = meta.get("media_type", "image")
            eng = str(meta.get("engine_id", "studio")).upper()
            seed = meta.get("seed", "-")

            if mtype == "video":
                mp4_file = jf.with_suffix(".mp4")
                jpg_file = jf.with_suffix(".jpg")
                if not mp4_file.is_file():
                    continue
                preview = str(jpg_file) if jpg_file.is_file() else str(mp4_file)
                fps = meta.get("fps", 16)
                dur = meta.get("duration_sec", "")
                caption = f"🎬 {eng} · {dur}s @ {fps}fps · seed {seed}"
                gallery_tuples.append((preview, caption))
                media_paths.append(str(mp4_file))
                vid_count += 1
            else:
                img_file = None
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    cand = jf.with_suffix(ext)
                    if cand.is_file() and not cand.name.endswith(".thumb.jpg"):
                        img_file = cand
                        break
                if img_file is None:
                    continue
                thumb_file = self._ensure_image_thumbnail(img_file)
                caption = f"🖼️ {eng} · {meta.get('width', '')}×{meta.get('height', '')} · seed {seed}"
                gallery_tuples.append((str(thumb_file), caption))
                media_paths.append(str(img_file))
                img_count += 1

        summary = (
            f"🗂️ **Galeri Sesi (`{self.gallery_dir}`)**: "
            f"**{img_count} Gambar** & **{vid_count} Video MP4** tersimpan *(Fast Thumbnail Cache Aktif)*."
        )
        return gallery_tuples, media_paths, summary

    def get_metadata_for_media(self, media_path_str: str) -> Dict[str, Any]:
        """Membaca file `.json` pendamping untuk path gambar/video yang dipilih."""
        if not media_path_str:
            return {}
        p = Path(media_path_str)
        jf = p.with_suffix(".json")
        if not jf.is_file():
            return {}
        try:
            return json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}

    @staticmethod
    def format_metadata_markdown(meta: Dict[str, Any]) -> str:
        """Menyusun tabel Markdown lengkap untuk History Inspector."""
        if not meta:
            return "Klik salah satu item di Galeri untuk melihat detail Prompt, Model, Seed, dan Waktu Render."

        mtype = "🎬 Video MP4" if meta.get("media_type") == "video" else "🖼️ Gambar PNG"
        loras = meta.get("loras", []) or meta.get("custom_loras", [])
        lora_str = (
            ", ".join(f"`{x.get('name')}` ({x.get('strength')})" for x in loras)
            if loras
            else "Tidak ada"
        )
        upscale_info = meta.get("upscale_info", "Tidak di-upscale")

        rows = [
            f"### 📋 Metadata Lengkap ({mtype})",
            "| Parameter | Nilai |",
            "| :--- | :--- |",
            f"| **File & Waktu** | `{meta.get('file_name', '-')}` · `{meta.get('timestamp', '-')}` |",
            f"| **Engine Utama** | **{meta.get('engine_display', meta.get('engine_id', '-'))}** |",
            f"| **Waktu Render** | **{meta.get('total_elapsed_sec', meta.get('elapsed_sec', '-'))} detik** (T1: `{meta.get('t_stage1', '-')}s`, T2: `{meta.get('t_stage2', '-')}s`, T3: `{meta.get('t_stage3', '-')}s`) |",
            f"| **Diffusion / UNet** | `{meta.get('unet_name', '-')}` |",
            f"| **Text Encoder & VAE** | `{meta.get('clip_name', '-')}` / `{meta.get('vae_name', '-')}` |",
            f"| **Resolusi & Seed** | `{meta.get('width', '-')} × {meta.get('height', '-')}` · **Seed:** `{meta.get('seed', '-')}` |",
            f"| **Sampling** | Steps: `{meta.get('steps', '-')}` · CFG: `{meta.get('cfg', '-')}` · Shift: `{meta.get('shift', '-')}` · `{meta.get('sampler', '-')}` / `{meta.get('scheduler', '-')}` |",
            f"| **LoRA Aktif** | {lora_str} |",
            f"| **Tiled Upscaler** | {upscale_info} |",
        ]
        if meta.get("media_type") == "video":
            rows.append(
                f"| **Durasi & Frame Video** | `{meta.get('frames', 49)}` Base Frames = **{meta.get('duration_sec', '-')} detik** @ `{meta.get('fps', 16)} FPS` |"
            )

        rows.append(f"\n**Positive Prompt:**\n```text\n{meta.get('prompt', '')}\n```")
        if meta.get("negative_prompt"):
            rows.append(f"**Negative Prompt:**\n```text\n{meta.get('negative_prompt', '')}\n```")
        return "\n".join(rows)

    def delete_item(self, media_path_str: str) -> bool:
        """Menghapus file gambar/video beserta thumbnail `.jpg`/`.thumb.jpg` dan metadata `.json`-nya."""
        if not media_path_str:
            return False
        base = Path(media_path_str)
        deleted = False
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".json"):
            cand = base.with_suffix(ext)
            if cand.is_file():
                try:
                    cand.unlink(missing_ok=True)
                    deleted = True
                except Exception:
                    pass
        thumb_cand = base.with_name(f"{base.stem}.thumb.jpg")
        if thumb_cand.is_file():
            try:
                thumb_cand.unlink(missing_ok=True)
                deleted = True
            except Exception:
                pass
        return deleted

    def export_all_to_zip(self) -> Optional[Path]:
        """Membungkus seluruh isi asli `/content/Studio_Gallery` (tanpa file cache `.thumb.jpg`) ke dalam satu file `.zip`."""
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".json"}
        files = [
            p
            for p in self.gallery_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in valid_exts
                and not p.name.startswith("_raw_")
                and not p.name.endswith(".thumb.jpg")
            )
        ]
        if not files:
            return None

        zip_dir = BASE_DIR if BASE_DIR.exists() else Path.cwd()
        zip_path = zip_dir / f"Studio_Gallery_Export_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        return zip_path
