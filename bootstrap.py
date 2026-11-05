#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap Script — Unified Multi-Engine AI Studio (T4 Edition)
Menginstal paket sistem (aria2c, ffmpeg, git-lfs), meng-clone ComfyUI,
memasang extra_model_paths.yaml, menginstal ComfyUI_GGUF (beserta patch 'krea2'),
menginstal custom node resmi OpenMOSS MOSS-Image, dan memasang dependensi Python
tanpa menimpa PyTorch/CUDA bawaan Google Colab T4.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.catalog import (
    BASE_DIR,
    COMFY_DIR,
    ENGINE_CATALOG,
    MODELS_DIR,
    UPSCALER_CATALOG,
    check_file_installed,
    ensure_model_directories,
    get_disk_usage_gb,
)


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd))
    for secret_key in ("HF_TOKEN", "CIVITAI_TOKEN", "MODELSCOPE_TOKEN"):
        tok = os.environ.get(secret_key, "").strip()
        if tok:
            printable = printable.replace(tok, f"***{secret_key}***")
    print(f"$ {printable}", flush=True)
    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def verify_cuda_gpu(require_gpu: bool = False) -> None:
    """Memeriksa ketersediaan GPU CUDA (Tesla T4) dan menampilkan telemetri awal."""
    try:
        import torch
    except Exception as exc:
        if require_gpu:
            raise RuntimeError(f"PyTorch gagal diimpor dari runtime: {exc}") from exc
        print(f"[WARN] PyTorch belum terpasang: {exc}", flush=True)
        return

    print(f"Python  : {sys.version.split()[0]}", flush=True)
    print(f"PyTorch : {torch.__version__} | CUDA build: {torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        msg = (
            "⚠️ PERINGATAN: CUDA GPU tidak terdeteksi! Pastikan di Google Colab Anda telah memilih: "
            "Runtime -> Change runtime type -> Hardware accelerator: T4 GPU."
        )
        if require_gpu:
            raise RuntimeError(msg)
        print(msg, flush=True)
        return

    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / (1024**3)
    _, free_gb, total_gb = get_disk_usage_gb(BASE_DIR)
    print(
        f"GPU     : {torch.cuda.get_device_name(0)} | VRAM: {vram_gib:.2f} GiB | "
        f"Disk Kosong: {free_gb:.2f} / {total_gb:.2f} GiB",
        flush=True,
    )


def install_system_packages() -> None:
    """Memastikan aria2c, ffmpeg, dan git-lfs terpasang di sistem Linux Colab."""
    missing_bins = [b for b in ("aria2c", "ffmpeg", "git-lfs") if shutil.which(b) is None]
    if not missing_bins:
        print("[OK] aria2c, ffmpeg, dan git-lfs sudah terpasang.", flush=True)
        return

    if shutil.which("apt-get") is None:
        print(f"[WARN] apt-get tidak tersedia; lewati instalasi sistem untuk: {missing_bins}", flush=True)
        return

    print(f"[SETUP] Menginstal paket sistem ({', '.join(missing_bins)})...", flush=True)
    run_cmd(["apt-get", "update", "-qq"], check=False)
    run_cmd(["apt-get", "install", "-y", "-qq", "aria2", "ffmpeg", "git-lfs"], check=False)
    run_cmd(["git", "lfs", "install", "--skip-repo"], check=False)


def setup_comfyui_core(comfy_dir: Path = COMFY_DIR) -> None:
    """Meng-clone ComfyUI dan menyalin extra_model_paths.yaml."""
    if not (comfy_dir / "main.py").exists():
        if comfy_dir.exists():
            shutil.rmtree(comfy_dir, ignore_errors=True)
        print(f"[SETUP] Meng-clone ComfyUI ke {comfy_dir}...", flush=True)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)])
    else:
        print(f"[OK] ComfyUI sudah tersedia di {comfy_dir}.", flush=True)

    ensure_model_directories(comfy_dir / "models")
    (comfy_dir / "input").mkdir(parents=True, exist_ok=True)
    (comfy_dir / "output").mkdir(parents=True, exist_ok=True)
    (comfy_dir / "custom_nodes").mkdir(parents=True, exist_ok=True)

    # Salin & sesuaikan extra_model_paths.yaml agar menunjuk ke comfy_dir aktual
    src_yaml = REPO_ROOT / "extra_model_paths.yaml"
    dst_yaml = comfy_dir / "extra_model_paths.yaml"
    if src_yaml.exists():
        yaml_text = src_yaml.read_text(encoding="utf-8")
        yaml_text = yaml_text.replace("/content/ComfyUI/", f"{comfy_dir.resolve()}/")
        dst_yaml.write_text(yaml_text, encoding="utf-8")
        print(f"[OK] extra_model_paths.yaml dikonfigurasi di {dst_yaml}.", flush=True)


def patch_gguf_krea2_allowlist(gguf_node_dir: Path) -> None:
    """
    Memastikan ComfyUI_GGUF/loader.py menyertakan 'krea2' di dalam IMG_ARCH_LIST
    agar model Krea 2 Turbo 12B GGUF dapat dimuat tanpa error arsitektur.
    """
    loader_py = gguf_node_dir / "loader.py"
    if not loader_py.exists():
        return
    txt = loader_py.read_text(encoding="utf-8", errors="ignore")
    if '"krea2"' not in txt and "'krea2'" not in txt:
        patched = txt.replace("IMG_ARCH_LIST = {", 'IMG_ARCH_LIST = {"krea2", ')
        if patched != txt:
            loader_py.write_text(patched, encoding="utf-8")
            print("[OK] Berhasil mem-patch ComfyUI_GGUF/loader.py dengan arsitektur 'krea2'.", flush=True)


def setup_custom_nodes(comfy_dir: Path = COMFY_DIR) -> None:
    """Memasang ComfyUI_GGUF (dengan patch krea2) dan node resmi OpenMOSS MOSS-Image."""
    custom_nodes_dir = comfy_dir / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    # 1. ComfyUI_GGUF (gunakan underscore agar dapat diimpor sebagai modul Python jika diperlukan)
    gguf_dir = custom_nodes_dir / "ComfyUI_GGUF"
    legacy_gguf_dir = custom_nodes_dir / "ComfyUI-GGUF"
    if legacy_gguf_dir.exists() and not gguf_dir.exists():
        shutil.move(str(legacy_gguf_dir), str(gguf_dir))

    if not (gguf_dir / "__init__.py").exists():
        if gguf_dir.exists():
            shutil.rmtree(gguf_dir, ignore_errors=True)
        print("[SETUP] Meng-clone custom node city96/ComfyUI-GGUF...", flush=True)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/city96/ComfyUI-GGUF.git", str(gguf_dir)])
    patch_gguf_krea2_allowlist(gguf_dir)

    # Buat symlink ComfyUI-GGUF -> ComfyUI_GGUF jika belum ada untuk kompatibilitas penuh
    if not legacy_gguf_dir.exists():
        try:
            legacy_gguf_dir.symlink_to(gguf_dir, target_is_directory=True)
        except Exception:
            pass

    # 2. OpenMOSS MOSS-Image Custom Node
    moss_node_dst = custom_nodes_dir / "moss-image"
    if not (moss_node_dst / "__init__.py").exists():
        try:
            moss_repo_tmp = BASE_DIR / "MOSS-Image"
            if not (moss_repo_tmp / "comfyui" / "moss-image" / "__init__.py").exists():
                if moss_repo_tmp.exists():
                    shutil.rmtree(moss_repo_tmp, ignore_errors=True)
                print("[SETUP] Meng-clone OpenMOSS/MOSS-Image untuk ekstraksi node ComfyUI...", flush=True)
                run_cmd(
                    ["git", "clone", "--depth", "1", "https://github.com/OpenMOSS/MOSS-Image.git", str(moss_repo_tmp)],
                    check=True,
                )
            plugin_src = moss_repo_tmp / "comfyui" / "moss-image"
            if plugin_src.exists():
                if moss_node_dst.exists():
                    shutil.rmtree(moss_node_dst, ignore_errors=True)
                shutil.copytree(plugin_src, moss_node_dst)
                print(f"[OK] Custom node MOSS-Image terpasang di {moss_node_dst}.", flush=True)
        except Exception as exc:
            print(f"[WARN] Gagal memasang node opsional MOSS-Image: {exc}", flush=True)
    else:
        print(f"[OK] Custom node MOSS-Image sudah tersedia di {moss_node_dst}.", flush=True)


def install_python_requirements(comfy_dir: Path = COMFY_DIR) -> None:
    """
    Menginstal requirements ComfyUI dan requirements Studio TANPA menimpa
    paket PyTorch/CUDA/Triton/xFormers bawaan Google Colab.
    """
    excluded = {"torch", "torchvision", "torchaudio", "triton", "xformers"}
    comfy_req = comfy_dir / "requirements.txt"
    if comfy_req.exists():
        filtered_lines: List[str] = []
        for raw in comfy_req.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("-"):
                continue
            m = re.match(r"^([A-Za-z0-9_.-]+)", s)
            pkg = m.group(1).lower().replace("_", "-") if m else ""
            if pkg in excluded:
                continue
            filtered_lines.append(s)

        filtered_req_path = BASE_DIR / "comfy_requirements_no_torch.txt"
        filtered_req_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        print("[SETUP] Menginstal dependensi ComfyUI (mempertahankan PyTorch Colab)...", flush=True)
        res = run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade-strategy",
                "only-if-needed",
                "-r",
                str(filtered_req_path),
            ],
            check=False,
        )
        if res.returncode != 0:
            print("[WARN] Beberapa paket opsional ComfyUI dilewati, menginstal paket inti satu per satu...", flush=True)
            for line in filtered_lines:
                run_cmd(
                    [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", line],
                    check=False,
                )

    # Instal paket inti Studio & akselerator (tanpa gagal jika salah satu wheel opsional belum rilis di Python 3.13)
    core_studio_pkgs = [
        "gradio>=4.44.0",
        "websocket-client>=1.6.0",
        "requests>=2.31.0",
        "huggingface_hub>=0.24.0",
        "hf_xet",
        "gguf>=0.13.0",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.4.9",
        "einops>=0.7.0",
        "torchsde",
        "kornia",
        "spandrel",
        "safetensors>=0.4.2",
        "sentencepiece",
        "protobuf>=4.25.0",
        "accelerate",
        "pyyaml>=6.0.0",
        "psutil>=5.9.0",
    ]
    print("[SETUP] Menginstal dependensi AI Studio...", flush=True)
    res_studio = run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade-strategy",
            "only-if-needed",
            *core_studio_pkgs,
        ],
        check=False,
    )
    if res_studio.returncode != 0:
        for pkg in core_studio_pkgs:
            run_cmd(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", pkg],
                check=False,
            )

    print("[SETUP] Memasang comfy-kitchen (Fused INT8/SiLU kernel speedup)...", flush=True)
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-U", "comfy-kitchen"], check=False)


def download_selected_bundles(
    selected_engines: List[str],
    install_upscalers: bool = True,
    hf_token: str = "",
    civitai_token: str = "",
    modelscope_token: str = "",
) -> None:
    """Mengunduh seluruh file model dari engine dan upscaler yang dipilih tanpa duplikasi shared files."""
    from core.downloader import UniversalDownloader

    downloader = UniversalDownloader(
        models_root=MODELS_DIR,
        hf_token=hf_token or os.environ.get("HF_TOKEN", ""),
        civitai_token=civitai_token or os.environ.get("CIVITAI_TOKEN", ""),
        modelscope_token=modelscope_token or os.environ.get("MODELSCOPE_TOKEN", ""),
    )

    for eng_id in selected_engines:
        if eng_id not in ENGINE_CATALOG:
            print(f"[WARN] Engine '{eng_id}' tidak dikenal, dilewati.", flush=True)
            continue
        meta = ENGINE_CATALOG[eng_id]
        print(f"\n📦 Memeriksa & mengunduh paket [{meta['display_name']}]...", flush=True)
        downloader.download_engine(eng_id)

    if install_upscalers:
        print(
            "\n🔍 Memeriksa & mengunduh model Spandrel Tiled Upscaler default (4x-UltraSharp & 4x-AnimeSharp)...",
            flush=True,
        )
        for up_name in ("4x-UltraSharp.safetensors", "4x-AnimeSharp.safetensors"):
            spec = UPSCALER_CATALOG[up_name]
            downloader.download_spec(spec)


def print_storage_summary() -> None:
    """Menampilkan tabel ringkasan status instalasi 6 engine + 3 upscaler dan sisa ruang disk."""
    print("\n" + "=" * 78, flush=True)
    print("📊 RINGKASAN STATUS ENGINE & PENYIMPANAN DISK (/content)", flush=True)
    print("=" * 78, flush=True)
    for eng_id, meta in ENGINE_CATALOG.items():
        specs = meta["files"]  # type: ignore[index]
        installed_count = sum(1 for s in specs if check_file_installed(s, MODELS_DIR)[0])
        status_txt = "✅ SIAP" if installed_count == len(specs) else f"⚪ {installed_count}/{len(specs)} file"
        print(f"  • {str(meta['short_name']):24s} : {status_txt:12s} (~{meta['approx_size_gb']} GB)", flush=True)

    up_ready = [name for name, spec in UPSCALER_CATALOG.items() if check_file_installed(spec, MODELS_DIR)[0]]
    print(
        f"  • {'Tiled Upscalers':24s} : {len(up_ready)}/{len(UPSCALER_CATALOG)} siap ({', '.join(up_ready) or 'kosong'})",
        flush=True,
    )
    used_gb, free_gb, total_gb = get_disk_usage_gb(BASE_DIR)
    print("-" * 78, flush=True)
    print(f"💾 Disk Terpakai: {used_gb:.2f} GiB | Disk Kosong: {free_gb:.2f} GiB | Total: {total_gb:.2f} GiB", flush=True)
    print("=" * 78, flush=True)


def run_bootstrap(
    engines: str = "anima",
    upscalers: bool = True,
    skip_pip: bool = False,
    require_gpu: bool = False,
    hf_token: str = "",
    civitai_token: str = "",
    modelscope_token: str = "",
) -> None:
    """Fungsi API langsung untuk dipanggil dari Notebook Colab maupun CLI."""
    verify_cuda_gpu(require_gpu=require_gpu)
    install_system_packages()
    setup_comfyui_core(COMFY_DIR)
    setup_custom_nodes(COMFY_DIR)
    if not skip_pip:
        install_python_requirements(COMFY_DIR)

    engines_list = [
        x.strip().lower()
        for x in engines.split(",")
        if x.strip() and x.strip().lower() != "none"
    ]
    if engines_list or upscalers:
        download_selected_bundles(
            selected_engines=engines_list,
            install_upscalers=upscalers,
            hf_token=hf_token,
            civitai_token=civitai_token,
            modelscope_token=modelscope_token,
        )

    print_storage_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Unified Multi-Engine AI Studio (T4 Edition)")
    parser.add_argument(
        "--engines",
        type=str,
        default="anima",
        help="Daftar engine_id dipisahkan koma (anima,moss,qwen21,krea2,chroma,wan21) atau 'none'",
    )
    parser.add_argument("--upscalers", action="store_true", default=False, help="Unduh 4x-UltraSharp & 4x-AnimeSharp")
    parser.add_argument("--skip-pip", action="store_true", help="Lewati instalasi pip")
    parser.add_argument("--require-gpu", action="store_true", default=False, help="Gagal jika dijalankan tanpa GPU")
    parser.add_argument("--no-gpu-check", action="store_true", help="Jangan gagal jika dijalankan tanpa GPU")
    parser.add_argument("--hf-token", type=str, default="", help="HuggingFace Token")
    parser.add_argument("--civitai-token", type=str, default="", help="Civitai Token")
    parser.add_argument("--modelscope-token", type=str, default="", help="ModelScope Token")
    args = parser.parse_args()

    run_bootstrap(
        engines=args.engines,
        upscalers=args.upscalers,
        skip_pip=args.skip_pip,
        require_gpu=bool(args.require_gpu and not args.no_gpu_check),
        hf_token=args.hf_token,
        civitai_token=args.civitai_token,
        modelscope_token=args.modelscope_token,
    )


if __name__ == "__main__":
    main()