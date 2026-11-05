#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Entrypoint (`run_studio.py`) — Unified Multi-Engine AI Studio (T4 Edition)
Menyalakan Headless ComfyUI Backend dan meluncurkan Fooocus-Style Gradio Studio.
"""

import argparse
import inspect
import os
import secrets
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.catalog import BASE_DIR, COMFY_DIR, GALLERY_DIR, MODELS_DIR
from core.backend import HeadlessComfyManager
from core.downloader import UniversalDownloader
from core.gallery import GalleryManager
from core.memory_manager import configure_cuda_alloc_env
from ui.fooocus_app import build_fooocus_studio_app, get_gradio_allowed_paths, load_theme_css


def find_free_port(start_port: int = 7860, max_tries: int = 20) -> int:
    """Mencari port TCP lokal yang bebas untuk server Gradio."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Unified Multi-Engine AI Studio (T4 Edition)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host Gradio (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port awal Gradio (default: 7860)")
    parser.add_argument("--comfy-port", type=int, default=8188, help="Port Headless ComfyUI (default: 8188)")
    parser.add_argument("--share", action="store_true", default=True, help="Buat link publik gradio.live")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Matikan link publik gradio.live")
    parser.add_argument("--auth", action="store_true", default=False, help="Aktifkan login password acak Gradio")
    parser.add_argument("--username", type=str, default="studio", help="Username login Gradio jika --auth aktif")
    parser.add_argument("--password", type=str, default="", help="Password login Gradio (kosong = buat acak)")
    parser.add_argument(
        "--skip-backend-start",
        action="store_true",
        default=False,
        help="Jangan nyalakan ComfyUI saat startup (nyalakan otomatis saat tombol Generate ditekan)",
    )
    args = parser.parse_args()

    configure_cuda_alloc_env()

    allowed_paths = get_gradio_allowed_paths(
        base_dir=BASE_DIR,
        comfy_dir=COMFY_DIR,
        gallery_dir=GALLERY_DIR,
    )
    os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(allowed_paths)

    backend = HeadlessComfyManager(comfy_dir=COMFY_DIR, port=args.comfy_port)
    downloader = UniversalDownloader(models_root=MODELS_DIR)
    gallery = GalleryManager(gallery_dir=GALLERY_DIR)

    if not args.skip_backend_start and (COMFY_DIR / "main.py").exists():
        print("🚀 Menyalakan Headless ComfyUI Backend...", flush=True)
        backend.start()
    else:
        print("ℹ️ Headless ComfyUI akan dinyalakan otomatis saat proses render pertama.", flush=True)

    demo = build_fooocus_studio_app(
        backend=backend,
        downloader=downloader,
        gallery=gallery,
    )
    # Gunakan default_concurrency_limit=8 agar Tab Galeri & interaksi UI tetap instan saat GPU sibuk
    queued_demo = demo.queue(default_concurrency_limit=8)

    gradio_port = find_free_port(args.port)
    auth_tuple: Optional[Tuple[str, str]] = None
    if args.auth:
        pwd = args.password.strip() or secrets.token_urlsafe(10)
        auth_tuple = (args.username, pwd)
        print("=" * 68, flush=True)
        print(f"🔐 GRADIO AUTH AKTIF -> Username: {args.username} | Password: {pwd}", flush=True)
        print("=" * 68, flush=True)

    launch_kwargs: Dict[str, Any] = {
        "server_name": args.host,
        "server_port": gradio_port,
        "share": args.share,
        "auth": auth_tuple,
        "show_error": True,
        "inline": False,
    }
    try:
        sig = inspect.signature(queued_demo.launch)
        if "allowed_paths" in sig.parameters:
            launch_kwargs["allowed_paths"] = allowed_paths
        if "css" in sig.parameters:
            launch_kwargs["css"] = load_theme_css()
        if "theme" in sig.parameters:
            launch_kwargs["theme"] = gr.themes.Soft()
    except Exception:
        launch_kwargs["allowed_paths"] = allowed_paths

    print(f"🌐 Meluncurkan Fooocus-Style Multi-Engine Studio di port {gradio_port}...", flush=True)
    print(f"📂 Gradio Allowed Paths: {allowed_paths}", flush=True)
    queued_demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
