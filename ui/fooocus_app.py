# -*- coding: utf-8 -*-
"""
Gradio 4/5/6 Compatible UI dengan Layout Bergaya Fooocus (`lllyasviel/Fooocus`)
(`ui/fooocus_app.py`)
Fitur Utama:
1. Top Telemetry Bar (`gr.HTML`): Status Backend, GPU & Active VRAM Bar (GB / 15.0 GB),
   CPU System RAM Bar (GB / 12.7 GB), Badge Engine Aktif, serta Status WanGP + C++ Memory Guard.
2. Top-Level Workspace Tabs:
   - Tab 1 (`🎨 Studio Workspace`):
     * Center Main Viewport (`gr.Gallery` / `gr.Video`) + 3-Stage Stepper Bar
     * Bottom Prompt Bar (`🚀 Generate`, `🛑 Stop / Interrupt`, `🧹 Flush RAM/VRAM`) + Fooocus Checkboxes
     * Bottom Drawer `Input Image` (3 Sub-Tab: `Standalone Upscaler`, `Wan 2.1 Start Frame`, `Qwen 2.1 Multi-Image Edit`)
     * Right Sidebar Drawer `Advanced` (3 Tab: `⚙️ Setting`, `📦 Models & LoRA`, `🛠️ Sampler & Developer`)
   - Tab 2 (`🗂️ Galeri & History Inspector` — Non-Blocking & Fast Thumbnail Cache):
     * Terpisah sepenuhnya dari output event `Generate` yang sedang berjalan (`queue=False`, `show_progress="hidden"`)
       sehingga pengguna dapat membuka galeri, menginspeksi gambar resolusi penuh / video MP4, menyalin parameter,
       atau mengirim item ke Standalone Upscaler kapan saja bahkan saat GPU sedang merender!
     * Memakai Fast Thumbnail Cache (`*.thumb.jpg` ~20 KB) untuk grid galeri agar puluhan item termuat instan.
"""

import inspect
import os
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from config.catalog import (
    BASE_DIR,
    COMFY_DIR,
    ENGINE_CATALOG,
    GALLERY_DIR,
    MODELS_DIR,
    UPSCALER_CATALOG,
    check_engine_installed,
    check_file_installed,
    get_disk_usage_gb,
    get_other_engine_exclusive_files,
    list_local_models,
)
from core.backend import HeadlessComfyManager
from core.downloader import UniversalDownloader
from core.gallery import GalleryManager
from core.memory_manager import WANGP_PROFILES, trim_native_memory
from engines import (
    MAIN_ENGINE_IDS,
    QUICK_MOTION_TAGS,
    WAN21_FPS_CHOICES,
    BaseEngine,
    Wan21Engine,
    create_engine_registry,
    resolve_engine_id,
)

THEME_CSS_PATH = Path(__file__).resolve().parent / "theme.css"

# Daftar Lengkap 45 Sampler Resmi ComfyUI (comfy/samplers.py -> KSampler.SAMPLERS)
SAMPLER_CHOICES: List[str] = [
    "euler",
    "er_sde",
    "euler_ancestral",
    "uni_pc",
    "uni_pc_bh2",
    "heun",
    "heunpp2",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "res_multistep",
    "res_multistep_ancestral",
    "lcm",
    "ddim",
    "ddpm",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "ipndm",
    "ipndm_v",
    "deis",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "gradient_estimation",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "euler_cfg_pp",
    "euler_ancestral_cfg_pp",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_2m_cfg_pp",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation_cfg_pp",
    "cfgpp_ud10_ab",
]

# Daftar Lengkap 9 Scheduler Resmi ComfyUI (comfy/samplers.py -> KSampler.SCHEDULERS)
SCHEDULER_CHOICES: List[str] = [
    "simple",
    "sgm_uniform",
    "beta",
    "normal",
    "karras",
    "exponential",
    "ddim_uniform",
    "linear_quadratic",
    "kl_optimal",
]

PRESET_MODES: List[str] = [
    "⚡ Extreme Turbo (8-Step / 4-Step LoRA)",
    "🚀 Speed (Balanced + LazyCache)",
    "🎯 High Quality",
]


def get_gradio_allowed_paths(
    base_dir: Optional[Path] = None,
    comfy_dir: Optional[Path] = None,
    gallery_dir: Optional[Path] = None,
) -> List[str]:
    """
    Mengembalikan daftar direktori yang diizinkan untuk disajikan/dipindahkan oleh Gradio 4/5/6
    (mencegah `gradio.exceptions.InvalidPathError` saat menyajikan gambar/video dari `/content/Studio_Gallery`).
    """
    b_dir = Path(base_dir or BASE_DIR).resolve()
    c_dir = Path(comfy_dir or COMFY_DIR).resolve()
    g_dir = Path(gallery_dir or GALLERY_DIR).resolve()

    candidates = [
        str(b_dir),
        str(g_dir),
        str(c_dir),
        str(c_dir / "output"),
        str(c_dir / "input"),
        str(Path.cwd().resolve()),
        "/content",
        "/content/Studio_Gallery",
        "/content/ComfyUI",
        "/content/ai-studio-t4",
        "/tmp",
    ]
    deduped: List[str] = []
    for p in candidates:
        if p and p not in deduped:
            deduped.append(p)
    return deduped


# Daftarkan GRADIO_ALLOWED_PATHS sejak modul diimpor
os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(get_gradio_allowed_paths())


def load_theme_css() -> str:
    """Membaca isi file `ui/theme.css`."""
    if THEME_CSS_PATH.exists():
        return THEME_CSS_PATH.read_text(encoding="utf-8", errors="ignore")
    return ""


def _safe_component(cls: Any, **kwargs: Any) -> Any:
    """
    Helper lintas versi Gradio 4/5/6: memfilter argumen yang tidak didukung oleh
    `inspect.signature(cls.__init__)` agar aplikasi tidak pernah crash karena perbedaan versi.
    """
    try:
        sig = inspect.signature(cls.__init__)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        pass
    return cls(**kwargs)


def render_telemetry_bar_html(
    backend: HeadlessComfyManager,
    engine_id: str,
) -> str:
    """Merender Top Telemetry Bar HTML5 dengan indikator VRAM, System RAM, Engine, dan WanGP Profile."""
    eng_meta = ENGINE_CATALOG.get(engine_id, ENGINE_CATALOG["anima"])
    eng_short = str(eng_meta.get("short_name", engine_id))
    snap = backend.get_telemetry_snapshot(current_engine_label=eng_short)

    vram_pct = snap["vram_pct"]
    ram_pct = snap["ram_pct"]
    vram_color = "#06b6d4" if vram_pct < 75 else ("#f59e0b" if vram_pct < 90 else "#ef4444")
    ram_color = "#10b981" if ram_pct < 75 else ("#f59e0b" if ram_pct < 90 else "#ef4444")
    badge_cls = "badge-online" if snap["online"] else "badge-standby"
    _, free_disk_gb, _ = get_disk_usage_gb(BASE_DIR)

    return f"""
    <div class="fooocus-topbar">
        <div class="fooocus-topbar-row">
            <div class="fooocus-brand">
                <h1 class="fooocus-brand-title">⚡ UNIFIED MULTI-ENGINE AI STUDIO (T4 EDITION)</h1>
                <p class="fooocus-brand-sub">
                    Fooocus Interface • 8 Engines (incl. Flux-Fusion V2, Z-Image-Turbo &amp; Chroma Flash) • C++ Memory Guard • Disk Kosong: {free_disk_gb:.1f} GiB
                </p>
            </div>
            <div class="telemetry-pills-wrap">
                <div class="telemetry-pill">
                    <div class="telemetry-pill-head"><span>BACKEND</span><span>PORT 8188</span></div>
                    <div class="telemetry-pill-val {badge_cls}">● {snap['status_text']}</div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-head">
                        <span>GPU VRAM ({snap['gpu_name'][:12]})</span>
                        <span>{vram_pct}%</span>
                    </div>
                    <div class="telemetry-pill-val">{snap['vram_used_gb']:.2f} GB / {snap['vram_total_gb']:.1f} GB</div>
                    <div class="telemetry-bar-track">
                        <div class="telemetry-bar-fill" style="width:{vram_pct}%; background:{vram_color};"></div>
                    </div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-head">
                        <span>CPU SYSTEM RAM</span>
                        <span>{ram_pct}%</span>
                    </div>
                    <div class="telemetry-pill-val">{snap['ram_used_gb']:.2f} GB / {snap['ram_total_gb']:.1f} GB</div>
                    <div class="telemetry-bar-track">
                        <div class="telemetry-bar-fill" style="width:{ram_pct}%; background:{ram_color};"></div>
                    </div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-head"><span>ACTIVE ENGINE</span><span>MODE</span></div>
                    <div class="telemetry-pill-val" style="color:#a5b4fc;">{snap['active_engine']}</div>
                </div>
                <div class="telemetry-pill">
                    <div class="telemetry-pill-head"><span>WANGP + C++ GUARD</span><span>MMGP</span></div>
                    <div class="telemetry-pill-val" style="color:#38bdf8;">{snap['wangp_profile']}</div>
                </div>
            </div>
        </div>
    </div>
    """


def render_stepper_html(
    s1_state: str = "idle",
    s1_val: str = "Siap",
    s1_sub: str = "Text / Vision Conditioning",
    s2_state: str = "idle",
    s2_val: str = "Siap",
    s2_sub: str = "DiT / UNet Sampling",
    s3_state: str = "idle",
    s3_val: str = "Siap",
    s3_sub: str = "VAE Decode & Tiled Upscale",
    banner_msg: str = "✨ Studio siap. Ketik prompt di bawah lalu klik 🚀 Generate.",
) -> str:
    """Merender 3-Stage Stepper Bar di bawah kanvas utama."""
    return f"""
    <div class="stepper-container">
        <div class="stepper-banner">{banner_msg}</div>
        <div class="stepper-grid">
            <div class="stepper-card {s1_state}">
                <div class="stepper-stage-title">1. Conditioning &amp; Guard</div>
                <div class="stepper-stage-val">{s1_val}</div>
                <div class="stepper-stage-sub">{s1_sub}</div>
            </div>
            <div class="stepper-card {s2_state}">
                <div class="stepper-stage-title">2. Diffusion Sampling</div>
                <div class="stepper-stage-val">{s2_val}</div>
                <div class="stepper-stage-sub">{s2_sub}</div>
            </div>
            <div class="stepper-card {s3_state}">
                <div class="stepper-stage-title">3. Decode / Post-Process</div>
                <div class="stepper-stage-val">{s3_val}</div>
                <div class="stepper-stage-sub">{s3_sub}</div>
            </div>
        </div>
    </div>
    """


def format_on_demand_status_md(engine_id: str) -> str:
    """Menampilkan status ketersediaan file bobot engine yang sedang dipilih beserta sisa ruang disk."""
    meta = ENGINE_CATALOG.get(engine_id, ENGINE_CATALOG["anima"])
    all_ok, missing_specs = check_engine_installed(engine_id, MODELS_DIR)
    _, free_gb, total_gb = get_disk_usage_gb(BASE_DIR)
    removable = get_other_engine_exclusive_files(engine_id, MODELS_DIR)
    removable_gb = sum(p.stat().st_size for p in removable if p.is_file()) / (1024**3)

    if all_ok:
        badge = "✅ **TERPASANG LENGKAP DI DISK**"
    else:
        missing_names = ", ".join(f"`{s.filename}`" for s in missing_specs)
        badge = f"⚠️ **BELUM LENGKAP** (Kurang {len(missing_specs)} file: {missing_names})"

    return (
        f"#### 📦 Status Engine: **{meta['short_name']}** (~{meta['approx_size_gb']} GB)\n"
        f"- **Status Bobot:** {badge}\n"
        f"- **Sisa Disk `/content`:** `{free_gb:.2f} GiB` kosong dari `{total_gb:.1f} GiB`\n"
        f"- **Bobot Engine Lain yang Bisa Dihapus:** `{len(removable)} file` (`~{removable_gb:.2f} GiB` dapat dibebaskan)"
    )


def get_file_dropdown_choices(engine_id: str) -> Tuple[List[str], str, List[str], str, List[str], str, List[str]]:
    """Mengembalikan pilihan dropdown (Diffusion, Text Encoder, VAE, LoRAs) untuk engine aktif."""
    defaults = ENGINE_CATALOG.get(engine_id, ENGINE_CATALOG["anima"])["defaults"]  # type: ignore[index]
    def_unet = str(defaults.get("unet", ""))
    def_te = str(defaults.get("text_encoder", ""))
    def_vae = str(defaults.get("vae", ""))

    disk_unets = list_local_models("diffusion", MODELS_DIR)
    disk_tes = list_local_models("text_encoder", MODELS_DIR)
    disk_vaes = list_local_models("vae", MODELS_DIR)
    disk_loras = ["(None)"] + list_local_models("loras", MODELS_DIR)

    if def_unet and def_unet not in disk_unets:
        disk_unets = [def_unet] + disk_unets
    if def_te and def_te not in disk_tes:
        disk_tes = [def_te] + disk_tes
    if def_vae and def_vae not in disk_vaes:
        disk_vaes = [def_vae] + disk_vaes

    return (
        disk_unets or ["(Tidak ada file)"],
        def_unet or (disk_unets[0] if disk_unets else "(Tidak ada file)"),
        disk_tes or ["(Tidak ada file)"],
        def_te or (disk_tes[0] if disk_tes else "(Tidak ada file)"),
        disk_vaes or ["(Tidak ada file)"],
        def_vae or (disk_vaes[0] if disk_vaes else "(Tidak ada file)"),
        disk_loras,
    )


def build_fooocus_studio_app(
    backend: Optional[HeadlessComfyManager] = None,
    downloader: Optional[UniversalDownloader] = None,
    gallery: Optional[GalleryManager] = None,
) -> gr.Blocks:
    """Membangun seluruh aplikasi Gradio bergaya Fooocus untuk ke-7 Engine + Spandrel Tiled Upscaler."""
    backend_mgr = backend or HeadlessComfyManager(comfy_dir=COMFY_DIR)
    dl_mgr = downloader or UniversalDownloader(models_root=MODELS_DIR)
    gal_mgr = gallery or GalleryManager(gallery_dir=GALLERY_DIR)
    engines = create_engine_registry(comfy_dir=COMFY_DIR)

    allowed_paths = get_gradio_allowed_paths(
        base_dir=BASE_DIR,
        comfy_dir=backend_mgr.comfy_dir,
        gallery_dir=gal_mgr.gallery_dir,
    )
    os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(allowed_paths)

    # Pilih engine awal: utamakan engine yang sudah terinstal di disk, default ke 'anima'
    initial_engine_id = "anima"
    for eid in MAIN_ENGINE_IDS:
        ok, _ = check_engine_installed(eid, MODELS_DIR)
        if ok:
            initial_engine_id = eid
            break

    init_eng = engines[initial_engine_id]
    init_def = init_eng.default_params
    init_preset_val = str(init_def.get("default_preset", PRESET_MODES[0]))
    init_unets, init_u, init_tes, init_t, init_vaes, init_v, init_loras = get_file_dropdown_choices(
        initial_engine_id
    )
    init_aspects = list(init_def["aspect_ratios"].keys()) + ["Custom"]
    init_aspect_val = init_def["default_aspect"]
    init_w, init_h = init_def["aspect_ratios"][init_aspect_val]

    engine_radio_choices = [
        (str(ENGINE_CATALOG[eid]["display_name"]), eid) for eid in MAIN_ENGINE_IDS
    ]
    wangp_profile_choices = [
        (prof.display_name, pid) for pid, prof in WANGP_PROFILES.items()
    ]

    theme_css = load_theme_css()
    blocks_kwargs: Dict[str, Any] = {
        "title": "Unified Multi-Engine AI Studio (T4 Edition)",
        "analytics_enabled": False,
    }
    try:
        gr_major = int(str(getattr(gr, "__version__", "5.0.0")).split(".")[0])
    except Exception:
        gr_major = 5

    if gr_major < 6:
        blocks_kwargs["css"] = theme_css
        try:
            blocks_kwargs["theme"] = gr.themes.Soft()
        except Exception:
            pass

    with _safe_component(gr.Blocks, **blocks_kwargs) as demo:
        demo._studio_allowed_paths = allowed_paths
        if gr_major >= 6 and theme_css:
            gr.HTML(f"<style>{theme_css}</style>", visible=False)

        # State Internal Sesi
        last_rendered_image_state = gr.State("")
        gallery_media_paths_state = gr.State([])
        selected_history_media_state = gr.State("")

        # ======================================================================
        # 1. TOP TELEMETRY BAR
        # ======================================================================
        telemetry_html = gr.HTML(
            value=render_telemetry_bar_html(backend_mgr, initial_engine_id)
        )

        # ======================================================================
        # 2. TOP-LEVEL WORKSPACE TABS (STUDIO WORKSPACE vs DEDICATED GALLERY TAB)
        # ======================================================================
        with gr.Tabs():
            # ==================================================================
            # TAB UTAMA 1 — 🎨 STUDIO WORKSPACE
            # ==================================================================
            with gr.Tab("🎨 Studio Workspace"):
                with gr.Row(equal_height=False):
                    # ---- CENTER MAIN CANVAS & BOTTOM PROMPT / INPUT DRAWER ----
                    with gr.Column(scale=7):
                        main_image_gallery = _safe_component(
                            gr.Gallery,
                            label="🎨 Studio Viewport (Image & Tiled Upscaler)",
                            columns=2,
                            height=510,
                            object_fit="contain",
                            preview=True,
                            visible=(not init_eng.is_video),
                            elem_classes=["fooocus-canvas-box"],
                        )
                        main_video_player = _safe_component(
                            gr.Video,
                            label="🎬 Wan 2.1 I2V Cinema Viewport (MP4)",
                            height=510,
                            autoplay=True,
                            visible=init_eng.is_video,
                            elem_classes=["fooocus-canvas-box"],
                        )

                        stepper_bar_html = gr.HTML(value=render_stepper_html())

                        # ======================================================
                        # 3. BOTTOM PROMPT BAR (FOOOCUS STYLE)
                        # ======================================================
                        with gr.Row(elem_classes=["fooocus-prompt-row"]):
                            prompt_box = _safe_component(
                                gr.Textbox,
                                label="Prompt (Teks Deskriptif, Tag Anime, Bilingual Z-Image, <image1> Edit Qwen 2.1, atau Kamera Wan 2.1)",
                                placeholder=init_def["placeholder"],
                                lines=3,
                                scale=5,
                            )
                            with gr.Column(scale=2, min_width=190):
                                generate_btn = _safe_component(
                                    gr.Button,
                                    value="🚀 Generate",
                                    variant="primary",
                                    elem_classes=["fooocus-generate-btn"],
                                )
                                with gr.Row():
                                    stop_btn = _safe_component(
                                        gr.Button,
                                        value="🛑 Stop / Interrupt",
                                        variant="stop",
                                        size="sm",
                                    )
                                    flush_btn = _safe_component(
                                        gr.Button,
                                        value="🧹 Flush RAM/VRAM",
                                        variant="secondary",
                                        size="sm",
                                    )

                        # Baris Checkbox Bawah (Persis Fooocus)
                        with gr.Row():
                            chk_input_image = _safe_component(
                                gr.Checkbox,
                                label="🖼️ Input Image (Upscale / Image Edit / Wan 2.1 I2V)",
                                value=False,
                            )
                            chk_auto_upscale = _safe_component(
                                gr.Checkbox,
                                label="⚡ Auto-Upscale Hasil Render (Tiled 2x)",
                                value=False,
                            )
                            chk_advanced = _safe_component(
                                gr.Checkbox,
                                label="⚙️ Advanced (Buka Panel Pengaturan & Model)",
                                value=True,
                            )

                        # ======================================================
                        # 4. BOTTOM DRAWER: INPUT IMAGE (3 SUB-TABS)
                        # ======================================================
                        with _safe_component(
                            gr.Column,
                            visible=False,
                            elem_classes=["fooocus-drawer-box"],
                        ) as input_image_drawer:
                            gr.Markdown("### 🖼️ Input Image & Multimodal Workspace")
                            with gr.Tabs():
                                # Sub-Tab 1: Standalone Upscaler
                                with gr.Tab("🔍 Standalone Upscaler (Cepat & Hemat VRAM)"):
                                    with gr.Row():
                                        with gr.Column(scale=4):
                                            upscale_input_img = _safe_component(
                                                gr.Image,
                                                label="Upload Gambar untuk Di-Upscale",
                                                type="filepath",
                                                height=240,
                                            )
                                            send_last_to_upscaler_btn = _safe_component(
                                                gr.Button,
                                                value="📥 Kirim Hasil Terakhir ke Upscaler",
                                                variant="secondary",
                                                size="sm",
                                            )
                                        with gr.Column(scale=4):
                                            upscaler_model_dropdown = _safe_component(
                                                gr.Dropdown,
                                                label="Pilih Model Spandrel Tiled Upscaler (<350 MB VRAM)",
                                                choices=list(UPSCALER_CATALOG.keys()),
                                                value="4x-UltraSharp.safetensors",
                                            )
                                            upscale_scale_radio = _safe_component(
                                                gr.Radio,
                                                label="Faktor Skala Akhir (Tiled 4x + Lanczos)",
                                                choices=["1.5x", "2.0x", "4.0x"],
                                                value="2.0x",
                                            )
                                            run_standalone_upscale_btn = _safe_component(
                                                gr.Button,
                                                value="✨ Upscale Sekarang (2–4 Detik)",
                                                variant="primary",
                                            )

                                # Sub-Tab 2: Wan 2.1 Start Frame (Image-to-Video)
                                with gr.Tab("🎬 Wan 2.1 Start Frame (Image-to-Video)"):
                                    with gr.Row():
                                        with gr.Column(scale=4):
                                            wan_start_img = _safe_component(
                                                gr.Image,
                                                label="Upload Gambar Frame Awal (Start Frame Wan 2.1)",
                                                type="pil",
                                                height=240,
                                            )
                                        with gr.Column(scale=5):
                                            wan_auto_crop_chk = _safe_component(
                                                gr.Checkbox,
                                                label="🔒 Auto-Aspect Center-Crop 480P (Anti-Gepeng Lanczos)",
                                                value=True,
                                            )
                                            wan_ratio_info_html = gr.HTML(
                                                "<div style='font-size:0.82rem; color:#94a3b8;'>"
                                                "Unggah gambar frame awal untuk menghitung ukuran 480P optimal bebas distorsi."
                                                "</div>"
                                            )
                                            gr.Markdown("**🎥 Quick Motion Tags (Klik untuk menambahkan ke Prompt):**")
                                            with gr.Row():
                                                btn_motion_push = _safe_component(gr.Button, value="+ Slow Push-In", size="sm")
                                                btn_motion_orbit = _safe_component(gr.Button, value="+ Orbit Shot", size="sm")
                                                btn_motion_pan = _safe_component(gr.Button, value="+ Smooth Pan", size="sm")
                                            with gr.Row():
                                                btn_motion_wind = _safe_component(gr.Button, value="+ Wind & Hair", size="sm")
                                                btn_motion_hand = _safe_component(gr.Button, value="+ Handheld Realism", size="sm")

                                # Sub-Tab 3: Qwen 2.1 Multi-Image Edit
                                with gr.Tab("🎨 Qwen 2.1 Multi-Image Edit (<image1>, <image2>, <image3>)"):
                                    gr.Markdown(
                                        "Unggah 1–3 gambar referensi lalu sebutkan **`<image1>`**, **`<image2>`**, "
                                        "atau **`<image3>`** di dalam prompt utama Anda."
                                    )
                                    with gr.Row():
                                        qwen_ref1 = _safe_component(
                                            gr.Image,
                                            label="Referensi <image1>",
                                            type="filepath",
                                            height=175,
                                        )
                                        qwen_ref2 = _safe_component(
                                            gr.Image,
                                            label="Referensi <image2>",
                                            type="filepath",
                                            height=175,
                                        )
                                        qwen_ref3 = _safe_component(
                                            gr.Image,
                                            label="Referensi <image3>",
                                            type="filepath",
                                            height=175,
                                        )
                                    with gr.Row():
                                        qwen_lock_ratio_chk = _safe_component(
                                            gr.Checkbox,
                                            label="Ikuti Rasio <image1> Otomatis",
                                            value=True,
                                        )
                                        qwen_ref_res_slider = _safe_component(
                                            gr.Slider,
                                            label="Resolusi Fitur Referensi (Default: 992 — Bebas Bug Noise #16435)",
                                            minimum=0,
                                            maximum=1536,
                                            step=32,
                                            value=992,
                                        )

                    # ==========================================================
                    # 5. RIGHT SIDEBAR DRAWER: ADVANCED (3 TABS)
                    # ==========================================================
                    with _safe_component(
                        gr.Column,
                        scale=5,
                        visible=True,
                        elem_classes=["fooocus-drawer-box"],
                    ) as advanced_drawer:
                        with gr.Tabs():
                            # --------------------------------------------------
                            # TAB 1 — SETTING
                            # --------------------------------------------------
                            with gr.Tab("⚙️ Setting"):
                                engine_selector = _safe_component(
                                    gr.Dropdown,
                                    label="🧠 Pilih Engine Utama (8 Arsitektur T4)",
                                    choices=engine_radio_choices,
                                    value=initial_engine_id,
                                )
                                preset_mode_radio = _safe_component(
                                    gr.Radio,
                                    label="⚡ Preset Mode (Otomatis Sinkron Steps, CFG, Sampler & Arsitektur)",
                                    choices=PRESET_MODES,
                                    value=init_preset_val,
                                )
                                aspect_dropdown = _safe_component(
                                    gr.Dropdown,
                                    label="📐 Aspect Ratio / Resolusi",
                                    choices=init_aspects,
                                    value=init_aspect_val,
                                )
                                with gr.Row():
                                    width_slider = _safe_component(
                                        gr.Slider,
                                        label="Width (px)",
                                        minimum=320,
                                        maximum=1536,
                                        step=16,
                                        value=init_w,
                                    )
                                    height_slider = _safe_component(
                                        gr.Slider,
                                        label="Height (px)",
                                        minimum=320,
                                        maximum=1536,
                                        step=16,
                                        value=init_h,
                                    )

                                # Kontrol Khusus Video (Hanya tampil saat Wan 2.1 aktif)
                                with _safe_component(
                                    gr.Group,
                                    visible=init_eng.is_video,
                                ) as video_specific_group:
                                    wan_frames_slider = _safe_component(
                                        gr.Slider,
                                        label="🎬 Base Frames ((4k)+1): 17..81 (Default 49 = 3.1 Detik)",
                                        minimum=17,
                                        maximum=81,
                                        step=4,
                                        value=49,
                                    )
                                    wan_fps_dropdown = _safe_component(
                                        gr.Dropdown,
                                        label="🎞️ Output FPS (True-Duration Motion Interpolation)",
                                        choices=WAN21_FPS_CHOICES,
                                        value=WAN21_FPS_CHOICES[0],
                                    )

                                # Kontrol Khusus Image (Sembunyi saat Wan 2.1 aktif)
                                with _safe_component(
                                    gr.Group,
                                    visible=(not init_eng.is_video),
                                ) as image_specific_group:
                                    batch_count_slider = _safe_component(
                                        gr.Slider,
                                        label="🖼️ Batch Count (Jumlah Gambar: 1..4)",
                                        minimum=1,
                                        maximum=4,
                                        step=1,
                                        value=1,
                                    )

                                negative_prompt_box = _safe_component(
                                    gr.Textbox,
                                    label="🚫 Negative Prompt (Otomatis Disesuaikan per Engine · Aktif saat CFG > 1.0)",
                                    lines=2,
                                    value=str(
                                        init_def.get(
                                            "negative_prompt",
                                            "worst quality, low quality, blurry, deformed, bad anatomy, watermark, jpeg artifacts",
                                        )
                                    ),
                                )
                                seed_number = _safe_component(
                                    gr.Number,
                                    label="🎲 Seed (-1 = Acak Setiap Render)",
                                    value=-1,
                                    precision=0,
                                )

                            # --------------------------------------------------
                            # TAB 2 — MODELS & LORA
                            # --------------------------------------------------
                            with gr.Tab("📦 Models & LoRA"):
                                on_demand_status_md = gr.Markdown(
                                    value=format_on_demand_status_md(initial_engine_id)
                                )
                                with gr.Row():
                                    btn_download_engine = _safe_component(
                                        gr.Button,
                                        value="⬇️ Download Engine Ini Sekarang",
                                        variant="primary",
                                        size="sm",
                                    )
                                    btn_clean_other_engines = _safe_component(
                                        gr.Button,
                                        value="🗑️ Hapus Bobot Engine Lain untuk Hemat Disk",
                                        variant="stop",
                                        size="sm",
                                    )

                                with gr.Accordion("📂 File Overrides (Diffusion / Text Encoder / VAE)", open=True):
                                    unet_override_dd = _safe_component(
                                        gr.Dropdown,
                                        label="Diffusion / UNet (.safetensors / .gguf)",
                                        choices=init_unets,
                                        value=init_u,
                                    )
                                    clip_override_dd = _safe_component(
                                        gr.Dropdown,
                                        label="Text Encoder (CLIP / Qwen / T5 / UMT5)",
                                        choices=init_tes,
                                        value=init_t,
                                    )
                                    with gr.Row():
                                        vae_override_dd = _safe_component(
                                            gr.Dropdown,
                                            label="VAE Model",
                                            choices=init_vaes,
                                            value=init_v,
                                            scale=3,
                                        )
                                        btn_refresh_disk = _safe_component(
                                            gr.Button,
                                            value="🔄 Refresh Disk",
                                            variant="secondary",
                                            scale=1,
                                        )

                                with gr.Accordion("🧩 3-Slot Custom LoRA Mixer", open=True):
                                    with gr.Row():
                                        lora1_dd = _safe_component(
                                            gr.Dropdown,
                                            label="LoRA Slot 1",
                                            choices=init_loras,
                                            value="(None)",
                                            scale=3,
                                        )
                                        lora1_str = _safe_component(
                                            gr.Slider,
                                            label="Strength 1",
                                            minimum=-2.0,
                                            maximum=2.0,
                                            step=0.05,
                                            value=0.8,
                                            scale=2,
                                        )
                                    with gr.Row():
                                        lora2_dd = _safe_component(
                                            gr.Dropdown,
                                            label="LoRA Slot 2",
                                            choices=init_loras,
                                            value="(None)",
                                            scale=3,
                                        )
                                        lora2_str = _safe_component(
                                            gr.Slider,
                                            label="Strength 2",
                                            minimum=-2.0,
                                            maximum=2.0,
                                            step=0.05,
                                            value=0.6,
                                            scale=2,
                                        )
                                    with gr.Row():
                                        lora3_dd = _safe_component(
                                            gr.Dropdown,
                                            label="LoRA Slot 3",
                                            choices=init_loras,
                                            value="(None)",
                                            scale=3,
                                        )
                                        lora3_str = _safe_component(
                                            gr.Slider,
                                            label="Strength 3",
                                            minimum=-2.0,
                                            maximum=2.0,
                                            step=0.05,
                                            value=0.5,
                                            scale=2,
                                        )

                                    with _safe_component(
                                        gr.Group,
                                        visible=init_eng.is_video,
                                    ) as wan_lora_Controls_group:
                                        with gr.Row():
                                            wan_lightx2v_slider = _safe_component(
                                                gr.Slider,
                                                label="⚡ Wan 2.1 Lightx2v 4-Step Strength",
                                                minimum=0.0,
                                                maximum=1.5,
                                                step=0.05,
                                                value=1.0,
                                                scale=3,
                                            )
                                            btn_wan_autotune = _safe_component(
                                                gr.Button,
                                                value="⚖️ Auto-Tune Custom LoRA",
                                                variant="secondary",
                                                scale=2,
                                            )

                                with gr.Accordion("🌐 In-App Universal Downloader (HF · Civitai .com/.red · ModelScope)", open=False):
                                    custom_dl_url_box = _safe_component(
                                        gr.Textbox,
                                        label="Paste Link Model/LoRA (Mendukung |nama_kustom.safetensors)",
                                        placeholder="https://civitai.com/api/download/models/... atau https://huggingface.co/...|my_lora.safetensors",
                                        lines=2,
                                    )
                                    with gr.Row():
                                        custom_dl_folder_dd = _safe_component(
                                            gr.Dropdown,
                                            label="Target Folder",
                                            choices=["loras", "diffusion_models", "text_encoders", "vae", "upscale_models"],
                                            value="loras",
                                            scale=2,
                                        )
                                        btn_custom_dl = _safe_component(
                                            gr.Button,
                                            value="⬇️ Unduh via Aria2c",
                                            variant="primary",
                                            scale=2,
                                        )
                                    with gr.Row():
                                        local_lora_file = _safe_component(
                                            gr.File,
                                            label="Atau Upload File .safetensors Lokal",
                                            file_count="single",
                                            type="filepath",
                                            file_types=[".safetensors"],
                                        )
                                        btn_upload_local_lora = _safe_component(
                                            gr.Button,
                                            value="⬆️ Pasang ke LoRA Slot 1",
                                            variant="secondary",
                                        )

                            # --------------------------------------------------
                            # TAB 3 — SAMPLER & DEVELOPER
                            # --------------------------------------------------
                            with gr.Tab("🛠️ Sampler & Developer"):
                                wangp_profile_dd = _safe_component(
                                    gr.Dropdown,
                                    label="🧠 WanGP Memory Profile (mmgp + C++ Native Allocator)",
                                    choices=wangp_profile_choices,
                                    value=backend_mgr.memory_manager.active_profile_id,
                                )
                                with gr.Row():
                                    steps_slider = _safe_component(
                                        gr.Slider,
                                        label="Steps",
                                        minimum=1,
                                        maximum=60,
                                        step=1,
                                        value=init_def["steps"],
                                    )
                                    cfg_slider = _safe_component(
                                        gr.Slider,
                                        label="CFG Scale",
                                        minimum=1.0,
                                        maximum=10.0,
                                        step=0.1,
                                        value=init_def["cfg"],
                                    )
                                    shift_slider = _safe_component(
                                        gr.Slider,
                                        label="Flow Shift / Flux Guidance",
                                        minimum=0.5,
                                        maximum=10.0,
                                        step=0.1,
                                        value=init_def["shift"],
                                        visible=(initial_engine_id in ("z_image", "flux_fusion", "moss", "chroma", "wan21")),
                                    )
                                with gr.Row():
                                    sampler_dd = _safe_component(
                                        gr.Dropdown,
                                        label="Sampler (Seluruh 45 Sampler Resmi ComfyUI)",
                                        choices=SAMPLER_CHOICES,
                                        value=init_def["sampler"],
                                    )
                                    scheduler_dd = _safe_component(
                                        gr.Dropdown,
                                        label="Scheduler (Seluruh 9 Scheduler Resmi ComfyUI)",
                                        choices=SCHEDULER_CHOICES,
                                        value=init_def["scheduler"],
                                    )

                                with gr.Accordion("🔬 Pengaturan Spesifik Arsitektur (LazyCache · MOSS SLG · Chroma Beta)", open=True):
                                    with _safe_component(
                                        gr.Row,
                                        visible=(initial_engine_id in ("anima", "z_image", "flux_fusion", "moss", "qwen21", "krea2", "chroma")),
                                    ) as lazycache_arch_row:
                                        use_lazycache_chk = _safe_component(
                                            gr.Checkbox,
                                            label="Aktifkan LazyCache (Steps ≥ 8)",
                                            value=bool(init_def.get("use_lazycache", False)),
                                        )
                                        lazycache_thresh_slider = _safe_component(
                                            gr.Slider,
                                            label="LazyCache Threshold",
                                            minimum=0.05,
                                            maximum=0.50,
                                            step=0.05,
                                            value=0.20,
                                        )
                                    with _safe_component(
                                        gr.Row,
                                        visible=(initial_engine_id == "moss"),
                                    ) as moss_arch_row:
                                        moss_slg_chk = _safe_component(
                                            gr.Checkbox,
                                            label="MOSS Skip-Layer Guidance (CFG > 1.0)",
                                            value=bool(init_def.get("use_slg", False)),
                                        )
                                        moss_skip_layer_num = _safe_component(
                                            gr.Number,
                                            label="MOSS Skip Layer Index",
                                            value=int(init_def.get("skip_layer", 22)),
                                            precision=0,
                                        )
                                    with _safe_component(
                                        gr.Row,
                                        visible=(initial_engine_id == "chroma"),
                                    ) as chroma_arch_row:
                                        chroma_alpha_slider = _safe_component(
                                            gr.Slider,
                                            label="Chroma Beta Alpha",
                                            minimum=0.1,
                                            maximum=1.0,
                                            step=0.05,
                                            value=float(init_def.get("beta_alpha", 0.45)),
                                        )
                                        chroma_beta_slider = _safe_component(
                                            gr.Slider,
                                            label="Chroma Beta Beta",
                                            minimum=0.1,
                                            maximum=1.0,
                                            step=0.05,
                                            value=float(init_def.get("beta_beta", 0.45)),
                                        )
                                        chroma_min_pad_slider = _safe_component(
                                            gr.Slider,
                                            label="T5 Min Padding",
                                            minimum=0,
                                            maximum=8,
                                            step=1,
                                            value=int(init_def.get("min_padding", 1)),
                                        )

                                with gr.Accordion("🖥️ Maintenance & Debug Box (ComfyUI Daemon)", open=False):
                                    daemon_log_box = _safe_component(
                                        gr.Code,
                                        label="ComfyUI Daemon Log",
                                        value="Klik 'Refresh Log' untuk menampilkan log terbaru Headless ComfyUI.",
                                        language="shell",
                                        lines=10,
                                    )
                                    with gr.Row():
                                        btn_refresh_log = _safe_component(
                                            gr.Button,
                                            value="🔄 Refresh Log",
                                            variant="secondary",
                                            size="sm",
                                        )
                                        btn_restart_daemon = _safe_component(
                                            gr.Button,
                                            value="♻️ Restart Headless ComfyUI Server",
                                            variant="stop",
                                            size="sm",
                                        )

            # ==================================================================
            # TAB UTAMA 2 — 🗂️ GALERI & HISTORY INSPECTOR (DEDICATED & NON-BLOCKING)
            # ==================================================================
            with gr.Tab("🗂️ Galeri & History Inspector"):
                with _safe_component(gr.Column, elem_classes=["fooocus-drawer-box"]):
                    with gr.Row():
                        with gr.Column(scale=7):
                            history_summary_md = gr.Markdown("🗂️ Memuat galeri sesi...")
                        with gr.Column(scale=5):
                            with gr.Row():
                                btn_refresh_history = _safe_component(
                                    gr.Button,
                                    value="🔄 Refresh Galeri (Instan)",
                                    variant="secondary",
                                    size="sm",
                                )
                                btn_export_zip = _safe_component(
                                    gr.Button,
                                    value="📦 Download Seluruh Galeri (ZIP)",
                                    variant="primary",
                                    size="sm",
                                )

                    with gr.Row(equal_height=False):
                        # Kolom Kiri: Grid Galeri Cepat berbasis Thumbnail JPEG (~20 KB)
                        with gr.Column(scale=7):
                            history_gallery = _safe_component(
                                gr.Gallery,
                                label="🖼️ Grid Galeri Sesi (Fast Thumbnail Cache — Klik Item untuk Melihat Resolusi Penuh & Metadata)",
                                columns=4,
                                height=540,
                                object_fit="cover",
                                preview=False,
                            )
                            zip_download_file = _safe_component(
                                gr.File,
                                label="📦 Unduh Arsip ZIP Seluruh Galeri (PNG + MP4 + JSON)",
                            )

                        # Kolom Kanan: Full-Resolution Preview + Action Buttons + Metadata Inspector
                        with gr.Column(scale=5):
                            inspector_image_preview = _safe_component(
                                gr.Image,
                                label="🖼️ Pratinjau Resolusi Penuh (PNG Asli — Klik Ikon Pojok Kanan Atas untuk Unduh / Layar Penuh)",
                                type="filepath",
                                height=320,
                                visible=True,
                            )
                            inspector_video_preview = _safe_component(
                                gr.Video,
                                label="🎬 Video Player (MP4 Asli)",
                                height=320,
                                visible=False,
                            )
                            with gr.Row():
                                btn_copy_params = _safe_component(
                                    gr.Button,
                                    value="♻️ Salin Parameter ke Studio",
                                    variant="primary",
                                    size="sm",
                                )
                                btn_send_selected_to_upscaler = _safe_component(
                                    gr.Button,
                                    value="🔍 Kirim ke Standalone Upscaler",
                                    variant="secondary",
                                    size="sm",
                                )
                                btn_delete_history_item = _safe_component(
                                    gr.Button,
                                    value="🗑️ Hapus Item",
                                    variant="stop",
                                    size="sm",
                                )
                            inspector_meta_md = gr.Markdown(
                                "Klik salah satu gambar atau video di sebelah kiri untuk melihat pratinjau resolusi penuh dan tabel metadata lengkap."
                            )

        # ======================================================================
        # 6. CALLBACKS & EVENT HANDLERS
        # ======================================================================
        def _get_merged_samplers_and_schedulers() -> Tuple[List[str], List[str]]:
            """Menggabungkan 45 Sampler & 9 Scheduler resmi dengan daftar dinamis dari server ComfyUI jika aktif."""
            samplers = list(SAMPLER_CHOICES)
            schedulers = list(SCHEDULER_CHOICES)
            dyn_s, dyn_sch = backend_mgr.get_samplers_and_schedulers()
            for s in dyn_s:
                if s not in samplers:
                    samplers.append(s)
            for sch in dyn_sch:
                if sch not in schedulers:
                    schedulers.append(sch)
            return samplers, schedulers

        def on_engine_change(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            eng = engines[eng_id]
            d = eng.default_params
            unets, u_val, tes, t_val, vaes, v_val, loras = get_file_dropdown_choices(eng_id)
            aspect_keys = list(d["aspect_ratios"].keys()) + ["Custom"]
            def_aspect = d["default_aspect"]
            w_val, h_val = d["aspect_ratios"][def_aspect]

            def_preset = str(d.get("default_preset", PRESET_MODES[0]))
            p_cfg = d.get("presets", {}).get(def_preset, {})

            samplers, schedulers = _get_merged_samplers_and_schedulers()
            target_sampler = str(p_cfg.get("sampler", d["sampler"]))
            target_scheduler = str(p_cfg.get("scheduler", d["scheduler"]))
            if target_sampler not in samplers:
                samplers.append(target_sampler)
            if target_scheduler not in schedulers:
                schedulers.append(target_scheduler)

            # Sesuaikan profil memori WanGP secara otomatis
            if eng_id == "wan21":
                backend_mgr.memory_manager.set_profile("profile_2_lowram")
            else:
                backend_mgr.memory_manager.set_profile("profile_1_speed")

            show_shift = eng_id in ("z_image", "flux_fusion", "moss", "chroma", "wan21")
            show_lazycache = eng_id in ("anima", "z_image", "flux_fusion", "moss", "qwen21", "krea2", "chroma")
            show_moss = eng_id == "moss"
            show_chroma = eng_id == "chroma"

            return (
                # 1-2. Tampilan kanvas tengah Image vs Video
                gr.update(visible=not eng.is_video),
                gr.update(visible=eng.is_video),
                # 3-5. Placeholder prompt, Negative prompt default, & Preset Mode default
                gr.update(placeholder=d["placeholder"]),
                gr.update(value=str(d.get("negative_prompt", ""))),
                gr.update(value=def_preset),
                # 6-8. Aspect ratio & ukuran
                gr.update(choices=aspect_keys, value=def_aspect),
                gr.update(value=w_val),
                gr.update(value=h_val),
                # 9-11. Grup kontrol Video vs Image
                gr.update(visible=eng.is_video),
                gr.update(visible=not eng.is_video),
                gr.update(visible=eng.is_video),
                # 12-16. Parameter sampling utama (Steps, CFG, Shift + visibilitas, Sampler, Scheduler)
                gr.update(value=int(p_cfg.get("steps", d["steps"]))),
                gr.update(value=float(p_cfg.get("cfg", d["cfg"]))),
                gr.update(value=float(p_cfg.get("shift", d["shift"])), visible=show_shift),
                gr.update(choices=samplers, value=target_sampler),
                gr.update(choices=schedulers, value=target_scheduler),
                # 17-23. Nilai parameter spesifik arsitektur
                gr.update(value=bool(p_cfg.get("use_lazycache", d.get("use_lazycache", False)))),
                gr.update(value=bool(p_cfg.get("use_slg", d.get("use_slg", False)))),
                gr.update(value=int(p_cfg.get("skip_layer", d.get("skip_layer", 22)))),
                gr.update(value=float(p_cfg.get("beta_alpha", d.get("beta_alpha", 0.45)))),
                gr.update(value=float(p_cfg.get("beta_beta", d.get("beta_beta", 0.45)))),
                gr.update(value=int(p_cfg.get("min_padding", d.get("min_padding", 1)))),
                gr.update(value=float(p_cfg.get("lightx2v_strength", d.get("lightx2v_strength", 1.0)))),
                # 24-26. Visibilitas baris spesifik arsitektur
                gr.update(visible=show_lazycache),
                gr.update(visible=show_moss),
                gr.update(visible=show_chroma),
                # 27-29. File overrides
                gr.update(choices=unets, value=u_val),
                gr.update(choices=tes, value=t_val),
                gr.update(choices=vaes, value=v_val),
                # 30-32. Reset ketiga slot Custom LoRA ke (None) agar bebas crash lintas arsitektur!
                gr.update(choices=loras, value="(None)"),
                gr.update(choices=loras, value="(None)"),
                gr.update(choices=loras, value="(None)"),
                # 33-36. Status On-Demand & WanGP Profile & Telemetry Bar & Stepper
                format_on_demand_status_md(eng_id),
                gr.update(value=backend_mgr.memory_manager.active_profile_id),
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(
                    banner_msg=(
                        f"🔄 Engine diubah ke [{eng.display_name}] · Preset '{def_preset}' "
                        f"({p_cfg.get('steps', d['steps'])} steps, CFG {p_cfg.get('cfg', d['cfg'])}, "
                        f"{target_sampler}/{target_scheduler}) tersinkronisasi otomatis!"
                    )
                ),
            )

        def on_preset_change(selected_eng_val: str, preset_label: str):
            eng_id = resolve_engine_id(selected_eng_val)
            eng = engines[eng_id]
            d = eng.default_params
            presets = d.get("presets", {})
            p_cfg = presets.get(preset_label, {})
            if not p_cfg:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            samplers, schedulers = _get_merged_samplers_and_schedulers()
            target_sampler = str(p_cfg.get("sampler", d["sampler"]))
            target_scheduler = str(p_cfg.get("scheduler", d["scheduler"]))
            if target_sampler not in samplers:
                samplers.append(target_sampler)
            if target_scheduler not in schedulers:
                schedulers.append(target_scheduler)

            steps_v = int(p_cfg.get("steps", d["steps"]))
            cfg_v = float(p_cfg.get("cfg", d["cfg"]))
            shift_v = float(p_cfg.get("shift", d["shift"]))

            return (
                gr.update(value=steps_v),
                gr.update(value=cfg_v),
                gr.update(value=shift_v),
                gr.update(choices=samplers, value=target_sampler),
                gr.update(choices=schedulers, value=target_scheduler),
                gr.update(value=bool(p_cfg.get("use_lazycache", False))),
                gr.update(value=bool(p_cfg.get("use_slg", False))),
                gr.update(value=int(p_cfg.get("skip_layer", d.get("skip_layer", 22)))),
                gr.update(value=float(p_cfg.get("beta_alpha", d.get("beta_alpha", 0.45)))),
                gr.update(value=float(p_cfg.get("beta_beta", d.get("beta_beta", 0.45)))),
                gr.update(value=int(p_cfg.get("min_padding", d.get("min_padding", 1)))),
                gr.update(value=float(p_cfg.get("lightx2v_strength", d.get("lightx2v_strength", 1.0)))),
                render_stepper_html(
                    banner_msg=(
                        f"⚡ Preset [{preset_label}] diterapkan untuk {eng_id.upper()}: "
                        f"Steps={steps_v}, CFG={cfg_v}, Sampler={target_sampler}, Scheduler={target_scheduler}."
                    )
                ),
            )

        def on_aspect_change(selected_eng_val: str, aspect_key: str):
            eng_id = resolve_engine_id(selected_eng_val)
            ratios = engines[eng_id].default_params.get("aspect_ratios", {})
            if aspect_key in ratios:
                w, h = ratios[aspect_key]
                return gr.update(value=w), gr.update(value=h)
            return gr.update(), gr.update()

        def on_wan_start_image_change(pil_img: Optional[Image.Image], auto_crop: bool, cur_w: int, cur_h: int):
            if pil_img is None:
                return gr.update(), gr.update(), "<div style='font-size:0.82rem; color:#94a3b8;'>Unggah gambar start frame terlebih dahulu.</div>"
            ow, oh = pil_img.size
            rec_w, rec_h = Wan21Engine.compute_optimal_480p_size(ow, oh, target_pixels=int(cur_w) * int(cur_h))
            new_w, new_h = (rec_w, rec_h) if auto_crop else (int(cur_w), int(cur_h))
            info = (
                f"<div style='font-size:0.82rem; padding:6px 10px; border-radius:8px; "
                f"background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3);'>"
                f"📐 <b>Asli:</b> {ow}×{oh} px ➔ <b>Target 480P Optimal:</b> {new_w}×{new_h} px (Center-Crop Lanczos)"
                f"</div>"
            )
            return gr.update(value=new_w), gr.update(value=new_h), info

        def append_tag_to_prompt(cur_prompt: str, tag_key: str) -> str:
            phrase = QUICK_MOTION_TAGS.get(tag_key, "")
            cur = (cur_prompt or "").strip().rstrip(",")
            if not cur:
                return phrase
            if phrase.lower() in cur.lower():
                return cur_prompt
            return f"{cur}, {phrase}"

        def on_download_active_engine(selected_eng_val: str, progress=gr.Progress(track_tqdm=True)):
            eng_id = resolve_engine_id(selected_eng_val)
            progress(0.05, desc=f"Mengunduh bobot {eng_id} via Aria2c 16-koneksi...")
            dl_mgr.download_engine(eng_id)
            unets, u_val, tes, t_val, vaes, v_val, loras = get_file_dropdown_choices(eng_id)
            return (
                format_on_demand_status_md(eng_id),
                gr.update(choices=unets, value=u_val),
                gr.update(choices=tes, value=t_val),
                gr.update(choices=vaes, value=v_val),
                gr.update(choices=loras),
                gr.update(choices=loras),
                gr.update(choices=loras),
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg=f"✅ Seluruh bobot untuk engine [{eng_id.upper()}] berhasil diunduh dan siap digunakan!"),
            )

        def on_clean_other_engines(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            backend_mgr.free_memory(unload_models=True)
            removable = get_other_engine_exclusive_files(eng_id, MODELS_DIR)
            freed_bytes = 0
            removed_count = 0
            for p in removable:
                if p.is_file():
                    try:
                        freed_bytes += p.stat().st_size
                        p.unlink(missing_ok=True)
                        removed_count += 1
                    except Exception:
                        pass
            trim_native_memory()
            freed_gb = freed_bytes / (1024**3)
            unets, u_val, tes, t_val, vaes, v_val, loras = get_file_dropdown_choices(eng_id)
            return (
                format_on_demand_status_md(eng_id),
                gr.update(choices=unets, value=u_val),
                gr.update(choices=tes, value=t_val),
                gr.update(choices=vaes, value=v_val),
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(
                    banner_msg=f"🗑️ Berhasil menghapus {removed_count} file engine lain dan membebaskan {freed_gb:.2f} GiB ruang disk!"
                ),
            )

        def on_refresh_disk(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            unets, u_val, tes, t_val, vaes, v_val, loras = get_file_dropdown_choices(eng_id)
            return (
                format_on_demand_status_md(eng_id),
                gr.update(choices=unets, value=u_val),
                gr.update(choices=tes, value=t_val),
                gr.update(choices=vaes, value=v_val),
                gr.update(choices=loras),
                gr.update(choices=loras),
                gr.update(choices=loras),
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg="🔄 Daftar model & LoRA di disk berhasil diperbarui."),
            )

        def on_custom_url_download(raw_urls: str, target_folder: str, selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            entries = UniversalDownloader.parse_entries(raw_urls)
            if not entries:
                raise gr.Error("Masukkan minimal satu link HuggingFace / Civitai / ModelScope.")
            downloaded_names: List[str] = []
            for item in entries:
                paths = dl_mgr.download_custom_url(item, target_subdir=target_folder)
                downloaded_names.extend(p.name for p in paths)
            unets, u_val, tes, t_val, vaes, v_val, loras = get_file_dropdown_choices(eng_id)
            first_lora = downloaded_names[0] if (target_folder == "loras" and downloaded_names) else "(None)"
            return (
                gr.update(choices=unets, value=u_val),
                gr.update(choices=tes, value=t_val),
                gr.update(choices=vaes, value=v_val),
                gr.update(choices=loras, value=first_lora if first_lora in loras else "(None)"),
                gr.update(choices=loras),
                gr.update(choices=loras),
                format_on_demand_status_md(eng_id),
                render_stepper_html(
                    banner_msg=f"✅ Berhasil mengunduh {len(downloaded_names)} file ke `{target_folder}`: {', '.join(downloaded_names)}"
                ),
            )

        def on_upload_local_lora(filepath: Optional[str], selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            if not filepath:
                raise gr.Error("Pilih file .safetensors terlebih dahulu.")
            src = Path(filepath)
            if src.suffix.lower() != ".safetensors":
                raise gr.Error("Hanya file .safetensors yang didukung.")
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", src.name)
            dst = MODELS_DIR / "loras" / safe_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            _, _, _, _, _, _, loras = get_file_dropdown_choices(eng_id)
            return (
                gr.update(choices=loras, value=dst.name),
                gr.update(choices=loras),
                gr.update(choices=loras),
                render_stepper_html(banner_msg=f"✅ LoRA lokal `{dst.name}` berhasil dipasang ke Slot 1!"),
            )

        def on_wan_autotune():
            return (
                gr.update(value=0.85),
                gr.update(value=6),
                gr.update(value=1.0),
                render_stepper_html(
                    banner_msg="⚖️ Wan 2.1 Auto-Tune aktif: Lightx2v Strength = 0.85, Steps = 6, CFG = 1.0."
                ),
            )

        def on_wangp_profile_change(profile_id: str, selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            prof = backend_mgr.memory_manager.set_profile(profile_id)
            return (
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg=f"🧠 WanGP Memory Profile diubah ke: {prof.display_name}"),
            )

        def on_stop_execution(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            msg = backend_mgr.interrupt()
            return (
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg=f"🛑 {msg}"),
            )

        def on_flush_memory(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            msg = backend_mgr.free_memory(unload_models=True)
            return (
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg=f"🧹 {msg}"),
            )

        def on_restart_daemon(selected_eng_val: str):
            eng_id = resolve_engine_id(selected_eng_val)
            msg = backend_mgr.restart(engine_id=eng_id)
            return (
                backend_mgr.get_recent_logs(40),
                render_telemetry_bar_html(backend_mgr, eng_id),
                render_stepper_html(banner_msg=f"♻️ {msg}"),
            )

        def on_run_standalone_upscale(
            img_path: Optional[str],
            upscaler_model: str,
            scale_str: str,
            selected_eng_val: str,
            progress=gr.Progress(track_tqdm=False),
        ):
            eng_id = resolve_engine_id(selected_eng_val)
            if not img_path or not Path(img_path).is_file():
                raise gr.Error("Unggah gambar atau klik 'Kirim Hasil Terakhir ke Upscaler' terlebih dahulu.")

            # Pastikan model upscaler terunduh
            if upscaler_model in UPSCALER_CATALOG:
                spec = UPSCALER_CATALOG[upscaler_model]
                ok, _ = check_file_installed(spec, MODELS_DIR)
                if not ok:
                    progress(0.05, desc=f"Mengunduh model upscaler {upscaler_model}...")
                    dl_mgr.download_upscaler(upscaler_model)

            up_eng = engines["upscaler"]
            wf = up_eng.build_workflow(
                {
                    "input_image": img_path,
                    "upscaler_model": upscaler_model,
                    "upscale_factor": scale_str,
                }
            )
            out_bytes_list, timings = backend_mgr.execute_workflow(
                engine_id="upscaler",
                workflow=wf,
                progress_cb=lambda r, d: progress(r, desc=d),
            )
            if not out_bytes_list:
                raise RuntimeError("Upscaler tidak mengembalikan output gambar.")

            saved_img, _ = gal_mgr.save_image_item(
                image_bytes=out_bytes_list[0],
                engine_id="upscaler",
                seed=0,
                metadata={
                    "engine_display": f"Standalone Tiled Upscaler ({upscaler_model})",
                    "prompt": f"Standalone Upscale ({scale_str}) menggunakan {upscaler_model}",
                    "upscale_info": f"{upscaler_model} ({scale_str} Lanczos)",
                    **timings,
                },
            )
            stepper_done = render_stepper_html(
                s1_state="done",
                s1_val=f"✔️ {timings['t_stage1']}s",
                s1_sub="Load Image & Spandrel Model",
                s2_state="done",
                s2_val=f"✔️ {timings['t_stage2']}s",
                s2_sub="512×512 Tiled Super-Resolution",
                s3_state="done",
                s3_val=f"✔️ {timings['t_stage3']}s",
                s3_sub=f"Lanczos {scale_str} Output",
                banner_msg=f"✨ Standalone Upscale ({scale_str} · {upscaler_model}) selesai dalam {timings['total_elapsed_sec']} detik!",
            )
            return (
                gr.update(visible=True, value=[(str(saved_img), f"Upscaled {scale_str} ({timings['total_elapsed_sec']}s)")]),
                gr.update(visible=False),
                str(saved_img),
                stepper_done,
                render_telemetry_bar_html(backend_mgr, eng_id),
            )

        def on_main_generate(
            selected_eng_val: str,
            preset_mode: str,
            prompt: str,
            negative_prompt: str,
            width: int,
            height: int,
            batch_count: int,
            seed: int,
            steps: int,
            cfg: float,
            shift: float,
            sampler: str,
            scheduler: str,
            unet_name: str,
            clip_name: str,
            vae_name: str,
            lora1: str,
            str1: float,
            lora2: str,
            str2: float,
            lora3: str,
            str3: float,
            use_lazycache: bool,
            cache_thresh: float,
            moss_slg: bool,
            moss_skip_layer: int,
            chroma_alpha: float,
            chroma_beta: float,
            chroma_min_pad: int,
            auto_upscale: bool,
            upscaler_model: str,
            upscale_scale_str: str,
            wan_start_pil: Optional[Image.Image],
            wan_auto_crop: bool,
            wan_frames: int,
            wan_fps_choice: str,
            wan_lightx2v_str: float,
            qwen_img1: Optional[str],
            qwen_img2: Optional[str],
            qwen_img3: Optional[str],
            qwen_lock_ratio: bool,
            qwen_ref_res: int,
            progress=gr.Progress(track_tqdm=False),
        ):
            eng_id = resolve_engine_id(selected_eng_val)
            eng = engines[eng_id]

            try:
                # 1. Pastikan seluruh bobot model untuk engine ini sudah ada (On-Demand Auto-Check)
                ok_installed, missing_specs = check_engine_installed(eng_id, MODELS_DIR)
                if not ok_installed:
                    progress(0.02, desc=f"Mengunduh {len(missing_specs)} file model yang belum ada untuk {eng_id}...")
                    dl_mgr.download_engine(eng_id)

                if auto_upscale and not eng.is_video and upscaler_model in UPSCALER_CATALOG:
                    up_ok, _ = check_file_installed(UPSCALER_CATALOG[upscaler_model], MODELS_DIR)
                    if not up_ok:
                        progress(0.04, desc=f"Mengunduh model Tiled Upscaler {upscaler_model}...")
                        dl_mgr.download_upscaler(upscaler_model)

                lora_slots = [(lora1, float(str1)), (lora2, float(str2)), (lora3, float(str3))]
                base_seed = int(seed)

                # ==============================================================
                # CABANG A: ENGINE VIDEO (WAN 2.1 I2V 14B 480P)
                # ==============================================================
                if eng.is_video:
                    actual_seed = random.randint(1, 2**31 - 1) if base_seed < 0 else base_seed
                    params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "frames": wan_frames,
                        "fps_choice": wan_fps_choice,
                        "steps": steps,
                        "cfg": cfg,
                        "shift": shift,
                        "sampler": sampler,
                        "scheduler": scheduler,
                        "seed": actual_seed,
                        "unet_name": unet_name,
                        "clip_name": clip_name,
                        "vae_name": vae_name,
                        "lightx2v_strength": wan_lightx2v_str,
                        "lora_slots": lora_slots,
                        "start_image": wan_start_pil,
                        "auto_aspect_crop": wan_auto_crop,
                    }
                    clean_p = eng.sanitize_params(params)
                    workflow = eng.build_workflow(params)

                    frame_bytes_list, timings = backend_mgr.execute_workflow(
                        engine_id=eng_id,
                        workflow=workflow,
                        progress_cb=lambda r, d: progress(r, desc=d),
                    )
                    target_fps = Wan21Engine.parse_target_fps(wan_fps_choice)
                    progress(0.94, desc=f"Menyusun video MP4 ({target_fps} FPS True-Duration)...")
                    mp4_tmp, jpg_tmp, true_dur = Wan21Engine.encode_frames_to_mp4(
                        frame_bytes_list=frame_bytes_list,
                        target_fps=target_fps,
                        output_dir=GALLERY_DIR,
                        seed=actual_seed,
                    )

                    active_loras = [
                        {"name": n, "strength": s}
                        for n, s in BaseEngine.filter_custom_loras(lora_slots)
                    ]
                    saved_mp4, _, _ = gal_mgr.save_video_item(
                        mp4_src_path=mp4_tmp,
                        thumbnail_jpg_src_path=jpg_tmp,
                        engine_id=eng_id,
                        seed=actual_seed,
                        metadata={
                            "engine_display": eng.display_name,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "width": clean_p["width"],
                            "height": clean_p["height"],
                            "frames": clean_p["frames"],
                            "fps": target_fps,
                            "duration_sec": true_dur,
                            "steps": clean_p["steps"],
                            "cfg": clean_p["cfg"],
                            "shift": clean_p["shift"],
                            "sampler": clean_p["sampler"],
                            "scheduler": clean_p["scheduler"],
                            "unet_name": unet_name,
                            "clip_name": clip_name,
                            "vae_name": vae_name,
                            "lightx2v_strength": wan_lightx2v_str,
                            "loras": active_loras,
                            **timings,
                        },
                    )
                    stepper_done = render_stepper_html(
                        s1_state="done",
                        s1_val=f"✔️ {timings['t_stage1']}s",
                        s1_sub=f"UMT5-XXL + CLIP Vision ({clean_p['width']}×{clean_p['height']})",
                        s2_state="done",
                        s2_val=f"✔️ {timings['t_stage2']}s",
                        s2_sub=f"Wan 14B ({clean_p['steps']} Steps, Lightx2v {wan_lightx2v_str})",
                        s3_state="done",
                        s3_val=f"✔️ {timings['t_stage3']}s",
                        s3_sub=f"{true_dur}s Video @ {target_fps} FPS",
                        banner_msg=(
                            f"✅ Video Wan 2.1 selesai dalam {timings['total_elapsed_sec']} detik · "
                            f"Durasi: {true_dur} dtk ({target_fps} FPS) · Seed: {actual_seed}"
                        ),
                    )
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True, value=str(saved_mp4)),
                        str(jpg_tmp) if jpg_tmp.is_file() else "",
                        stepper_done,
                        render_telemetry_bar_html(backend_mgr, eng_id),
                    )

                # ==============================================================
                # CABANG B: 6 ENGINE GAMBAR (ANIMA, Z-IMAGE, MOSS, QWEN 2.1, KREA 2, CHROMA)
                # ==============================================================
                count = max(1, min(4, int(batch_count)))
                rendered_gallery_items: List[Tuple[str, str]] = []
                last_img_path_str = ""
                total_s1, total_s2, total_s3, total_all = 0.0, 0.0, 0.0, 0.0
                used_seeds: List[int] = []

                has_lazy = backend_mgr.has_node("LazyCache")
                has_qwen_c = backend_mgr.has_node("QwenImage21Cache")

                for idx in range(count):
                    this_seed = (
                        random.randint(1, 2**31 - 1)
                        if base_seed < 0
                        else (base_seed + idx) % (2**31 - 1)
                    )
                    used_seeds.append(this_seed)

                    params = {
                        "preset_mode": preset_mode,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg": cfg,
                        "shift": shift,
                        "flow_shift": shift,
                        "sampler": sampler,
                        "scheduler": scheduler,
                        "seed": this_seed,
                        "unet_name": unet_name,
                        "clip_name": clip_name,
                        "vae_name": vae_name,
                        "lora_slots": lora_slots,
                        "use_lazycache": use_lazycache,
                        "cache_threshold": cache_thresh,
                        "has_lazycache": has_lazy,
                        "has_qwen_cache": has_qwen_c,
                        "use_slg": moss_slg,
                        "skip_layer": moss_skip_layer,
                        "beta_alpha": chroma_alpha,
                        "beta_beta": chroma_beta,
                        "min_padding": chroma_min_pad,
                        "ref_img1": qwen_img1,
                        "ref_img2": qwen_img2,
                        "ref_img3": qwen_img3,
                        "lock_ref_size": qwen_lock_ratio,
                        "ref_resolution": qwen_ref_res,
                        "auto_upscale": auto_upscale,
                        "upscaler_model": upscaler_model,
                        "upscale_factor": upscale_scale_str,
                    }
                    clean_p = eng.sanitize_params(params)
                    workflow = eng.build_workflow(params)

                    out_bytes_list, timings = backend_mgr.execute_workflow(
                        engine_id=eng_id,
                        workflow=workflow,
                        progress_cb=lambda r, d, b_idx=idx: progress(
                            (b_idx + r) / count,
                            desc=f"[Batch {b_idx+1}/{count}] {d}",
                        ),
                    )
                    total_s1 += timings["t_stage1"]
                    total_s2 += timings["t_stage2"]
                    total_s3 += timings["t_stage3"]
                    total_all += timings["total_elapsed_sec"]

                    active_loras = [
                        {"name": n, "strength": s}
                        for n, s in BaseEngine.filter_custom_loras(lora_slots)
                    ]
                    upscale_note = (
                        f"Tiled {upscale_scale_str} ({upscaler_model})"
                        if auto_upscale
                        else "Native Resolution"
                    )
                    for img_bytes in out_bytes_list:
                        saved_img, _ = gal_mgr.save_image_item(
                            image_bytes=img_bytes,
                            engine_id=eng_id,
                            seed=this_seed,
                            metadata={
                                "engine_display": eng.display_name,
                                "prompt": prompt,
                                "negative_prompt": negative_prompt,
                                "width": clean_p["width"],
                                "height": clean_p["height"],
                                "steps": clean_p["steps"],
                                "cfg": clean_p["cfg"],
                                "shift": clean_p.get("shift", 1.0),
                                "sampler": clean_p["sampler"],
                                "scheduler": clean_p["scheduler"],
                                "unet_name": unet_name,
                                "clip_name": clip_name,
                                "vae_name": vae_name,
                                "loras": active_loras,
                                "upscale_info": upscale_note,
                                **timings,
                            },
                        )
                        last_img_path_str = str(saved_img)
                        rendered_gallery_items.append(
                            (str(saved_img), f"{eng_id.upper()} · seed {this_seed} ({timings['total_elapsed_sec']}s)")
                        )

                stepper_done = render_stepper_html(
                    s1_state="done",
                    s1_val=f"✔️ {total_s1:.1f}s",
                    s1_sub=f"WanGP Guard + {clip_name[:22]}",
                    s2_state="done",
                    s2_val=f"✔️ {total_s2:.1f}s",
                    s2_sub=f"{unet_name[:24]} ({steps} Steps)",
                    s3_state="done",
                    s3_val=f"✔️ {total_s3:.1f}s",
                    s3_sub="VAE Decode + Tiled Upscale" if auto_upscale else "VAE Decode",
                    banner_msg=(
                        f"✅ Selesai {len(rendered_gallery_items)} gambar dalam {total_all:.1f} detik · "
                        f"Engine: {eng_id.upper()} · Seed: {', '.join(map(str, used_seeds))}"
                    ),
                )
                return (
                    gr.update(visible=True, value=rendered_gallery_items),
                    gr.update(visible=False),
                    last_img_path_str,
                    stepper_done,
                    render_telemetry_bar_html(backend_mgr, eng_id),
                )
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, flush=True)
                err_stepper = render_stepper_html(
                    s1_state="idle",
                    s2_state="idle",
                    s3_state="idle",
                    banner_msg=f"❌ Generate gagal: {type(exc).__name__}: {exc}",
                )
                return (
                    gr.update(),
                    gr.update(),
                    "",
                    err_stepper,
                    render_telemetry_bar_html(backend_mgr, eng_id),
                )

        def on_inspect_history_item(evt: gr.SelectData, media_paths: List[str]):
            if not media_paths or evt is None or evt.index >= len(media_paths):
                return (
                    gr.update(visible=True, value=None),
                    gr.update(visible=False, value=None),
                    "Item tidak ditemukan.",
                    "",
                )
            selected_path = media_paths[evt.index]
            meta = gal_mgr.get_metadata_for_media(selected_path)
            md = GalleryManager.format_metadata_markdown(meta)
            is_vid = selected_path.lower().endswith(".mp4")
            return (
                gr.update(visible=(not is_vid), value=(selected_path if not is_vid else None)),
                gr.update(visible=is_vid, value=(selected_path if is_vid else None)),
                md,
                selected_path,
            )

        def on_copy_history_params(selected_media_path: str):
            meta = gal_mgr.get_metadata_for_media(selected_media_path)
            if not meta:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    render_stepper_html(banner_msg="⚠️ Klik salah satu item di Galeri terlebih dahulu."),
                )
            return (
                gr.update(value=meta.get("prompt", "")),
                gr.update(value=meta.get("negative_prompt", "")),
                gr.update(value=int(meta.get("width", 1024))),
                gr.update(value=int(meta.get("height", 1024))),
                gr.update(value=int(meta.get("steps", 8))),
                gr.update(value=float(meta.get("cfg", 1.0))),
                gr.update(value=float(meta.get("shift", 1.0))),
                gr.update(value=int(meta.get("seed", -1))),
                render_stepper_html(
                    banner_msg=f"♻️ Parameter dari `{meta.get('file_name', '')}` berhasil disalin ke Studio Workspace!"
                ),
            )

        def on_send_selected_to_upscaler(selected_media_path: str):
            if not selected_media_path or selected_media_path.lower().endswith(".mp4"):
                return (
                    gr.update(),
                    gr.update(),
                    render_stepper_html(banner_msg="⚠️ Pilih sebuah item gambar (.png) di Galeri untuk dikirim ke Standalone Upscaler."),
                )
            return (
                gr.update(value=True),
                gr.update(value=selected_media_path),
                render_stepper_html(
                    banner_msg=f"🔍 Gambar `{Path(selected_media_path).name}` telah dimuat ke Standalone Upscaler di tab Studio Workspace!"
                ),
            )

        def on_delete_history_item(selected_media_path: str):
            gal_mgr.delete_item(selected_media_path)
            gal_tuples, gal_paths, gal_summary = gal_mgr.list_items()
            return (
                gal_tuples,
                gal_paths,
                gal_summary,
                gr.update(visible=True, value=None),
                gr.update(visible=False, value=None),
                "🗑️ Item yang dipilih telah dihapus dari galeri.",
                "",
            )

        def on_export_gallery_zip():
            zp = gal_mgr.export_all_to_zip()
            return str(zp) if zp else None

        # ======================================================================
        # 7. BINDING SELURUH EVENT LISTENER (DENGAN NON-BLOCKING GALLERY QUEUE)
        # ======================================================================
        chk_input_image.change(
            fn=lambda v: gr.update(visible=bool(v)),
            inputs=[chk_input_image],
            outputs=[input_image_drawer],
            queue=False,
        )
        chk_advanced.change(
            fn=lambda v: gr.update(visible=bool(v)),
            inputs=[chk_advanced],
            outputs=[advanced_drawer],
            queue=False,
        )

        engine_selector.change(
            fn=on_engine_change,
            inputs=[engine_selector],
            outputs=[
                main_image_gallery,
                main_video_player,
                prompt_box,
                negative_prompt_box,
                preset_mode_radio,
                aspect_dropdown,
                width_slider,
                height_slider,
                video_specific_group,
                image_specific_group,
                wan_lora_Controls_group,
                steps_slider,
                cfg_slider,
                shift_slider,
                sampler_dd,
                scheduler_dd,
                use_lazycache_chk,
                moss_slg_chk,
                moss_skip_layer_num,
                chroma_alpha_slider,
                chroma_beta_slider,
                chroma_min_pad_slider,
                wan_lightx2v_slider,
                lazycache_arch_row,
                moss_arch_row,
                chroma_arch_row,
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                lora1_dd,
                lora2_dd,
                lora3_dd,
                on_demand_status_md,
                wangp_profile_dd,
                telemetry_html,
                stepper_bar_html,
            ],
            queue=False,
        )

        preset_mode_radio.change(
            fn=on_preset_change,
            inputs=[engine_selector, preset_mode_radio],
            outputs=[
                steps_slider,
                cfg_slider,
                shift_slider,
                sampler_dd,
                scheduler_dd,
                use_lazycache_chk,
                moss_slg_chk,
                moss_skip_layer_num,
                chroma_alpha_slider,
                chroma_beta_slider,
                chroma_min_pad_slider,
                wan_lightx2v_slider,
                stepper_bar_html,
            ],
            queue=False,
        )

        aspect_dropdown.change(
            fn=on_aspect_change,
            inputs=[engine_selector, aspect_dropdown],
            outputs=[width_slider, height_slider],
            queue=False,
        )

        wan_start_img.change(
            fn=on_wan_start_image_change,
            inputs=[wan_start_img, wan_auto_crop_chk, width_slider, height_slider],
            outputs=[width_slider, height_slider, wan_ratio_info_html],
            queue=False,
        )

        # Quick Motion Tags untuk Wan 2.1
        btn_motion_push.click(lambda p: append_tag_to_prompt(p, "+ Slow Push-In"), inputs=[prompt_box], outputs=[prompt_box], queue=False)
        btn_motion_orbit.click(lambda p: append_tag_to_prompt(p, "+ Orbit Shot"), inputs=[prompt_box], outputs=[prompt_box], queue=False)
        btn_motion_pan.click(lambda p: append_tag_to_prompt(p, "+ Smooth Pan"), inputs=[prompt_box], outputs=[prompt_box], queue=False)
        btn_motion_wind.click(lambda p: append_tag_to_prompt(p, "+ Wind & Hair"), inputs=[prompt_box], outputs=[prompt_box], queue=False)
        btn_motion_hand.click(lambda p: append_tag_to_prompt(p, "+ Handheld Realism"), inputs=[prompt_box], outputs=[prompt_box], queue=False)

        # Standalone Upscaler (Non-Blocking terhadap History Gallery)
        send_last_to_upscaler_btn.click(
            fn=lambda p: gr.update(value=p) if p else gr.update(),
            inputs=[last_rendered_image_state],
            outputs=[upscale_input_img],
            queue=False,
        )
        upscale_evt = run_standalone_upscale_btn.click(
            fn=on_run_standalone_upscale,
            inputs=[
                upscale_input_img,
                upscaler_model_dropdown,
                upscale_scale_radio,
                engine_selector,
            ],
            outputs=[
                main_image_gallery,
                main_video_player,
                last_rendered_image_state,
                stepper_bar_html,
                telemetry_html,
            ],
            concurrency_limit=1,
        )
        upscale_evt.then(
            fn=gal_mgr.list_items,
            inputs=[],
            outputs=[history_gallery, gallery_media_paths_state, history_summary_md],
            show_progress="hidden",
            queue=False,
        )

        # Models & LoRA Tab Events
        btn_download_engine.click(
            fn=on_download_active_engine,
            inputs=[engine_selector],
            outputs=[
                on_demand_status_md,
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                lora1_dd,
                lora2_dd,
                lora3_dd,
                telemetry_html,
                stepper_bar_html,
            ],
        )
        btn_clean_other_engines.click(
            fn=on_clean_other_engines,
            inputs=[engine_selector],
            outputs=[
                on_demand_status_md,
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                telemetry_html,
                stepper_bar_html,
            ],
        )
        btn_refresh_disk.click(
            fn=on_refresh_disk,
            inputs=[engine_selector],
            outputs=[
                on_demand_status_md,
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                lora1_dd,
                lora2_dd,
                lora3_dd,
                telemetry_html,
                stepper_bar_html,
            ],
            queue=False,
        )
        btn_custom_dl.click(
            fn=on_custom_url_download,
            inputs=[custom_dl_url_box, custom_dl_folder_dd, engine_selector],
            outputs=[
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                lora1_dd,
                lora2_dd,
                lora3_dd,
                on_demand_status_md,
                stepper_bar_html,
            ],
        )
        btn_upload_local_lora.click(
            fn=on_upload_local_lora,
            inputs=[local_lora_file, engine_selector],
            outputs=[lora1_dd, lora2_dd, lora3_dd, stepper_bar_html],
            queue=False,
        )
        btn_wan_autotune.click(
            fn=on_wan_autotune,
            inputs=[],
            outputs=[wan_lightx2v_slider, steps_slider, cfg_slider, stepper_bar_html],
            queue=False,
        )

        # Sampler & Developer Tab Events
        wangp_profile_dd.change(
            fn=on_wangp_profile_change,
            inputs=[wangp_profile_dd, engine_selector],
            outputs=[telemetry_html, stepper_bar_html],
            queue=False,
        )
        btn_refresh_log.click(
            fn=lambda: backend_mgr.get_recent_logs(40),
            inputs=[],
            outputs=[daemon_log_box],
            queue=False,
        )
        btn_restart_daemon.click(
            fn=on_restart_daemon,
            inputs=[engine_selector],
            outputs=[daemon_log_box, telemetry_html, stepper_bar_html],
        )

        # Main Generate / Stop / Flush Events
        # PENTING: history_gallery TIDAK dimasukkan ke dalam outputs langsung generate_btn.click(...)
        # agar selama proses render berjalan, tab "🗂️ Galeri & History Inspector" TIDAK terkunci oleh
        # loading spinner Gradio! Sebagai gantinya, galeri diperbarui secara instan & senyap melalui .then(...)
        gen_evt = generate_btn.click(
            fn=on_main_generate,
            inputs=[
                engine_selector,
                preset_mode_radio,
                prompt_box,
                negative_prompt_box,
                width_slider,
                height_slider,
                batch_count_slider,
                seed_number,
                steps_slider,
                cfg_slider,
                shift_slider,
                sampler_dd,
                scheduler_dd,
                unet_override_dd,
                clip_override_dd,
                vae_override_dd,
                lora1_dd,
                lora1_str,
                lora2_dd,
                lora2_str,
                lora3_dd,
                lora3_str,
                use_lazycache_chk,
                lazycache_thresh_slider,
                moss_slg_chk,
                moss_skip_layer_num,
                chroma_alpha_slider,
                chroma_beta_slider,
                chroma_min_pad_slider,
                chk_auto_upscale,
                upscaler_model_dropdown,
                upscale_scale_radio,
                wan_start_img,
                wan_auto_crop_chk,
                wan_frames_slider,
                wan_fps_dropdown,
                wan_lightx2v_slider,
                qwen_ref1,
                qwen_ref2,
                qwen_ref3,
                qwen_lock_ratio_chk,
                qwen_ref_res_slider,
            ],
            outputs=[
                main_image_gallery,
                main_video_player,
                last_rendered_image_state,
                stepper_bar_html,
                telemetry_html,
            ],
            concurrency_limit=1,
        )
        gen_evt.then(
            fn=gal_mgr.list_items,
            inputs=[],
            outputs=[history_gallery, gallery_media_paths_state, history_summary_md],
            show_progress="hidden",
            queue=False,
        )

        stop_btn.click(
            fn=on_stop_execution,
            inputs=[engine_selector],
            outputs=[telemetry_html, stepper_bar_html],
            queue=False,
        )
        flush_btn.click(
            fn=on_flush_memory,
            inputs=[engine_selector],
            outputs=[telemetry_html, stepper_bar_html],
            queue=False,
        )

        # Dedicated Galeri & History Inspector Tab Events (100% Non-Blocking: queue=False)
        demo.load(
            fn=gal_mgr.list_items,
            inputs=[],
            outputs=[history_gallery, gallery_media_paths_state, history_summary_md],
            show_progress="hidden",
            queue=False,
        )
        btn_refresh_history.click(
            fn=gal_mgr.list_items,
            inputs=[],
            outputs=[history_gallery, gallery_media_paths_state, history_summary_md],
            show_progress="hidden",
            queue=False,
        )
        history_gallery.select(
            fn=on_inspect_history_item,
            inputs=[gallery_media_paths_state],
            outputs=[
                inspector_image_preview,
                inspector_video_preview,
                inspector_meta_md,
                selected_history_media_state,
            ],
            show_progress="hidden",
            queue=False,
        )
        btn_copy_params.click(
            fn=on_copy_history_params,
            inputs=[selected_history_media_state],
            outputs=[
                prompt_box,
                negative_prompt_box,
                width_slider,
                height_slider,
                steps_slider,
                cfg_slider,
                shift_slider,
                seed_number,
                stepper_bar_html,
            ],
            show_progress="hidden",
            queue=False,
        )
        btn_send_selected_to_upscaler.click(
            fn=on_send_selected_to_upscaler,
            inputs=[selected_history_media_state],
            outputs=[chk_input_image, upscale_input_img, stepper_bar_html],
            show_progress="hidden",
            queue=False,
        )
        btn_delete_history_item.click(
            fn=on_delete_history_item,
            inputs=[selected_history_media_state],
            outputs=[
                history_gallery,
                gallery_media_paths_state,
                history_summary_md,
                inspector_image_preview,
                inspector_video_preview,
                inspector_meta_md,
                selected_history_media_state,
            ],
            show_progress="hidden",
            queue=False,
        )
        btn_export_zip.click(
            fn=on_export_gallery_zip,
            inputs=[],
            outputs=[zip_download_file],
            queue=False,
        )

    return demo
