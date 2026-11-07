# -*- coding: utf-8 -*-
"""
WanGP-Style (`mmgp`) RAM/VRAM Manager & Native C++ Memory Optimizer (`core/memory_manager.py`)
Khusus untuk Google Colab Tesla T4 (15.0 GiB GPU VRAM & ~12.7 GiB CPU System RAM).
Dilengkapi akselerator C++ (`core/studio_native_mem.cpp` -> `libstudio_mem.so`):
- `mallopt(M_MMAP_THRESHOLD, 128 KB)` & `M_ARENA_MAX = 2` (mencegah fragmentasi heap C++ pada GGUF/Safetensors)
- `malloc_trim(0)` via C++ native bridge
- `posix_fadvise(..., POSIX_FADV_DONTNEED)` untuk membebaskan Linux VFS page cache dari file bobot yang tidak aktif
- Fast `/proc/meminfo` C++ reader.
"""

import ctypes
import gc
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 1. Set konfigurasi alokasi CUDA sebelum proses PyTorch dimulai
CUDA_ALLOC_CONF_VALUE = (
    "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF_VALUE

# Ambang batas darurat System RAM CPU Colab untuk engine yang sama (1.2 GiB)
# Agar model tetap hangat (Warm-Cache) pada render ke-2 dst. tanpa di-unload paksa.
PREFLIGHT_MIN_AVAILABLE_RAM_GIB: float = 1.2

_CORE_DIR = Path(__file__).resolve().parent
_CPP_SRC_PATH = _CORE_DIR / "studio_native_mem.cpp"
_SO_LIB_PATH = _CORE_DIR / "libstudio_mem.so"
_NATIVE_LIB: Optional[ctypes.CDLL] = None
_NATIVE_INIT_DONE: bool = False


def load_or_compile_native_mem_lib() -> Optional[ctypes.CDLL]:
    """
    Mengompilasi `core/studio_native_mem.cpp` menjadi `core/libstudio_mem.so` secara otomatis
    menggunakan `g++ -O3 -shared -fPIC` jika belum ada, lalu memuatnya via `ctypes.CDLL`
    dan menjalankan `studio_init_allocator(128, 128, 2)`.
    """
    global _NATIVE_LIB, _NATIVE_INIT_DONE
    if _NATIVE_INIT_DONE:
        return _NATIVE_LIB

    _NATIVE_INIT_DONE = True
    try:
        need_compile = (
            _CPP_SRC_PATH.is_file()
            and (
                not _SO_LIB_PATH.is_file()
                or _CPP_SRC_PATH.stat().st_mtime > _SO_LIB_PATH.stat().st_mtime
            )
        )
        if need_compile and shutil.which("g++"):
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(_CPP_SRC_PATH),
                    "-o",
                    str(_SO_LIB_PATH),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if _SO_LIB_PATH.is_file():
            lib = ctypes.CDLL(str(_SO_LIB_PATH))
            lib.studio_init_allocator.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.studio_init_allocator.restype = ctypes.c_int
            lib.studio_aggressive_trim.argtypes = [ctypes.c_size_t]
            lib.studio_aggressive_trim.restype = ctypes.c_int
            lib.studio_evict_file_cache.argtypes = [ctypes.c_char_p]
            lib.studio_evict_file_cache.restype = ctypes.c_int
            lib.studio_read_meminfo_kib.argtypes = [
                ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long),
            ]
            lib.studio_read_meminfo_kib.restype = ctypes.c_int

            # Set M_MMAP_THRESHOLD=128KB, M_TRIM_THRESHOLD=128KB, M_ARENA_MAX=2
            lib.studio_init_allocator(128, 128, 2)
            _NATIVE_LIB = lib
    except Exception:
        _NATIVE_LIB = None

    return _NATIVE_LIB


# Inisialisasi C++ allocator sejak modul diimpor
load_or_compile_native_mem_lib()


def configure_cuda_alloc_env() -> str:
    """Memastikan environment variable PYTORCH_CUDA_ALLOC_CONF dan C++ allocator aktif."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF_VALUE
    load_or_compile_native_mem_lib()
    return os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def evict_file_page_cache(filepath: Path) -> bool:
    """
    Memanggil fungsi C++ `studio_evict_file_cache` (`posix_fadvise(POSIX_FADV_DONTNEED)`)
    untuk membebaskan Linux VFS page cache dari file bobot model berukuran besar.
    """
    lib = load_or_compile_native_mem_lib()
    if lib is None or not filepath.is_file():
        return False
    try:
        rc = lib.studio_evict_file_cache(str(filepath).encode("utf-8"))
        return rc == 0
    except Exception:
        return False


def evict_models_page_cache(models_dir: Path) -> int:
    """
    Melepas VFS page cache Linux untuk seluruh file bobot `.safetensors` / `.gguf` di `models_dir`
    saat transisi engine agar tidak memakan kuota System RAM cgroup Colab.
    """
    if not models_dir.exists():
        return 0
    evicted = 0
    for ext in ("*.safetensors", "*.gguf", "*.pth", "*.pt"):
        for fp in models_dir.rglob(ext):
            if fp.is_file() and fp.stat().st_size >= 50_000_000:
                if evict_file_page_cache(fp):
                    evicted += 1
    return evicted


def trim_native_memory() -> bool:
    """
    Membersihkan garbage collector Python, mengosongkan CUDA caching allocator,
    dan memanggil C++ `studio_aggressive_trim(0)` (atau fallback `libc.so.6:malloc_trim(0)`)
    agar fragmen heap C++ dari GGUF/Safetensors segera dikembalikan ke kernel Linux Google Colab.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    lib = load_or_compile_native_mem_lib()
    if lib is not None:
        try:
            lib.studio_aggressive_trim(0)
            return True
        except Exception:
            pass

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except Exception:
        return False


def get_system_ram_metrics() -> Tuple[float, float, float]:
    """
    Membaca `/proc/meminfo` (`MemTotal` & `MemAvailable`) melalui C++ native reader
    (atau fallback Python). Mengembalikan tuple: `(used_gib, available_gib, total_gib)`.
    """
    lib = load_or_compile_native_mem_lib()
    if lib is not None:
        try:
            t_kib = ctypes.c_long(0)
            a_kib = ctypes.c_long(0)
            if lib.studio_read_meminfo_kib(ctypes.byref(t_kib), ctypes.byref(a_kib)) == 0 and t_kib.value > 0:
                total_gib = t_kib.value / (1024**2)
                avail_gib = a_kib.value / (1024**2)
                used_gib = max(0.0, total_gib - avail_gib)
                return used_gib, avail_gib, total_gib
        except Exception:
            pass

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            text = meminfo_path.read_text(encoding="utf-8", errors="ignore")
            m_total = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
            m_avail = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
            if m_total and m_avail:
                total_gib = int(m_total.group(1)) / (1024**2)
                avail_gib = int(m_avail.group(1)) / (1024**2)
                used_gib = max(0.0, total_gib - avail_gib)
                return used_gib, avail_gib, total_gib
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_gib = (page_size * total_pages) / (1024**3)
        avail_gib = (page_size * avail_pages) / (1024**3)
        used_gib = max(0.0, total_gib - avail_gib)
        return used_gib, avail_gib, total_gib
    except Exception:
        return 2.5, 10.2, 12.7


def get_gpu_vram_metrics() -> Tuple[float, float, float, str]:
    """
    Mengembalikan tuple `(used_gib, free_gib, total_gib, gpu_name)` dari PyTorch CUDA.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gib = props.total_memory / (1024**3)
            used_gib = torch.cuda.memory_allocated(0) / (1024**3)
            free_gib = max(0.0, total_gib - used_gib)
            return used_gib, free_gib, total_gib, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return 0.0, 15.0, 15.0, "NVIDIA Tesla T4 (Standby)"


@dataclass(frozen=True)
class WanGPMemoryProfile:
    """Definisi profil manajemen memori bergaya WanGP (`mmgp`)."""

    profile_id: str
    display_name: str
    short_badge: str
    comfy_cli_flags: Tuple[str, ...]
    sequential_stage_flush: bool
    description: str


# CATATAN PENTING HASIL AUDIT 8 ENGINE:
# - JANGAN memasang `--fp16-unet` agar model native BF16 (Anima), INT8 ConvRot (MOSS, Qwen 2.1, Z-Image),
#   dan GGUF (Krea 2, Chroma, Flux-Fusion, Wan 2.1) berjalan di presisi aslinya tanpa overhead konversi dtype.
# - Gunakan `--reserve-vram 0.4` (memberikan +200 MB ekstra ruang VRAM usable untuk UNet dibanding 0.6).
# - Pertahankan `--fp16-vae` karena menghemat VRAM secara signifikan saat decoding gambar/video tanpa mengurangi kualitas.
WANGP_PROFILES: Dict[str, WanGPMemoryProfile] = {
    "profile_1_speed": WanGPMemoryProfile(
        profile_id="profile_1_speed",
        display_name="Profile 1: Speed / VRAM-Resident (Default Image Models)",
        short_badge="WanGP P1 · VRAM-Resident + C++ Guard",
        comfy_cli_flags=(
            "--reserve-vram",
            "0.4",
            "--fp16-vae",
            "--use-pytorch-cross-attention",
            "--preview-method",
            "none",
        ),
        sequential_stage_flush=False,
        description=(
            "Model DiT (3.9 GB - 7.9 GB) bertahan penuh di VRAM 15 GB T4 selama sampling "
            "tanpa konversi dtype paksa."
        ),
    ),
    "profile_2_lowram": WanGPMemoryProfile(
        profile_id="profile_2_lowram",
        display_name="Profile 2: Low-RAM Sequential Stage (Otomatis Wan 2.1 14B / Anti-OOM)",
        short_badge="WanGP P2 · Low-RAM + C++ Guard",
        comfy_cli_flags=(
            "--lowvram",
            "--reserve-vram",
            "0.4",
            "--fp16-vae",
            "--use-pytorch-cross-attention",
            "--preview-method",
            "none",
        ),
        sequential_stage_flush=True,
        description=(
            "Eksekusi bertahap ala WanGP: melakukan flush model & C++ malloc_trim(0) di antara "
            "Conditioning, Sampling, dan Tiled VAE Decode / Upscaling."
        ),
    ),
}


class WanGPMemoryManager:
    """
    Mengelola profil memori WanGP, Pre-Flight RAM Watchdog (< 1.2 GiB),
    C++ VFS Page-Cache Eviction, dan Engine Transition Guard saat pengguna berganti arsitektur model.
    """

    def __init__(
        self,
        default_profile_id: str = "profile_1_speed",
        min_available_ram_gib: float = PREFLIGHT_MIN_AVAILABLE_RAM_GIB,
    ) -> None:
        configure_cuda_alloc_env()
        self.active_profile_id: str = (
            default_profile_id if default_profile_id in WANGP_PROFILES else "profile_1_speed"
        )
        self.active_engine_id: Optional[str] = None
        self.min_available_ram_gib: float = min_available_ram_gib

    @property
    def active_profile(self) -> WanGPMemoryProfile:
        return WANGP_PROFILES[self.active_profile_id]

    def set_profile(self, profile_id_or_label: str) -> WanGPMemoryProfile:
        """Mengubah profil aktif berdasarkan ID atau label tampilan."""
        if profile_id_or_label in WANGP_PROFILES:
            self.active_profile_id = profile_id_or_label
            return self.active_profile
        for pid, prof in WANGP_PROFILES.items():
            if prof.display_name == profile_id_or_label or prof.short_badge == profile_id_or_label:
                self.active_profile_id = pid
                return prof
        if "2" in str(profile_id_or_label) or "low" in str(profile_id_or_label).lower():
            self.active_profile_id = "profile_2_lowram"
        else:
            self.active_profile_id = "profile_1_speed"
        return self.active_profile

    def get_comfy_launch_flags(self, engine_id: Optional[str] = None) -> List[str]:
        """Mengembalikan daftar CLI flags ComfyUI sesuai profil memori aktif."""
        if engine_id == "wan21":
            return list(WANGP_PROFILES["profile_2_lowram"].comfy_cli_flags)
        return list(self.active_profile.comfy_cli_flags)

    def check_preflight_ram(
        self,
        free_comfy_cb: Optional[Callable[[bool], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Pre-Flight System RAM Watchdog:
        Membaca `/proc/meminfo` (`MemAvailable`). Jika `< 1.2 GiB`, otomatis memanggil
        `free_comfy_cb(False)` + `trim_native_memory()` untuk mencegah Linux OOM Killer
        tanpa membuang model yang sedang hangat.
        """
        _, avail_before, total_gib = get_system_ram_metrics()
        if avail_before < self.min_available_ram_gib:
            if free_comfy_cb is not None:
                try:
                    free_comfy_cb(False)
                except Exception:
                    pass
            trim_native_memory()
            _, avail_after, _ = get_system_ram_metrics()
            msg = (
                f"[WanGP C++ Watchdog] System RAM rendah ({avail_before:.2f}/{total_gib:.1f} GiB < "
                f"{self.min_available_ram_gib:.1f} GiB). Otomatis mem-flush cache & "
                f"C++ malloc_trim(0) -> Tersedia: {avail_after:.2f} GiB."
            )
            print(msg, flush=True)
            return True, msg
        return False, f"System RAM aman ({avail_before:.2f}/{total_gib:.1f} GiB tersedia)."

    def enforce_engine_transition(
        self,
        requested_engine_id: str,
        free_comfy_cb: Optional[Callable[[bool], None]] = None,
        models_dir: Optional[Path] = None,
    ) -> Tuple[bool, str]:
        """
        Engine Transition Guard:
        - Jika `requested_engine_id != self.active_engine_id`, WAJIB memanggil
          `free_comfy_cb(True)` (`POST /free`) + `evict_models_page_cache()` + `trim_native_memory()`
          sebelum merender.
        - Jika sama, pertahankan model di VRAM/cache ComfyUI agar render ke-2 dst. instan.
        """
        if requested_engine_id == "wan21":
            self.active_profile_id = "profile_2_lowram"

        previous_engine = self.active_engine_id
        if previous_engine is not None and previous_engine != requested_engine_id:
            if free_comfy_cb is not None:
                try:
                    free_comfy_cb(True)
                except Exception:
                    pass
            if models_dir is not None:
                evict_models_page_cache(models_dir)
            trim_native_memory()
            self.active_engine_id = requested_engine_id
            msg = (
                f"[WanGP C++ Transition Guard] Transisi arsitektur '{previous_engine}' -> "
                f"'{requested_engine_id}': model lama di-unload, page cache & heap C++ dibersihkan."
            )
            print(msg, flush=True)
            return True, msg

        self.active_engine_id = requested_engine_id
        flushed_by_watchdog, watchdog_msg = self.check_preflight_ram(free_comfy_cb=free_comfy_cb)
        if flushed_by_watchdog:
            return True, watchdog_msg
        return False, f"Engine '{requested_engine_id}' tetap hangat di memori ({watchdog_msg})"