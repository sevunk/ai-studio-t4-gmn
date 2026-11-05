# -*- coding: utf-8 -*-
"""
Headless ComfyUI Daemon Manager & Cold-Load Resilient WebSocket Client (`core/backend.py`)
Mengintegrasikan:
- WanGP-style Memory Manager (`WanGPMemoryManager`, `trim_native_memory()`, Pre-Flight RAM Watchdog)
- Engine Transition Guard (`self.active_engine_id` -> otomatis `POST /free` + `malloc_trim(0)` saat ganti engine)
- WebSocket Client dengan socket timeout 4.0 detik yang tahan terhadap cold-load bobot 6–10 GB di disk Colab
- Auto-Heal Daemon: otomatis mendeteksi jika proses ComfyUI mati dan menyalakannya kembali dengan `self.start()`.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.catalog import BASE_DIR, COMFY_DIR
from core.memory_manager import (
    WanGPMemoryManager,
    get_gpu_vram_metrics,
    get_system_ram_metrics,
    trim_native_memory,
)


class HeadlessComfyManager:
    """Mengelola siklus hidup server Headless ComfyUI, memori WanGP, dan eksekusi workflow."""

    def __init__(
        self,
        comfy_dir: Path = COMFY_DIR,
        host: str = "127.0.0.1",
        port: int = 8188,
        memory_manager: Optional[WanGPMemoryManager] = None,
    ) -> None:
        self.comfy_dir = comfy_dir
        self.host = host
        self.port = port
        self.http_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.log_file = BASE_DIR / "comfyui_headless.log"
        self.pid_file = BASE_DIR / "comfyui_headless.pid"
        self.process: Optional[subprocess.Popen] = None
        self.memory_manager = memory_manager or WanGPMemoryManager()
        self.active_engine_id: Optional[str] = None
        self._lock = threading.Lock()

    def is_ready(self) -> bool:
        """Memeriksa apakah endpoint `/system_stats` ComfyUI merespons HTTP 200."""
        try:
            req = urllib.request.Request(f"{self.http_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def stop(self) -> None:
        """Menghentikan proses ComfyUI Headless dengan bersih."""
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                except Exception:
                    pass
                self.process = None

            if self.pid_file.exists():
                try:
                    old_pid = int(self.pid_file.read_text(encoding="utf-8").strip())
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(1.0)
                except Exception:
                    pass
                try:
                    self.pid_file.unlink(missing_ok=True)
                except Exception:
                    pass

            trim_native_memory()

    def start(self, engine_id: Optional[str] = None, timeout_sec: int = 240, force_restart: bool = False) -> None:
        """Menyalakan daemon Headless ComfyUI dengan flag profil memori WanGP."""
        if force_restart:
            self.stop()

        with self._lock:
            if self.is_ready():
                return

            if not (self.comfy_dir / "main.py").exists():
                raise RuntimeError(
                    f"ComfyUI belum ditemukan di {self.comfy_dir}. Jalankan bootstrap.py terlebih dahulu."
                )

            cli_flags = self.memory_manager.get_comfy_launch_flags(engine_id=engine_id)
            cmd = [
                sys.executable,
                "main.py",
                "--listen",
                self.host,
                "--port",
                str(self.port),
                *cli_flags,
                "--disable-auto-launch",
            ]
            print(f"[COMFYUI] Menyalakan Headless Daemon: {' '.join(cmd)}", flush=True)
            log_fh = open(self.log_file, "a", encoding="utf-8", errors="replace")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.comfy_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            except Exception:
                pass

            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                if self.is_ready():
                    print(
                        f"[COMFYUI] Headless Backend siap di {self.http_url} ({time.time() - t0:.1f}s)!",
                        flush=True,
                    )
                    return
                if self.process.poll() is not None:
                    tail = self.get_recent_logs(50)
                    raise RuntimeError(f"Daemon ComfyUI berhenti mendadak saat startup:\n{tail}")
                time.sleep(1.0)

            raise TimeoutError(
                f"ComfyUI melewati batas waktu startup {timeout_sec}s. Log:\n{self.get_recent_logs(50)}"
            )

    def restart(self, engine_id: Optional[str] = None) -> str:
        """Me-restart penuh server Headless ComfyUI dan mengosongkan RAM/VRAM."""
        self.stop()
        self.active_engine_id = None
        self.memory_manager.active_engine_id = None
        self.start(engine_id=engine_id, force_restart=False)
        return "Server Headless ComfyUI berhasil di-restart dan memori telah disegarkan."

    def get_recent_logs(self, lines: int = 40) -> str:
        """Membaca N baris terakhir dari file log daemon ComfyUI."""
        if not self.log_file.exists():
            return "File log ComfyUI belum tersedia."
        try:
            all_lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(all_lines[-lines:])
        except Exception as exc:
            return f"Gagal membaca log: {exc}"

    def interrupt(self) -> str:
        """Mengirim sinyal `POST /interrupt` ke ComfyUI untuk membatalkan generasi aktif."""
        try:
            req = urllib.request.Request(
                f"{self.http_url}/interrupt",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as _:
                pass
            return "Sinyal pembatalan (Interrupt) berhasil dikirim ke ComfyUI."
        except Exception as exc:
            return f"Gagal mengirim sinyal interrupt: {exc}"

    def free_memory(self, unload_models: bool = True) -> str:
        """
        Memanggil endpoint `POST /free` di ComfyUI lalu mengeksekusi `trim_native_memory()`
        (`gc.collect()`, `torch.cuda.empty_cache()`, `libc.malloc_trim(0)`).
        """
        status_note = ""
        if self.is_ready():
            try:
                payload = json.dumps(
                    {"unload_models": bool(unload_models), "free_memory": True}
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.http_url}/free",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as _:
                    pass
                status_note = "ComfyUI /free OK"
            except Exception as exc:
                status_note = f"ComfyUI /free warning: {exc}"
        else:
            status_note = "ComfyUI offline"

        trimmed = trim_native_memory()
        if unload_models:
            self.active_engine_id = None
            self.memory_manager.active_engine_id = None
        return f"RAM & VRAM dibersihkan ({status_note}, malloc_trim={'OK' if trimmed else 'N/A'})."

    def prepare_for_engine(self, requested_engine_id: str) -> Tuple[bool, str]:
        """
        Menjalankan Engine Transition Guard & Pre-Flight System RAM Watchdog
        sebelum mengirim workflow ke ComfyUI.
        """
        flushed, msg = self.memory_manager.enforce_engine_transition(
            requested_engine_id=requested_engine_id,
            free_comfy_cb=lambda unload: self.free_memory(unload_models=unload),
            models_dir=self.comfy_dir / "models",
        )
        self.active_engine_id = requested_engine_id
        return flushed, msg

    def get_samplers_and_schedulers(self) -> Tuple[List[str], List[str]]:
        """Mengambil daftar sampler_name dan scheduler dinamis dari /object_info/KSampler jika server aktif."""
        if not self.is_ready():
            return [], []
        try:
            req = urllib.request.Request(f"{self.http_url}/object_info/KSampler", method="GET")
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                req_in = data.get("KSampler", {}).get("input", {}).get("required", {})
                samplers = req_in.get("sampler_name", [[]])[0]
                schedulers = req_in.get("scheduler", [[]])[0]
                return (
                    [str(s) for s in samplers if isinstance(s, str)],
                    [str(s) for s in schedulers if isinstance(s, str)],
                )
        except Exception:
            return [], []

    def has_node(self, node_class_name: str) -> bool:
        """Memeriksa apakah custom node tertentu terdaftar di `/object_info/{node}`."""
        if not self.is_ready():
            return False
        try:
            req = urllib.request.Request(f"{self.http_url}/object_info/{node_class_name}", method="GET")
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return node_class_name in data
        except Exception:
            return False

    def get_telemetry_snapshot(self, current_engine_label: str = "Anima 2B") -> Dict[str, Any]:
        """Mengambil metrik real-time GPU VRAM, CPU System RAM, Status Backend, dan Profil WanGP."""
        ram_used, ram_avail, ram_total = get_system_ram_metrics()
        vram_used, _, vram_total, gpu_name = get_gpu_vram_metrics()
        online = self.is_ready()

        if online:
            try:
                req = urllib.request.Request(f"{self.http_url}/system_stats", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    devices = data.get("devices", [])
                    if devices:
                        dev = devices[0]
                        gpu_name = dev.get("name", gpu_name)
                        vt = dev.get("vram_total", 0) / (1024**3)
                        vf = dev.get("vram_free", 0) / (1024**3)
                        if vt > 0:
                            vram_total = vt
                            vram_used = max(0.0, vt - vf)
            except Exception:
                pass

        return {
            "online": online,
            "status_text": "ONLINE" if online else "STANDBY / OFFLINE",
            "gpu_name": gpu_name,
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 1),
            "vram_pct": min(100, int((vram_used / max(0.1, vram_total)) * 100)),
            "ram_used_gb": round(ram_used, 2),
            "ram_avail_gb": round(ram_avail, 2),
            "ram_total_gb": round(ram_total, 1),
            "ram_pct": min(100, int((ram_used / max(0.1, ram_total)) * 100)),
            "active_engine": current_engine_label,
            "wangp_profile": self.memory_manager.active_profile.short_badge,
        }

    def _fetch_history_entry(self, prompt_id: str) -> Dict[str, Any]:
        """Mengambil entri riwayat eksekusi dari `/history/{prompt_id}`."""
        try:
            with urllib.request.urlopen(f"{self.http_url}/history/{prompt_id}", timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get(prompt_id, {})
        except Exception:
            return {}

    def execute_workflow(
        self,
        engine_id: str,
        workflow: Dict[str, Any],
        progress_cb: Optional[Callable[[float, str], None]] = None,
        timeout_sec: int = 2400,
    ) -> Tuple[List[bytes], Dict[str, float]]:
        """
        Mengeksekusi workflow JSON ComfyUI dengan:
        1. Engine Transition Guard + Pre-Flight System RAM Watchdog
        2. Pemantauan WebSocket real-time (`ws.settimeout(4.0)` tahan cold-load 6-10 GB)
        3. Auto-Heal Daemon: jika proses ComfyUI mati karena OOM, otomatis `self.start()` ulang
        4. Sequential Flush (jika `Profile 2: Low-RAM Sequential Stage` aktif)
        Mengembalikan `(daftar_bytes_output_image_atau_frame, dict_waktu_tahap)`.
        """
        if not self.is_ready():
            self.start(engine_id=engine_id)

        # 1. Jalankan Engine Transition Guard & Pre-Flight System RAM Watchdog
        self.prepare_for_engine(engine_id)

        import websocket

        client_id = uuid.uuid4().hex
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={client_id}", timeout=30)
        ws.settimeout(4.0)  # Heartbeat pendek tahan cold-load disk Colab

        t_start = time.time()
        t_stage1_end = t_start
        t_stage2_end = t_start
        last_ratio = 0.04
        stage_desc = f"[{engine_id.upper()}] Memuat bobot model & menyandikan prompt..."

        try:
            payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.http_url}/prompt",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    prompt_id = resp_data["prompt_id"]
            except urllib.error.HTTPError as http_err:
                err_body = http_err.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ComfyUI menolak workflow (HTTP {http_err.code}): {err_body}")

            if progress_cb is not None:
                progress_cb(last_ratio, stage_desc)

            while True:
                elapsed_sec = time.time() - t_start
                if elapsed_sec > timeout_sec:
                    raise TimeoutError(f"Eksekusi melebihi batas waktu {timeout_sec} detik.")

                try:
                    raw_msg = ws.recv()
                except (websocket.WebSocketTimeoutException, TimeoutError, socket.timeout):
                    # Normal terjadi saat ComfyUI membaca bobot 6–10 GB dari disk Colab:
                    # 1. Cek apakah prompt_id sudah selesai di /history/{prompt_id}
                    hist_entry = self._fetch_history_entry(prompt_id)
                    if hist_entry and hist_entry.get("outputs"):
                        break

                    # 2. Cek apakah proses daemon ComfyUI mati karena OOM Linux
                    if self.process is not None and self.process.poll() is not None:
                        tail_log = self.get_recent_logs(30)
                        print("[AUTO-HEAL] Daemon ComfyUI terdeteksi berhenti! Menyalakan ulang...", flush=True)
                        trim_native_memory()
                        self.start(engine_id=engine_id, force_restart=True)
                        raise RuntimeError(
                            "Daemon ComfyUI sempat terhenti (kemungkinan kehabisan memori/OOM) dan telah "
                            f"dinyalakan ulang secara otomatis (Auto-Healed). Silakan klik Generate lagi.\nLog:\n{tail_log}"
                        )

                    # 3. Update progress bar Gradio dengan waktu tunggu berjalan (elapsed_sec)
                    if progress_cb is not None:
                        progress_cb(
                            last_ratio,
                            f"{stage_desc} (Cold-load / proses berjalan {int(elapsed_sec)} dtk...)",
                        )
                    continue

                if isinstance(raw_msg, bytes):
                    continue

                msg = json.loads(raw_msg)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break
                    node_id = str(data.get("node", ""))
                    node_cls = workflow.get(node_id, {}).get("class_type", "")
                    if "Encode" in node_cls or "Loader" in node_cls:
                        stage_desc = f"Tahap 1/3: Menjalankan {node_cls}..."
                        last_ratio = max(last_ratio, 0.10)
                    elif "Sampler" in node_cls:
                        t_stage1_end = time.time()
                        stage_desc = f"Tahap 2/3: Sampling Denoising ({node_cls})..."
                        last_ratio = max(last_ratio, 0.18)
                    elif "Decode" in node_cls or "Upscale" in node_cls or "Save" in node_cls:
                        if t_stage2_end <= t_stage1_end:
                            t_stage2_end = time.time()
                        stage_desc = f"Tahap 3/3: Decoding & Post-Processing ({node_cls})..."
                        last_ratio = max(last_ratio, 0.90)
                    if progress_cb is not None:
                        progress_cb(last_ratio, f"{stage_desc} ({int(elapsed_sec)} dtk)")

                elif msg_type == "progress" and data.get("prompt_id") == prompt_id:
                    val = float(data.get("value", 0))
                    max_val = float(data.get("max", 1))
                    if max_val > 0:
                        if t_stage1_end == t_start:
                            t_stage1_end = time.time()
                        last_ratio = min(0.89, 0.15 + 0.73 * (val / max_val))
                        stage_desc = f"Tahap 2/3: Sampling Step {int(val)}/{int(max_val)}"
                        if progress_cb is not None:
                            progress_cb(last_ratio, f"{stage_desc} ({int(elapsed_sec)} dtk)")

                elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                    exc_msg = data.get("exception_message", "Unknown error")
                    node_type = data.get("node_type", "UnknownNode")
                    raise RuntimeError(f"ComfyUI gagal pada node {node_type}: {exc_msg}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

        t_end = time.time()
        if t_stage1_end <= t_start:
            t_stage1_end = t_start + (t_end - t_start) * 0.15
        if t_stage2_end <= t_stage1_end:
            t_stage2_end = t_start + (t_end - t_start) * 0.85

        # Ambil seluruh gambar/frame yang dihasilkan oleh node SaveImage
        hist_record = self._fetch_history_entry(prompt_id)
        if not hist_record:
            raise RuntimeError("Eksekusi selesai tetapi output tidak ditemukan di /history ComfyUI.")

        status_info = hist_record.get("status", {})
        if status_info.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI melaporkan error: {status_info.get('messages', [])}")

        output_bytes_list: List[bytes] = []
        outputs = hist_record.get("outputs", {})
        for _, node_out in outputs.items():
            for img_meta in node_out.get("images", []):
                q = urllib.parse.urlencode(
                    {
                        "filename": img_meta["filename"],
                        "subfolder": img_meta.get("subfolder", ""),
                        "type": img_meta.get("type", "output"),
                    }
                )
                with urllib.request.urlopen(f"{self.http_url}/view?{q}", timeout=60) as view_resp:
                    output_bytes_list.append(view_resp.read())

        # Jika Profile 2 (Low-RAM Sequential Stage) aktif, bersihkan cache setelah render selesai
        if self.memory_manager.active_profile.sequential_stage_flush:
            self.free_memory(unload_models=False)

        timings = {
            "t_stage1": round(max(0.1, t_stage1_end - t_start), 2),
            "t_stage2": round(max(0.1, t_stage2_end - t_stage1_end), 2),
            "t_stage3": round(max(0.1, t_end - t_stage2_end), 2),
            "total_elapsed_sec": round(max(0.1, t_end - t_start), 2),
        }
        return output_bytes_list, timings
