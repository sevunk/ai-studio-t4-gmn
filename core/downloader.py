# -*- coding: utf-8 -*-
"""
Universal Multi-Connection Downloader (`aria2c` 16-connection + Smart URL Resolver)
Mendukung:
1. Hugging Face (`huggingface.co` / `hf.co`) & ModelScope (`modelscope.cn` / `modelscope.ai`)
   - Auto-rewrite `/blob/`, `/raw/`, `/file/view/` -> `/resolve/`
   - Header `Authorization: Bearer {TOKEN}`
   - Fallback otomatis ke `huggingface_hub.hf_hub_download` (Xet-compatible) jika `aria2c` gagal
   - Git LFS (`git clone --depth 1`) untuk URL repositori penuh
2. Civitai (`civitai.com` & `civitai.red`)
   - Menerima `/api/download/models/xxxx`, `?modelVersionId=xxxx`, atau `/models/xxxx`
   - Pre-flight request (`allow_redirects=False`) dengan `CIVITAI_TOKEN` untuk mendapatkan
     direct pre-signed URL Cloudflare R2/S3 dan nama file dari `Content-Disposition`
3. Custom Rename (`url|nama_kustom.safetensors`) & Anti-HTML Guard (`< 500 KB` HTML check).
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests

from config.catalog import (
    BASE_DIR,
    ENGINE_CATALOG,
    MODELS_DIR,
    UPSCALER_CATALOG,
    ModelFileSpec,
    check_file_installed,
    ensure_model_directories,
)

MODEL_EXTENSIONS = (".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".bin")


class UniversalDownloader:
    """Pengunduh model multi-koneksi berbasis aria2c dengan fallback hf_hub_download & Anti-HTML Guard."""

    def __init__(
        self,
        models_root: Path = MODELS_DIR,
        hf_token: str = "",
        civitai_token: str = "",
        modelscope_token: str = "",
    ) -> None:
        self.models_root = models_root
        ensure_model_directories(self.models_root)
        self.hf_token = (hf_token or os.environ.get("HF_TOKEN", "")).strip()
        self.civitai_token = (civitai_token or os.environ.get("CIVITAI_TOKEN", "")).strip()
        self.modelscope_token = (modelscope_token or os.environ.get("MODELSCOPE_TOKEN", "")).strip()

    def _run_cmd(
        self,
        cmd: List[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        printable = " ".join(map(str, cmd))
        for tok in (self.hf_token, self.civitai_token, self.modelscope_token):
            if tok:
                printable = printable.replace(tok, "***TOKEN***")
        print(f"$ {printable}", flush=True)
        return subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            env=env,
            check=check,
            text=True,
        )

    @staticmethod
    def parse_entries(raw_str: str) -> List[str]:
        """Memisahkan daftar URL yang dipisahkan koma, titik koma, atau baris baru."""
        if not raw_str or not raw_str.strip():
            return []
        return [item.strip() for item in re.split(r"[,\n;]+", raw_str) if item.strip()]

    @staticmethod
    def validate_anti_html(filepath: Path, source_url: str = "") -> None:
        """
        Validasi Anti-HTML:
        Jika file hasil unduhan berukuran < 500 KB dan 512 byte pertamanya mengandung
        `<!doctype html` atau `<html` (halaman login/error), hapus file tersebut dan
        lemparkan RuntimeError yang jelas bahwa Token diperlukan.
        """
        if not filepath.is_file():
            raise RuntimeError(f"File hasil unduhan tidak ditemukan: {filepath}")

        fsize = filepath.stat().st_size
        if fsize < 500_000:
            head_bytes = filepath.read_bytes()[:512].lower()
            if (
                b"<!doctype html" in head_bytes
                or b"<html" in head_bytes
                or b"unauthorized" in head_bytes
                or b"access denied" in head_bytes
            ):
                filepath.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Unduhan gagal untuk '{source_url or filepath.name}': Server mengembalikan halaman "
                    "HTML/Login (bukan bobot model). Pastikan HF_TOKEN / CIVITAI_TOKEN / MODELSCOPE_TOKEN "
                    "yang valid sudah diisi untuk model gated/early-access."
                )

    def resolve_civitai_direct_url(self, url_part: str) -> Tuple[str, Optional[str]]:
        """
        Menangani URL Civitai (`civitai.com` & `civitai.red`):
        - Menerima `/api/download/models/xxxx`, `?modelVersionId=xxxx`, atau `/models/xxxx`
        - Melakukan pre-flight request dengan `allow_redirects=False` menggunakan `CIVITAI_TOKEN`
          untuk mendapatkan direct pre-signed URL Cloudflare R2/S3 dan nama file asli dari
          header `Content-Disposition` sebelum diserahkan ke `aria2c`.
        """
        parsed = urlparse(url_part)
        host = parsed.netloc.lower() or "civitai.com"
        qs = parse_qs(parsed.query)

        if "/models/" in parsed.path and "/api/download/" not in parsed.path:
            if "modelVersionId" in qs and qs["modelVersionId"]:
                ver_id = qs["modelVersionId"][0]
                api_url = f"https://{host}/api/download/models/{ver_id}"
            else:
                m = re.search(r"/models/(\d+)", parsed.path)
                if not m:
                    raise ValueError(f"URL Civitai tidak memiliki ID model yang valid: {url_part}")
                model_id = m.group(1)
                info_resp = requests.get(f"https://{host}/api/v1/models/{model_id}", timeout=20)
                info_resp.raise_for_status()
                versions = info_resp.json().get("modelVersions", [])
                if not versions:
                    raise ValueError(f"Tidak ditemukan modelVersions pada {url_part}")
                ver_id = versions[0]["id"]
                api_url = f"https://{host}/api/download/models/{ver_id}"
        else:
            api_url = url_part

        if self.civitai_token and "token=" not in api_url:
            sep = "&" if "?" in api_url else "?"
            api_url = f"{api_url}{sep}token={self.civitai_token}"

        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AIStudioT4/1.0"}
        if self.civitai_token:
            headers["Authorization"] = f"Bearer {self.civitai_token}"

        resp = requests.get(api_url, headers=headers, stream=True, allow_redirects=False, timeout=25)
        try:
            if resp.status_code in (301, 302, 303, 307, 308) and "Location" in resp.headers:
                direct_url = resp.headers["Location"]
            else:
                direct_url = api_url

            detected_filename: Optional[str] = None
            cd = resp.headers.get("Content-Disposition", "")
            m_cd = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", cd, flags=re.IGNORECASE)
            if m_cd:
                detected_filename = unquote(m_cd.group(1).strip())

            if not detected_filename:
                redir_qs = parse_qs(urlparse(direct_url).query)
                for key in ("response-content-disposition", "filename"):
                    if key in redir_qs and redir_qs[key]:
                        m_q = re.search(
                            r"filename=[\"']?([^\"';\r\n]+)",
                            unquote(redir_qs[key][0]),
                            flags=re.IGNORECASE,
                        )
                        if m_q:
                            detected_filename = m_q.group(1).strip()
                            break

            if not detected_filename:
                clean_p = Path(unquote(urlparse(direct_url).path)).name
                if any(clean_p.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                    detected_filename = clean_p
        finally:
            resp.close()

        return direct_url, detected_filename

    def _try_hf_hub_fallback(self, download_url: str, final_path: Path) -> bool:
        """
        Jika URL berasal dari Hugging Face (`huggingface.co/{org}/{repo}/resolve/{rev}/{path}`)
        dan `aria2c` gagal (mis. karena redirect Xet/CDN), unduh menggunakan `hf_hub_download`.
        """
        m = re.match(
            r"^https?://(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$",
            download_url.split("?")[0],
        )
        if not m:
            return False
        repo_id, revision, filename_in_repo = m.group(1), m.group(2), m.group(3)
        try:
            from huggingface_hub import hf_hub_download

            print(
                f"[FALLBACK] Mengunduh via huggingface_hub ({repo_id} -> {filename_in_repo})...",
                flush=True,
            )
            dl_cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename_in_repo,
                revision=revision,
                token=self.hf_token or None,
            )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dl_cached, final_path)
            return final_path.is_file()
        except Exception as exc:
            print(f"[WARN] Fallback hf_hub_download gagal: {exc}", flush=True)
            return False

    def _download_with_aria2c(
        self,
        download_url: str,
        target_dir: Path,
        filename: str,
        extra_headers: Optional[List[str]] = None,
        min_bytes: int = 0,
    ) -> Path:
        """Menjalankan aria2c 16-koneksi dengan fallback otomatis ke hf_hub_download & requests streaming."""
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / filename

        aria_ok = False
        if shutil.which("aria2c"):
            aria_cmd = [
                "aria2c",
                "--console-log-level=warn",
                "--summary-interval=5",
                "-c",
                "-x",
                "16",
                "-s",
                "16",
                "-k",
                "1M",
                "--max-tries=5",
                "--retry-wait=3",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--content-disposition-default-utf8=true",
                "-d",
                str(target_dir),
                "-o",
                filename,
            ]
            if extra_headers:
                aria_cmd.extend(extra_headers)
            aria_cmd.append(download_url)
            res = self._run_cmd(aria_cmd, check=False)
            aria_ok = res.returncode == 0 and final_path.is_file()

        if not aria_ok:
            if not self._try_hf_hub_fallback(download_url, final_path):
                print(f"[FALLBACK] Mengunduh {filename} via requests stream...", flush=True)
                req_headers: Dict[str, str] = {"User-Agent": "Mozilla/5.0 AIStudioT4/1.0"}
                for h in extra_headers or []:
                    if h.startswith("--header=") and ":" in h:
                        kv = h.removeprefix("--header=")
                        k, v = kv.split(":", 1)
                        req_headers[k.strip()] = v.strip()
                with requests.get(download_url, headers=req_headers, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(final_path, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)

        self.validate_anti_html(final_path, source_url=download_url)

        if min_bytes > 0 and final_path.stat().st_size < min_bytes:
            actual = final_path.stat().st_size
            raise RuntimeError(
                f"File '{filename}' belum lengkap ({actual:,} bytes < minimal {min_bytes:,} bytes). "
                "Silakan ulangi unduhan untuk melanjutkan (resume)."
            )
        return final_path

    def download_spec(self, spec: ModelFileSpec) -> Path:
        """
        Mengunduh satu `ModelFileSpec` jika belum ada di disk atau ukurannya < `spec.min_bytes`.
        Mendukung *shared files* lintas folder/engine secara otomatis.
        """
        already_ok, existing_path = check_file_installed(spec, self.models_root)
        if already_ok and existing_path is not None:
            size_gb = existing_path.stat().st_size / (1024**3)
            print(
                f"[CACHE] {spec.filename} sudah tersedia ({size_gb:.2f} GB) di {existing_path}.",
                flush=True,
            )
            return existing_path

        target_dir = self.models_root / spec.subdir
        print(f"⬇️ Mengunduh [{spec.description or spec.filename}] -> {spec.rel_path} ...", flush=True)
        extra_headers: List[str] = []
        if ("huggingface.co" in spec.url or "hf.co" in spec.url) and self.hf_token:
            extra_headers.append(f"--header=Authorization: Bearer {self.hf_token}")

        path = self._download_with_aria2c(
            download_url=spec.url,
            target_dir=target_dir,
            filename=spec.filename,
            extra_headers=extra_headers,
            min_bytes=spec.min_bytes,
        )
        size_mb = path.stat().st_size / (1024**2)
        print(f"✅ Selesai: {path.name} ({size_mb:,.1f} MB)", flush=True)
        return path

    def download_engine(self, engine_id: str) -> List[Path]:
        """Mengunduh seluruh file yang dibutuhkan oleh `engine_id`."""
        if engine_id not in ENGINE_CATALOG:
            raise KeyError(f"Engine '{engine_id}' tidak ditemukan di katalog.")
        specs: List[ModelFileSpec] = ENGINE_CATALOG[engine_id]["files"]  # type: ignore[assignment]
        downloaded_paths: List[Path] = []
        for spec in specs:
            downloaded_paths.append(self.download_spec(spec))
        return downloaded_paths

    def download_upscaler(self, upscaler_name: str) -> Path:
        """Mengunduh model upscaler berdasarkan namanya di UPSCALER_CATALOG."""
        if upscaler_name not in UPSCALER_CATALOG:
            raise KeyError(f"Upscaler '{upscaler_name}' tidak terdaftar di UPSCALER_CATALOG.")
        return self.download_spec(UPSCALER_CATALOG[upscaler_name])

    def download_custom_url(self, entry: str, target_subdir: str = "loras") -> List[Path]:
        """
        Mengunduh URL kustom dari HuggingFace, ModelScope, Civitai (.com/.red), atau Git LFS repo.
        Mendukung sintaks `https://link-model...|nama_kustom.safetensors`.
        """
        target_dir = self.models_root / target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        custom_filename: Optional[str] = None
        if "|" in entry:
            url_part, custom_filename = [x.strip() for x in entry.split("|", 1)]
        else:
            url_part = entry.strip()

        parsed = urlparse(url_part.removeprefix("git+"))
        host = parsed.netloc.lower()
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        # 1. Deteksi apakah URL adalah repositori Git LFS penuh
        is_git_repo = False
        clone_url = url_part.removeprefix("git+")
        if url_part.startswith("git+") or url_part.endswith(".git"):
            is_git_repo = True
        elif (
            ("huggingface.co" in host or "hf.co" in host)
            and len(path_parts) == 2
            and not any(url_part.lower().endswith(ext) for ext in MODEL_EXTENSIONS)
        ):
            is_git_repo = True
            clone_url = f"https://huggingface.co/{path_parts[0]}/{path_parts[1]}.git"
        elif (
            ("modelscope.cn" in host or "modelscope.ai" in host)
            and len(path_parts) == 3
            and path_parts[0] == "models"
        ):
            is_git_repo = True
            clone_url = f"https://{host}/{path_parts[1]}/{path_parts[2]}.git"

        if is_git_repo:
            auth_clone_url = clone_url
            if ("huggingface.co" in host or "hf.co" in host) and self.hf_token:
                auth_clone_url = clone_url.replace("https://", f"https://user:{self.hf_token}@")
            elif ("modelscope.cn" in host or "modelscope.ai" in host) and self.modelscope_token:
                auth_clone_url = clone_url.replace("https://", f"https://oauth2:{self.modelscope_token}@")

            repo_name = Path(parsed.path.rstrip("/")).stem or f"repo_{int(time.time())}"
            tmp_dir = BASE_DIR / "tmp_git_lfs_clone" / repo_name
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.parent.mkdir(parents=True, exist_ok=True)

            print(f"🔄 Meng-clone repositori via Git LFS: {clone_url} ...", flush=True)
            env = os.environ.copy()
            env["GIT_LFS_SKIP_SMUDGE"] = "0"
            self._run_cmd(["git", "clone", "--depth", "1", auth_clone_url, str(tmp_dir)], env=env)

            moved_files: List[Path] = []
            for ext in ("*.safetensors", "*.gguf", "*.pt", "*.pth", "*.ckpt", "*.bin"):
                for f in tmp_dir.rglob(ext):
                    if f.is_file() and f.stat().st_size > 10_000:
                        dst_name = (
                            custom_filename
                            if (custom_filename and len(moved_files) == 0)
                            else f.name
                        )
                        dst = target_dir / dst_name
                        shutil.move(str(f), str(dst))
                        self.validate_anti_html(dst, source_url=clone_url)
                        moved_files.append(dst)
                        print(f"  ✔️ Dipindahkan: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not moved_files:
                raise RuntimeError(
                    f"Repositori {clone_url} berhasil di-clone, tetapi tidak ditemukan file model di dalamnya."
                )
            return moved_files

        # 2. Direct file download (Hugging Face / ModelScope / Civitai)
        download_url = url_part
        extra_headers: List[str] = []

        if "civitai.com" in host or "civitai.red" in host:
            download_url, detected_name = self.resolve_civitai_direct_url(url_part)
            if not custom_filename and detected_name:
                custom_filename = detected_name
        elif "huggingface.co" in host or "hf.co" in host:
            download_url = re.sub(r"/(blob|raw)/", "/resolve/", download_url)
            if self.hf_token:
                extra_headers.append(f"--header=Authorization: Bearer {self.hf_token}")
        elif "modelscope.cn" in host or "modelscope.ai" in host:
            download_url = re.sub(r"/(blob|file/view)/", "/resolve/", download_url)
            if self.modelscope_token:
                extra_headers.append(f"--header=Authorization: Bearer {self.modelscope_token}")

        if not custom_filename:
            candidate = Path(unquote(urlparse(download_url).path)).name
            if any(candidate.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                custom_filename = candidate
            else:
                custom_filename = f"custom_model_{int(time.time())}.safetensors"

        custom_filename = re.sub(r"[^A-Za-z0-9._-]", "_", custom_filename)
        if not any(custom_filename.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
            custom_filename += ".safetensors"

        out_path = self._download_with_aria2c(
            download_url=download_url,
            target_dir=target_dir,
            filename=custom_filename,
            extra_headers=extra_headers,
            min_bytes=0,
        )
        return [out_path]
