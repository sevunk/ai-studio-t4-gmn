// -*- coding: utf-8 -*-
// Unified Multi-Engine AI Studio (T4 Edition)
// Native C++ Linux Memory & Page-Cache Manager (`core/studio_native_mem.cpp`)
// Compiled automatically via: g++ -O3 -shared -fPIC studio_native_mem.cpp -o libstudio_mem.so

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <malloc.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

extern "C" {

// 1. Konfigurasi allocator glibc (ptmalloc2) agar alokasi tensor >= 128 KB
// selalu menggunakan mmap() langsung dan segera kembali ke kernel Linux saat di-free,
// serta membatasi jumlah arena heap agar tidak membengkak di multi-thread ComfyUI.
int studio_init_allocator(int mmap_threshold_kb, int trim_threshold_kb, int arena_max) {
    int ok = 1;
    if (mmap_threshold_kb > 0) {
        ok &= mallopt(M_MMAP_THRESHOLD, mmap_threshold_kb * 1024);
    }
    if (trim_threshold_kb >= 0) {
        ok &= mallopt(M_TRIM_THRESHOLD, trim_threshold_kb * 1024);
    }
    if (arena_max > 0) {
        ok &= mallopt(M_ARENA_MAX, arena_max);
    }
    return ok;
}

// 2. Pemangkasan agresif seluruh arena heap C++ dan pengembalian halaman bebas ke OS Linux.
int studio_aggressive_trim(size_t pad_bytes) {
    return malloc_trim(pad_bytes);
}

// 3. Pengosongan Linux VFS Page Cache (POSIX_FADV_DONTNEED) untuk file bobot model
// berukuran besar (5-10 GB) yang sudah di-unload agar tidak memenuhi kuota RAM cgroup Google Colab.
int studio_evict_file_cache(const char* filepath) {
    if (!filepath || filepath[0] == '\0') {
        return -1;
    }
    int fd = open(filepath, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -2;
    }
    struct stat st;
    int res = 0;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        res = posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
    }
    close(fd);
    return res;
}

// 4. Pembacaan cepat MemTotal & MemAvailable ( dalam KiB ) langsung dari /proc/meminfo
// menggunakan parser C++ tanpa overhead regex Python.
int studio_read_meminfo_kib(long* out_total_kib, long* out_avail_kib) {
    if (!out_total_kib || !out_avail_kib) {
        return -1;
    }
    *out_total_kib = 0;
    *out_avail_kib = 0;

    FILE* fp = fopen("/proc/meminfo", "r");
    if (!fp) {
        return -2;
    }

    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            *out_total_kib = strtol(line + 9, nullptr, 10);
            found |= 1;
        } else if (strncmp(line, "MemAvailable:", 13) == 0) {
            *out_avail_kib = strtol(line + 13, nullptr, 10);
            found |= 2;
        }
        if (found == 3) {
            break;
        }
    }
    fclose(fp);
    return (found == 3) ? 0 : -3;
}

} // extern "C"
