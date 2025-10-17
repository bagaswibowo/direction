# Simulasi Algoritma Pencarian Jalur (A*/BFS/Dijkstra)

Simulasi pencarian jalur terpendek di labirin acak dengan progresi level. Setiap level: labirin jadi lebih sulit (lebih banyak dinding) dan jumlah target bertambah (1, 2, 3, ...). Algoritma menghubungkan titik terdekat terlebih dahulu hingga terjauh. UI dipertahankan seperti semula dengan tambahan kontrol kecil.

## Cara Menjalankan

1. Aktifkan virtualenv dan install dependensi
2. Jalankan server Python
3. Buka browser ke `http://127.0.0.1:5000`

### Perintah (macOS, zsh)

```bash
# 1) (opsional) buat venv
python3 -m venv .venv
source .venv/bin/activate

# 2) install dependensi
pip install -r requirements.txt

# 3) jalankan server
python app.py
```

### Mode Streamlit (UI 3 simulasi berdampingan)

```bash
# 1) aktifkan venv (lihat di atas), lalu install dependensi
pip install -r requirements.txt

# 2) jalankan streamlit
streamlit run streamlit_app.py
```

Jika berjalan di Streamlit Community Cloud:
- Pastikan file `streamlit_app.py` ada di root repo.
- Pastikan `requirements.txt` berisi `streamlit` dan `Pillow`.
- Saat membuat app baru, pilih file utama: `streamlit_app.py`.

## Kontrol di UI

- Generate Labirin: regenerasi level sekarang dengan labirin baru.
- Mulai Simulasi: mulai otomatis dari level 1 dan naik terus.
- Pause/Resume: jeda/lanjutkan animasi.
- Algoritma: pilih A*, BFS, atau Dijkstra. Dipakai saat generate/next berikutnya.
- Kecepatan: atur kecepatan animasi (ms per langkah).
- Simpan Hasil: menulis ringkasan run ke `runs.jsonl`.

## Detail Teknis

- Backend Flask menghasilkan grid yang bisa diselesaikan (nearest-first chain) menggunakan algoritma yang dipilih.
- Algoritma tersedia: A* (heuristik Manhattan), BFS, dan Dijkstra.
- Metrik ditampilkan: total panjang jalur gabungan semua segmen.

## Catatan

- Server berjalan untuk pengembangan (debug). Jangan gunakan langsung untuk produksi.
- Ukuran grid default 20x20; ubah di `State` pada `app.py` bila perlu.
 - Untuk Streamlit, `app.py` tidak akan menyalakan server Flask secara otomatis (agar tidak konflik dengan runtime Streamlit).
