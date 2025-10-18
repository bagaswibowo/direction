from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any

import streamlit as st

# Reuse logic from Flask app
from app import (
    Point,
    Grid,
    build_nearest_chain,
    find_common_solvable_layout,
)


def draw_grid_image(
    grid: Grid,
    start: Point,
    targets: List[Point],
    path: List[Point] | None = None,
    visited: set[Tuple[int, int]] | None = None,
    scale: int = 16,
):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        st.error("Pillow belum terpasang. Tambahkan 'Pillow' di requirements.txt.")
        return None

    if grid and len(grid) > 0 and len(grid[0]) > 0:
        rows, cols = len(grid), len(grid[0])
    else:
        rows, cols = 0, 0

    img = Image.new("RGB", (cols * scale, rows * scale), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # cells
    for y in range(rows):
        for x in range(cols):
            color = (34, 34, 34) if grid[y][x] else (238, 238, 238)
            d.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=color, outline=(180, 180, 180))
    # visited
    if visited:
        for (x, y) in visited:
            d.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=(144, 202, 249))
    # path (current step highlight)
    if path:
        for p in path:
            d.rectangle([p.x * scale, p.y * scale, (p.x + 1) * scale - 1, (p.y + 1) * scale - 1], fill=(255, 235, 59))
    # start/targets
    d.rectangle([start.x * scale, start.y * scale, (start.x + 1) * scale - 1, (start.y + 1) * scale - 1], fill=(76, 175, 80))
    for t in targets:
        d.rectangle([t.x * scale, t.y * scale, (t.x + 1) * scale - 1, (t.y + 1) * scale - 1], fill=(244, 67, 54))
    return img


def render_stats(label: str, res: Dict):
    if not res or not res.get("ok"):
        st.markdown(f"**{label}**: Tidak ditemukan jalur.")
        return
    ms = res.get("durationMs", 0)
    expanded = res.get("expanded", 0)
    total = res.get("totalLength", 0)
    st.markdown(
        f"**{label}** — Waktu: {ms:.3f} ms • Node dikembangkan: {expanded} • Panjang jalur: {total} • Berhasil: Ya"
    )


st.set_page_config(page_title="Perbandingan A*, BFS, Dijkstra", layout="wide")
st.title("Simulasi Pathfinding — A*, BFS, Dijkstra")

with st.sidebar:
    st.header("Pengaturan")
    rows = st.slider("Jumlah Baris", 10, 100, 100)
    cols = st.slider("Jumlah Kolom", 10, 100, 100)
    wall_prob = st.slider("Kerapatan Dinding", 0.05, 0.6, 0.35, 0.01)
    level = st.slider("Level (jumlah target)", 1, 100, 60)
    animate = st.checkbox("Animasi jalur", value=True)
    speed_ms = st.slider("Kecepatan animasi (ms)", 5, 150, 30, 5)

st.caption("Tip: Atur level untuk menambah jumlah target.")
if level >= 30:
    st.info("Level tinggi (>=30) dapat memerlukan waktu lebih lama untuk membangkitkan layout dan menjalankan simulasi.")

# Buat layout yang bisa diselesaikan ketiga algoritma
grid, start, targets = find_common_solvable_layout(rows, cols, wall_prob, level, ["astar", "bfs", "dijkstra"])

colA, colB, colC = st.columns(3)


def run_algo(algo: str) -> Dict:
    stats: Dict = {}
    t0 = time.perf_counter()
    segs, _ordered = build_nearest_chain(start, targets, rows, cols, grid, algo, stats)
    t1 = time.perf_counter()
    total_len = sum(len(s) for s in segs) if segs else 0
    return {
        "algorithm": algo,
        "ok": bool(segs) and len(segs) == len(targets),
        "durationMs": round((t1 - t0) * 1000, 3),
        "expanded": stats.get("expanded", 0),
        "totalLength": total_len,
        "segments": segs or [],
    }


with st.spinner("Menjalankan algoritma..."):
    res_astar = run_algo("astar")
    res_bfs = run_algo("bfs")
    res_dij = run_algo("dijkstra")


def concurrent_animate(col_label_pairs: List[Tuple[Any, str, Dict]]):
    # Render headers
    placeholders = []
    visited_map: Dict[str, set[Tuple[int, int]]] = {}
    paths_map: Dict[str, List[Point]] = {}

    for col, label, res in col_label_pairs:
        col.subheader(label)
        placeholders.append((label, col.empty()))
        # flatten segments into a single list of Points
        if res.get("ok") and res.get("segments"):
            flat: List[Point] = []
            for seg in res["segments"]:
                flat.extend(seg)
            paths_map[label] = flat
            visited_map[label] = set([(start.x, start.y)])
        else:
            paths_map[label] = []
            visited_map[label] = set()

    if not animate:
        # Just render static grids and stats
        for (label, ph), (_, _, res) in zip(placeholders, col_label_pairs):
            img = draw_grid_image(grid, start, targets)
            if img is not None:
                ph.image(img, use_column_width=True)
            render_stats(label, res)
        return

    # Determine the maximum steps among all algorithms
    max_steps = max((len(p) for p in paths_map.values()), default=0)
    # Perform lockstep animation
    for step_idx in range(max_steps):
        for (label, ph) in placeholders:
            res = next((r for (_c, _l, r) in col_label_pairs if _l == label), None)
            if res is None:
                continue
            path = paths_map.get(label, [])
            if step_idx < len(path):
                p = path[step_idx]
                visited = visited_map[label]
                visited.add((p.x, p.y))
                img = draw_grid_image(grid, start, targets, [p], visited)
                if img is not None:
                    ph.image(img, use_column_width=True)
        time.sleep(max(0.001, speed_ms / 1000.0))

    # After loop, render final stats below each
    for (label, _ph), (_, _, res) in zip(placeholders, col_label_pairs):
        render_stats(label, res)


concurrent_animate([
    (colA, "A*", res_astar),
    (colB, "BFS", res_bfs),
    (colC, "Dijkstra", res_dij),
])

# Highlight pemenang
ok_res = [r for r in [res_astar, res_bfs, res_dij] if r["ok"]]
if ok_res:
    winner = min(ok_res, key=lambda r: r["durationMs"])
    st.success(f"Pemenang: {winner['algorithm'].upper()} ({winner['durationMs']} ms)")
else:
    st.warning("Tidak ada jalur yang ditemukan oleh ketiga algoritma.")
