from __future__ import annotations

import random
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory


@dataclass
class Point:
    x: int
    y: int


Grid = List[List[int]]


def heuristic(a: Point, b: Point) -> int:
    # Manhattan distance
    return abs(a.x - b.x) + abs(a.y - b.y)


def neighbors(node: Point, rows: int, cols: int, grid: Grid) -> List[Point]:
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    res: List[Point] = []
    for dx, dy in dirs:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
            res.append(Point(nx, ny))
    return res


def a_star(start: Point, goal: Point, rows: int, cols: int, grid: Grid, stats: Optional[Dict] = None) -> List[Point]:
    import heapq
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    inf = 10 ** 9
    g_score = [[inf for _ in range(cols)] for _ in range(rows)]
    f_score = [[inf for _ in range(cols)] for _ in range(rows)]
    g_score[start.y][start.x] = 0
    f_score[start.y][start.x] = heuristic(start, goal)
    # heap entries: (f, x, y)
    open_heap: List[Tuple[int, int, int]] = [(f_score[start.y][start.x], start.x, start.y)]
    while open_heap:
        fcur, cx, cy = heapq.heappop(open_heap)
        if stats is not None:
            stats["expanded"] = stats.get("expanded", 0) + 1
        if fcur != f_score[cy][cx]:
            continue
        if cx == goal.x and cy == goal.y:
            chain: List[Tuple[int, int]] = [(cx, cy)]
            cur2 = (cx, cy)
            while cur2 in came_from:
                cur2 = came_from[cur2]
                chain.append(cur2)
            chain.reverse()
            return [Point(x, y) for (x, y) in chain[1:]]
        current = Point(cx, cy)
        for n in neighbors(current, rows, cols, grid):
            tentative = g_score[cy][cx] + 1
            if tentative < g_score[n.y][n.x]:
                came_from[(n.x, n.y)] = (cx, cy)
                g_score[n.y][n.x] = tentative
                f_score[n.y][n.x] = tentative + heuristic(n, goal)
                heapq.heappush(open_heap, (f_score[n.y][n.x], n.x, n.y))
    return []


def bfs(start: Point, goal: Point, rows: int, cols: int, grid: Grid, stats: Optional[Dict] = None) -> List[Point]:
    from collections import deque
    q = deque([start])
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    seen[start.y][start.x] = True
    while q:
        cur = q.popleft()
        if stats is not None:
            stats["expanded"] = stats.get("expanded", 0) + 1
        if cur.x == goal.x and cur.y == goal.y:
            chain: List[Tuple[int, int]] = [(cur.x, cur.y)]
            cur2 = (cur.x, cur.y)
            while cur2 in came_from:
                cur2 = came_from[cur2]
                chain.append(cur2)
            chain.reverse()
            return [Point(x, y) for (x, y) in chain[1:]]
        for n in neighbors(cur, rows, cols, grid):
            if not seen[n.y][n.x]:
                seen[n.y][n.x] = True
                came_from[(n.x, n.y)] = (cur.x, cur.y)
                q.append(n)
    return []


def dijkstra(start: Point, goal: Point, rows: int, cols: int, grid: Grid, stats: Optional[Dict] = None) -> List[Point]:
    import heapq
    # Heap holds only primitives: (distance, x, y)
    pq: List[Tuple[int, int, int]] = []
    heapq.heappush(pq, (0, start.x, start.y))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    inf = 10 ** 9
    dist = [[inf for _ in range(cols)] for _ in range(rows)]
    dist[start.y][start.x] = 0
    while pq:
        d, cx, cy = heapq.heappop(pq)
        if stats is not None:
            stats["expanded"] = stats.get("expanded", 0) + 1
        if d != dist[cy][cx]:
            continue
        if cx == goal.x and cy == goal.y:
            chain: List[Tuple[int, int]] = [(cx, cy)]
            cur2 = (cx, cy)
            while cur2 in came_from:
                cur2 = came_from[cur2]
                chain.append(cur2)
            chain.reverse()
            return [Point(x, y) for (x, y) in chain[1:]]
        cur_point = Point(cx, cy)
        for n in neighbors(cur_point, rows, cols, grid):
            nd = d + 1
            if nd < dist[n.y][n.x]:
                dist[n.y][n.x] = nd
                came_from[(n.x, n.y)] = (cx, cy)
                heapq.heappush(pq, (nd, n.x, n.y))
    return []


def find_path(start: Point, goal: Point, rows: int, cols: int, grid: Grid, algorithm: str, stats: Optional[Dict] = None) -> List[Point]:
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
    algo = (algorithm or "astar").lower()
    if algo == "bfs":
        return bfs(start, goal, rows, cols, grid, stats)
    if algo == "dijkstra":
        return dijkstra(start, goal, rows, cols, grid, stats)
    return a_star(start, goal, rows, cols, grid, stats)


def generate_grid(rows: int, cols: int, wall_prob: float) -> Grid:
    return [[1 if random.random() < wall_prob else 0 for _ in range(cols)] for _ in range(rows)]


def random_free_cell(rows: int, cols: int, grid: Grid, exclude: List[Tuple[int, int]]) -> Optional[Point]:
    free: List[Tuple[int, int]] = [
        (x, y)
        for y in range(rows)
        for x in range(cols)
        if grid[y][x] == 0 and (x, y) not in exclude
    ]
    if not free:
        return None
    x, y = random.choice(free)
    return Point(x, y)


def build_nearest_chain(start: Point, targets: List[Point], rows: int, cols: int, grid: Grid, algorithm: str, stats: Optional[Dict] = None) -> Tuple[List[List[Point]], List[Point]]:
    # Returns segments and the ordered targets
    current = start
    remaining = targets[:]
    ordered: List[Point] = []
    segments: List[List[Point]] = []

    while remaining:
        best_idx = -1
        best_len = 10 ** 9
        best_path: List[Point] = []
        for idx, t in enumerate(remaining):
            path = find_path(current, t, rows, cols, grid, algorithm, stats)
            if path and len(path) < best_len:
                best_len = len(path)
                best_idx = idx
                best_path = path
        if best_idx == -1:
            # no reachable remaining target
            return [], []
        segments.append(best_path)
        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        current = chosen

    return segments, ordered


def find_solvable_layout(rows: int, cols: int, wall_prob: float, n_targets: int, algorithm: str, max_attempts: int = 200) -> Tuple[Grid, Point, List[Point], List[List[Point]], List[Point]]:
    # Try until we get a grid with a valid nearest-first chain connecting all targets
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        grid = generate_grid(rows, cols, wall_prob)
        start = Point(0, 0)
        grid[start.y][start.x] = 0

        # Choose targets on free cells, avoiding duplicates
        targets: List[Point] = []
        exclude = {(start.x, start.y)}
        ok = True
        for _ in range(n_targets):
            p = random_free_cell(rows, cols, grid, list(exclude))
            if not p:
                ok = False
                break
            targets.append(p)
            grid[p.y][p.x] = 0
            exclude.add((p.x, p.y))
        if not ok:
            continue

        segments, ordered = build_nearest_chain(start, targets, rows, cols, grid, algorithm, None)
        if segments and len(segments) == len(targets):
            return grid, start, targets, segments, ordered
    # If we fail, lower wall probability and try again recursively once
    if wall_prob > 0.15:
        return find_solvable_layout(rows, cols, max(0.15, wall_prob - 0.05), n_targets, algorithm, max_attempts)
    # Give up with an empty solution; frontend should handle gracefully
    empty = [[0 for _ in range(cols)] for _ in range(rows)]
    s = Point(0, 0)
    return empty, s, [], [], []


def find_common_solvable_layout(rows: int, cols: int, wall_prob: float, n_targets: int, algorithms: List[str], max_attempts: int = 300) -> Tuple[Grid, Point, List[Point]]:
    attempts = 0
    algos = [a.lower() for a in algorithms]
    while attempts < max_attempts:
        attempts += 1
        grid = generate_grid(rows, cols, wall_prob)
        start = Point(0, 0)
        grid[start.y][start.x] = 0
        targets: List[Point] = []
        exclude = {(start.x, start.y)}
        ok = True
        for _ in range(n_targets):
            p = random_free_cell(rows, cols, grid, list(exclude))
            if not p:
                ok = False
                break
            targets.append(p)
            # ensure target is walkable
            grid[p.y][p.x] = 0
            exclude.add((p.x, p.y))
        if not ok:
            continue
        all_ok = True
        for algo in algos:
            segs, ordered = build_nearest_chain(start, targets, rows, cols, grid, algo, None)
            if not segs or len(segs) != len(targets):
                all_ok = False
                break
        if all_ok:
            return grid, start, targets
    # fallback: lower walls once then return empty if fail
    if wall_prob > 0.15:
        return find_common_solvable_layout(rows, cols, max(0.15, wall_prob - 0.05), n_targets, algos, max_attempts)
    empty = [[0 for _ in range(cols)] for _ in range(rows)]
    return empty, Point(0, 0), []


 


class State:
    def __init__(self) -> None:
        self.level = 1
        self.rows = 20
        self.cols = 20
        self.wall_prob = 0.25
        self.grid: Grid = []
        self.start = Point(0, 0)
        self.targets: List[Point] = []
        self.ordered_targets: List[Point] = []
        self.segments: List[List[Point]] = []
        self.algorithm: str = "astar"
        self.segment_lengths: List[int] = []
        self.total_length: int = 0

    def to_payload(self) -> Dict:
        return {
            "level": self.level,
            "rows": self.rows,
            "cols": self.cols,
            "wallProb": self.wall_prob,
            "grid": self.grid,
            "start": {"x": self.start.x, "y": self.start.y},
            "targets": [{"x": t.x, "y": t.y} for t in self.targets],
            "orderedTargets": [{"x": t.x, "y": t.y} for t in self.ordered_targets],
            "segments": [[{"x": p.x, "y": p.y} for p in seg] for seg in self.segments],
            "solvable": bool(self.segments),
            "algorithm": self.algorithm,
            "segmentLengths": self.segment_lengths,
            "totalLength": self.total_length,
        }


app = Flask(__name__, static_folder=None)
state = State()


@app.route("/")
def root():
    # Serve the existing index.html from workspace root
    return send_from_directory(".", "index.html")


@app.post("/api/compare")
def api_compare():
    # Compare A*, BFS, Dijkstra on the same layout and targets
    n_targets = state.level
    wall_prob = state.wall_prob
    algorithms = ["astar", "bfs", "dijkstra"]
    grid, start_pt, targets = find_common_solvable_layout(state.rows, state.cols, wall_prob, n_targets, algorithms)
    results = []
    for algo in algorithms:
        stats: Dict = {}
        t0 = time.perf_counter()
        segs, ordered = build_nearest_chain(start_pt, targets, state.rows, state.cols, grid, algo, stats)
        t1 = time.perf_counter()
        total_len = sum(len(seg) for seg in segs) if segs else 0
        ok = bool(segs) and len(segs) == len(targets)
        results.append({
            "algorithm": algo,
            "ok": ok,
            "durationMs": round((t1 - t0) * 1000, 3),
            "expanded": stats.get("expanded", 0),
            "totalLength": total_len,
            "segments": [[{"x": p.x, "y": p.y} for p in seg] for seg in (segs or [])],
            "orderedTargets": [{"x": t.x, "y": t.y} for t in (ordered or [])],
        })
    payload = {
        "rows": state.rows,
        "cols": state.cols,
        "grid": grid,
        "start": {"x": start_pt.x, "y": start_pt.y},
        "targets": [{"x": t.x, "y": t.y} for t in targets],
        "results": results,
    }
    return jsonify(payload)


@app.post("/api/start")
def api_start():
    # Reset to level 1, build solvable maze with 1 target
    state.level = 1
    state.wall_prob = 0.25
    algo = (request.json or {}).get("algorithm") if request.is_json else None
    if algo:
        algo = str(algo).lower()
    if algo not in {None, "astar", "bfs", "dijkstra"}:
        return jsonify({"error": "algorithm harus salah satu dari astar, bfs, dijkstra"}), 400
    state.algorithm = algo or state.algorithm or "astar"
    n_targets = 1
    grid, start_pt, targets, segments, ordered = find_solvable_layout(state.rows, state.cols, state.wall_prob, n_targets, state.algorithm)
    state.grid = grid
    state.start = start_pt
    state.targets = targets
    state.segments = segments
    state.ordered_targets = ordered
    state.segment_lengths = [len(seg) for seg in segments]
    state.total_length = sum(state.segment_lengths)
    return jsonify(state.to_payload())


@app.post("/api/next")
def api_next():
    # Increase difficulty: more walls and more targets
    state.level += 1
    state.wall_prob = min(0.5, state.wall_prob + 0.03)
    algo = (request.json or {}).get("algorithm") if request.is_json else None
    if algo:
        algo = str(algo).lower()
    if algo not in {None, "astar", "bfs", "dijkstra"}:
        return jsonify({"error": "algorithm harus salah satu dari astar, bfs, dijkstra"}), 400
    if algo:
        state.algorithm = algo
    n_targets = state.level  # 1,2,3,...
    grid, start_pt, targets, segments, ordered = find_solvable_layout(state.rows, state.cols, state.wall_prob, n_targets, state.algorithm)
    state.grid = grid
    state.start = start_pt
    state.targets = targets
    state.segments = segments
    state.ordered_targets = ordered
    state.segment_lengths = [len(seg) for seg in segments]
    state.total_length = sum(state.segment_lengths)
    return jsonify(state.to_payload())


@app.post("/api/regenerate")
def api_regenerate():
    # Regenerate current level with current difficulty
    algo = (request.json or {}).get("algorithm") if request.is_json else None
    if algo:
        algo = str(algo).lower()
    if algo not in {None, "astar", "bfs", "dijkstra"}:
        return jsonify({"error": "algorithm harus salah satu dari astar, bfs, dijkstra"}), 400
    if algo:
        state.algorithm = algo
    n_targets = state.level
    grid, start_pt, targets, segments, ordered = find_solvable_layout(state.rows, state.cols, state.wall_prob, n_targets, state.algorithm)
    state.grid = grid
    state.start = start_pt
    state.targets = targets
    state.segments = segments
    state.ordered_targets = ordered
    state.segment_lengths = [len(seg) for seg in segments]
    state.total_length = sum(state.segment_lengths)
    return jsonify(state.to_payload())


@app.post("/api/save")
def api_save():
    # Save current payload to runs.jsonl
    payload = state.to_payload()
    payload["timestamp"] = int(time.time())
    try:
        with open("runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Jangan jalankan server Flask jika file ini diimport/dijalankan oleh Streamlit
    import sys
    if "streamlit" not in sys.modules:
        app.run(host="127.0.0.1", port=5000, debug=True)
