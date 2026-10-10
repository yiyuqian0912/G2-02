"""Interactive local web browser for dataset-g1024-l128-binary.

Uses Flask + Pillow + vanilla HTML/CSS/JS. Completely local, zero external CDNs.
Provides:
- Paginated Gallery mode with thumbnails
- Scene Detail mode with a sampled global preview, continuous geometry overlay,
  adaptive view selector, diagnostic metrics, and local view filters.
"""

import argparse
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import threading
from typing import Dict, Any, Optional
import webbrowser

from flask import Flask, jsonify, request, send_file, Response
from PIL import Image
import numpy as np

from rind_dataset.constants import (
    GLOBAL_SIZE,
    OBSTACLE_RECTANGLE,
    OBSTACLE_ELLIPSE,
    OBSTACLE_TRIANGLE,
    OBSTACLE_POLYGON,
    SHAPE_ID_TO_NAME,
)
from rind_dataset.dataset import RINDDataset

_COLOR_STOPS = np.array([
    [45, 55, 72],    # zero response
    [41, 112, 174],  # low response
    [57, 188, 202],  # medium response
    [246, 207, 86],  # high response
    [255, 247, 179], # maximum possible response
], dtype=np.float64)
_STOP_POSITIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_OBSTACLE_COLOR = np.array([229, 62, 62], dtype=np.uint8)


def encode_intensity_png(response: np.ndarray, obstacle: np.ndarray, scale: float) -> bytes:
    """Color the actual additive response; obstacles have their own fixed color."""
    normalized = np.clip(np.asarray(response, dtype=np.float64) / max(scale, 1e-12), 0, 1)
    rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.rint(np.interp(
            normalized, _STOP_POSITIONS, _COLOR_STOPS[:, channel]
        )).astype(np.uint8)
    rgb[obstacle] = _OBSTACLE_COLOR
    img = Image.fromarray(rgb, mode="RGB")
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=False)
    return bio.getvalue()


def create_app(dataset_path: str) -> Flask:
    """Create and configure the Flask web browser application."""
    ds = RINDDataset(dataset_path, return_scene=True)
    app = Flask(__name__)

    @lru_cache(maxsize=2)
    def scene_field(scene_id: int) -> Dict[str, np.ndarray]:
        return ds.get_region(scene_id, 0, 0, ds.global_size)

    def image_plane(scene_id: int, channel: int):
        field = scene_field(scene_id)
        if channel == -1:
            scale = float(np.sum(ds.driver_strengths[scene_id, :int(ds.driver_counts[scene_id])]))
            return field["response"], field["obstacle"], scale
        return (field["channels"][channel], field["obstacle"],
                float(ds.driver_strengths[scene_id, channel]))

    def requested_channel(scene_id: int):
        channel = request.args.get("channel", "total")
        if channel == "total":
            return -1
        try:
            index = int(channel)
        except ValueError:
            return None
        return index if 0 <= index < int(ds.driver_counts[scene_id]) else None

    @lru_cache(maxsize=8)
    def get_global_png_bytes(scene_id: int, channel: int) -> bytes:
        response, obstacle, scale = image_plane(scene_id, channel)
        return encode_intensity_png(response, obstacle, scale)

    @lru_cache(maxsize=64)
    def get_thumbnail_png_bytes(scene_id: int) -> bytes:
        # 128x128 nearest-neighbor subsample for fast thumbnails
        response, obstacle, scale = image_plane(scene_id, -1)
        stride = max(1, ds.global_size // 128)
        return encode_intensity_png(response[::stride, ::stride], obstacle[::stride, ::stride], scale)

    @lru_cache(maxsize=128)
    def get_view_png_bytes(scene_id: int, view_id: int, channel: int) -> bytes:
        x0, y0, side = map(int, ds.local_views[scene_id, view_id])
        response, obstacle, scale = image_plane(scene_id, channel)
        crop = np.s_[y0 : y0 + side, x0 : x0 + side]
        return encode_intensity_png(response[crop], obstacle[crop], scale)

    # -----------------------------------------------------------------------
    # API Endpoints
    # -----------------------------------------------------------------------

    @app.route("/api/manifest")
    def api_manifest():
        return jsonify(ds.manifest)

    @app.route("/api/scenes")
    def api_scenes():
        page = request.args.get("page", 0, type=int)
        page_size = request.args.get("page_size", 16, type=int)
        start = page * page_size
        end = min(start + page_size, ds.num_scenes)

        items = []
        for s in range(start, end):
            count = int(ds.obstacle_counts[s])
            items.append({
                "scene_id": s,
                "obstacle_count": count,
            })

        return jsonify({
            "page": page,
            "page_size": page_size,
            "total_scenes": ds.num_scenes,
            "total_pages": math.ceil(ds.num_scenes / max(page_size, 1)),
            "scenes": items,
        })

    @app.route("/api/scene/<int:scene_id>")
    def api_scene_meta(scene_id: int):
        if scene_id < 0 or scene_id >= ds.num_scenes:
            return jsonify({"error": f"Invalid scene_id {scene_id}"}), 404

        scene = ds.get_scene(scene_id)
        count = scene["obstacle_count"]
        drivers = scene["drivers"].tolist()
        driver = drivers[0]
        driver_count = int(scene["driver_count"])

        # Obstacles formatting
        obstacles = []
        family_counts = {"rectangle": 0, "ellipse": 0, "triangle": 0, "polygon": 0}
        types = scene["obstacle_types"]
        params = scene["obstacle_params"]
        for i in range(count):
            stype = int(types[i])
            sname = SHAPE_ID_TO_NAME.get(stype, "rectangle")
            family_counts[sname] = family_counts.get(sname, 0) + 1
            p = params[i].tolist()
            obs_dict: Dict[str, Any] = {
                "type": sname,
                "type_id": stype,
                "params": p,
            }
            if stype == OBSTACLE_RECTANGLE:
                obs_dict.update({
                    "cx": p[0], "cy": p[1], "hw": p[2], "hh": p[3], "theta": p[4],
                    "width": p[2] * 2, "height": p[3] * 2,
                })
            elif stype == OBSTACLE_ELLIPSE:
                obs_dict.update({
                    "cx": p[0], "cy": p[1], "rx": p[2], "ry": p[3], "theta": p[4],
                    "diam_x": p[2] * 2, "diam_y": p[3] * 2,
                })
            elif stype == OBSTACLE_TRIANGLE:
                obs_dict.update({
                    "v1": [p[0], p[1]], "v2": [p[2], p[3]], "v3": [p[4], p[5]],
                })
            elif stype == OBSTACLE_POLYGON:
                n_verts = int(p[0])
                verts = [[p[1 + 2 * j], p[2 + 2 * j]] for j in range(n_verts)]
                obs_dict.update({
                    "n_vertices": n_verts,
                    "vertices": verts,
                })
            obstacles.append(obs_dict)
        field = scene_field(scene_id)
        response = field["response"]
        obstacle = field["obstacle"]
        free = ~obstacle
        positive = (response > 0) & free
        c0 = int(np.count_nonzero((response == 0) & free))
        c1 = int(np.count_nonzero(positive))
        c2 = int(np.count_nonzero(obstacle))
        total_cells = ds.global_size * ds.global_size
        response_scale = float(np.sum(scene["driver_strengths"]))
        response_max = float(np.max(response, initial=0))
        response_mean = float(np.mean(response[free])) if np.any(free) else 0.0

        # Local views analysis
        origins = scene["local_views"].tolist()
        view_diagnostics = []
        for v in range(len(origins)):
            x0, y0 = origins[v][:2]
            side = origins[v][2]
            area = side * side
            crop = response[y0 : y0 + side, x0 : x0 + side]
            crop_obstacle = obstacle[y0 : y0 + side, x0 : x0 + side]
            crop_free = ~crop_obstacle
            lc0 = int(np.count_nonzero((crop == 0) & crop_free))
            lc1 = int(np.count_nonzero((crop > 0) & crop_free))
            lc2 = int(np.count_nonzero(crop_obstacle))
            has_obs = lc2 > 0

            drivers_in_view = [d for d in drivers if x0 <= d[0] < x0 + side and y0 <= d[1] < y0 + side]
            num_drivers_in_view = len(drivers_in_view)
            has_driver = num_drivers_in_view > 0

            is_mixed = (lc0 > 0 and lc1 > 0)
            is_all_resp = (lc1 == area)
            is_no_resp = (lc1 == 0)

            view_diagnostics.append({
                "view_id": v,
                "origin": [x0, y0],
                "size": side,
                "fraction_0": round(lc0 / area, 4),
                "fraction_1": round(lc1 / area, 4),
                "fraction_2": round(lc2 / area, 4),
                "response_max": float(np.max(crop, initial=0)),
                "response_mean": float(np.mean(crop[crop_free])) if np.any(crop_free) else 0.0,
                "has_obstacle": bool(has_obs),
                "has_driver": bool(has_driver),
                "num_drivers_in_view": num_drivers_in_view,
                "is_mixed": bool(is_mixed),
                "is_all_response": bool(is_all_resp),
                "is_no_response": bool(is_no_resp),
            })

        return jsonify({
            "scene_id": scene_id,
            "global_size": ds.global_size,
            "driver": driver,
            "drivers": drivers,
            "driver_strengths": scene["driver_strengths"].tolist(),
            "driver_count": driver_count,
            "obstacle_count": count,
            "family_counts": family_counts,
            "obstacles": obstacles,
            "global_fractions": {
                "state_0": round(c0 / total_cells, 4),
                "state_1": round(c1 / total_cells, 4),
                "state_2": round(c2 / total_cells, 4),
            },
            "response_scale": response_scale,
            "response_max": response_max,
            "response_mean": response_mean,
            "local_views": origins,
            "view_diagnostics": view_diagnostics,
        })

    @app.route("/api/scene/<int:scene_id>/global.png")
    def api_global_png(scene_id: int):
        if scene_id < 0 or scene_id >= ds.num_scenes:
            return Response("Scene not found", status=404)
        channel = requested_channel(scene_id)
        if channel is None:
            return Response("Invalid channel", status=400)
        data = get_global_png_bytes(scene_id, channel)
        return Response(data, mimetype="image/png")

    @app.route("/api/scene/<int:scene_id>/thumbnail.png")
    def api_thumbnail_png(scene_id: int):
        if scene_id < 0 or scene_id >= ds.num_scenes:
            return Response("Scene not found", status=404)
        data = get_thumbnail_png_bytes(scene_id)
        return Response(data, mimetype="image/png")

    @app.route("/api/scene/<int:scene_id>/view/<int:view_id>.png")
    def api_view_png(scene_id: int, view_id: int):
        count = int(ds.view_counts[scene_id]) if 0 <= scene_id < ds.num_scenes else 0
        if scene_id < 0 or scene_id >= ds.num_scenes or view_id < 0 or view_id >= count:
            return Response("View not found", status=404)
        channel = requested_channel(scene_id)
        if channel is None:
            return Response("Invalid channel", status=400)
        data = get_view_png_bytes(scene_id, view_id, channel)
        return Response(data, mimetype="image/png")

    @app.route("/api/scene/<int:scene_id>/view/<int:view_id>/candidates")
    def api_view_candidates(scene_id: int, view_id: int):
        count = int(ds.view_counts[scene_id]) if 0 <= scene_id < ds.num_scenes else 0
        if view_id < 0 or view_id >= count:
            return jsonify({"error": "View not found"}), 404
        return jsonify({
            "scene_id": scene_id,
            "view_id": view_id,
            "view": ds.local_views[scene_id, view_id].tolist(),
            "equivalence": ("sampled-view" if ds.candidate_layout == "single-position-list"
                            else "continuous-field"),
            "candidates": [candidate.tolist() for candidate in ds.get_candidates(scene_id, view_id)],
        })

    @app.route("/api/scene/<int:scene_id>/pixel")
    def api_pixel(scene_id: int):
        if scene_id < 0 or scene_id >= ds.num_scenes:
            return jsonify({"error": "Scene not found"}), 404
        x = request.args.get("x", type=int)
        y = request.args.get("y", type=int)
        if x is None or y is None or not (0 <= x < ds.global_size and 0 <= y < ds.global_size):
            return jsonify({"error": "Pixel outside scene"}), 400
        field = scene_field(scene_id)
        return jsonify({
            "x": x, "y": y,
            "obstacle": bool(field["obstacle"][y, x]),
            "response": float(field["response"][y, x]),
            "channels": field["channels"][:, y, x].tolist(),
        })

    # -----------------------------------------------------------------------
    # Embedded HTML / CSS / JS Single Page App
    # -----------------------------------------------------------------------

    @app.route("/")
    def index():
        variant = "Single-source" if ds.manifest["max_drivers"] == 1 else "Multi-source"
        return (INDEX_HTML.replace("RIND Dataset Browser", f"RIND {variant} Browser")
                .replace("dataset-g1024-l128-binary", ds.root.name))

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RIND Dataset Browser (dataset-g1024-l128-binary)</title>
<style>
  :root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --card-hover: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --border: #334155;
    --color-0: #2d3748;
    --color-1: #f6e05e;
    --color-2: #e53e3e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* Navbar */
  header {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  .brand { display: flex; align-items: center; gap: 12px; font-weight: bold; font-size: 16px; color: var(--accent); }
  .nav-btn {
    background: #0f172a; border: 1px solid var(--border); color: var(--text);
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500;
  }
  .nav-btn:hover { background: var(--border); }
  .nav-btn.active { background: var(--accent); color: #000; border-color: var(--accent); }

  /* Main Container */
  main { flex: 1; overflow: hidden; display: flex; position: relative; }
  .view-container { width: 100%; height: 100%; display: none; overflow-y: auto; }
  .view-container.active { display: flex; }

  /* Gallery Mode */
  #gallery-view { flex-direction: column; padding: 20px; gap: 16px; }
  .gallery-controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .gallery-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px; overflow-y: auto; padding-bottom: 30px;
  }
  .thumb-card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; cursor: pointer; display: flex; flex-direction: column; gap: 8px;
    transition: transform 0.1s, border-color 0.1s;
  }
  .thumb-card:hover { transform: translateY(-2px); border-color: var(--accent); background: var(--card-hover); }
  .thumb-img {
    width: 100%; aspect-ratio: 1/1; image-rendering: pixelated; border-radius: 4px;
    background: #000; object-fit: contain;
  }
  .thumb-info { font-size: 12px; color: var(--text-muted); display: flex; justify-content: space-between; }

  /* Detail Mode */
  #detail-view { display: none; height: 100%; width: 100%; }
  #detail-view.active { display: flex; }
  .detail-main {
    flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 16px; position: relative; background: #070d19;
  }
  .canvas-wrapper {
    position: relative; width: min(78vh, 78vw); height: min(78vh, 78vw);
    border: 2px solid var(--border); border-radius: 4px; overflow: hidden; background: #000;
  }
  .global-img { width: 100%; height: 100%; image-rendering: pixelated; position: absolute; left: 0; top: 0; }
  .overlay-svg { width: 100%; height: 100%; position: absolute; left: 0; top: 0; pointer-events: none; }

  /* Detail Sidebar */
  .detail-sidebar {
    width: 380px; background: var(--card-bg); border-left: 1px solid var(--border);
    display: flex; flex-direction: column; overflow-y: auto; padding: 16px; gap: 16px; flex-shrink: 0;
  }
  .section-title { font-size: 13px; font-weight: 700; text-transform: uppercase; color: var(--accent); letter-spacing: 0.5px; }
  .panel-box {
    background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px; font-size: 12px;
  }
  .stat-row { display: flex; justify-content: space-between; color: var(--text-muted); }
  .stat-val { color: var(--text); font-weight: 600; }

  /* Local Crop Box */
  .local-crop-container { display: flex; gap: 12px; align-items: center; }
  .local-crop-img {
    width: 128px; height: 128px; border: 2px solid var(--accent);
    image-rendering: pixelated; background: #000; flex-shrink: 0; border-radius: 4px;
  }

  /* Form Controls */
  .btn-group { display: flex; gap: 6px; }
  button {
    background: var(--card-bg); border: 1px solid var(--border); color: var(--text);
    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 500;
  }
  button:hover { background: var(--border); }
  input[type="number"], select {
    background: #0f172a; border: 1px solid var(--border); color: var(--text);
    padding: 5px 8px; border-radius: 4px; font-size: 12px; outline: none;
  }
  input[type="range"] { accent-color: var(--accent); }

  /* Filter tags */
  .filter-grid { display: flex; flex-wrap: wrap; gap: 4px; max-height: 120px; overflow-y: auto; padding: 4px 0; }
  .filter-chip {
    font-size: 11px; padding: 3px 8px; border-radius: 4px; background: #1e293b;
    border: 1px solid var(--border); cursor: pointer; color: var(--text-muted);
  }
  .filter-chip:hover { color: var(--text); border-color: var(--accent); }
  .filter-chip.active { background: var(--accent); color: #000; font-weight: 600; }
  .view-badge {
    font-size: 10px; padding: 2px 6px; border-radius: 3px; background: #334155;
    cursor: pointer; display: inline-block; color: var(--text);
  }
  .view-badge.selected { background: var(--accent); color: #000; font-weight: bold; }

  /* Legend */
  .legend-bar { display: flex; gap: 12px; font-size: 11px; justify-content: center; align-items: center; margin-top: 8px; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .color-dot { width: 10px; height: 10px; border-radius: 2px; }
  .gradient-key { width: 180px; height: 10px; border-radius: 2px; background: linear-gradient(90deg, #2d3748, #2970ae 25%, #39bcca 50%, #f6cf56 75%, #fff7b3); }
  .pixel-inspector { color: var(--text-muted); font-size: 11px; text-align: center; margin-top: 7px; max-width: 80%; }
  @media (max-width: 900px) {
    #detail-view.active { flex-direction: column; overflow-y: auto; }
    .detail-main { flex: none; }
    .canvas-wrapper { width: min(76vw, 60vh); height: min(76vw, 60vh); }
    .detail-sidebar { width: 100%; flex: none; overflow-y: visible; border-left: 0; border-top: 1px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/>
    </svg>
    RIND Dataset Browser
  </div>
  <div class="btn-group">
    <button id="btn-mode-gallery" class="nav-btn active" onclick="setMode('gallery')">Gallery</button>
    <button id="btn-mode-detail" class="nav-btn" onclick="setMode('detail')">Scene Detail</button>
  </div>
</header>

<main>
  <!-- Gallery View -->
  <div id="gallery-view" class="view-container active">
    <div class="gallery-controls">
      <div class="btn-group">
        <button onclick="changePage(-1)">‹ Prev Page</button>
        <button onclick="changePage(1)">Next Page ›</button>
        <button onclick="randomScene()">🎲 Random Scene</button>
      </div>
      <span id="gallery-page-info" style="font-size: 13px; color: var(--text-muted);">Page 1 / 1</span>
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="font-size:12px; color:var(--text-muted);">Jump to Scene:</span>
        <input type="number" id="input-jump-scene" min="0" style="width: 70px;" onkeydown="if(event.key==='Enter') jumpScene()"/>
        <button onclick="jumpScene()">Go</button>
      </div>
    </div>
    <div id="gallery-grid" class="gallery-grid"></div>
  </div>

  <!-- Detail View -->
  <div id="detail-view" class="view-container">
    <div class="detail-main">
      <div class="canvas-wrapper" onclick="inspectPixel(event, 'global')" title="Click to inspect the exact stored response">
        <img id="global-raster-img" class="global-img" alt="Global Raster"/>
        <svg id="global-overlay-svg" class="overlay-svg" viewBox="0 0 1024 1024"></svg>
      </div>
      <div class="legend-bar">
        <span>0</span><div class="gradient-key" title="Linear response intensity scale"></div><span id="legend-max">1</span>
        <div class="legend-item"><div class="color-dot" style="background:#e53e3e;"></div>Obstacle</div>
      </div>
      <div id="pixel-inspector" class="pixel-inspector">Click the scene or crop to inspect exact response and source contributions.</div>
    </div>

    <div class="detail-sidebar">
      <!-- Scene Navigation -->
      <div class="panel-box">
        <div class="section-title">Scene Selector</div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <button onclick="changeScene(-1)">‹ Prev</button>
          <div style="display:flex; align-items:center; gap:6px;">
            <span>Scene #</span>
            <input type="number" id="detail-scene-input" min="0" style="width:70px;" onchange="loadScene(parseInt(this.value))"/>
          </div>
          <button onclick="changeScene(1)">Next ›</button>
        </div>
        <button onclick="randomSceneDetail()">🎲 Random Scene</button>
      </div>

      <!-- Overlays & Toggles -->
      <div class="panel-box">
        <div class="section-title">Display Toggles</div>
        <label style="display:flex; align-items:center; gap:8px;">Response layer <select id="response-layer" onchange="setResponseLayer(this.value)" style="flex:1;"><option value="total">Total response</option></select></label>
        <label style="display:flex; gap:8px; cursor:pointer;"><input type="checkbox" id="toggle-geo" checked onchange="renderOverlays()"/> Continuous Geometry</label>
        <label style="display:flex; gap:8px; cursor:pointer;"><input type="checkbox" id="toggle-sel" checked onchange="renderOverlays()"/> Selected Window</label>
        <label style="display:flex; gap:8px; cursor:pointer;"><input type="checkbox" id="toggle-all" onchange="renderOverlays()"/> All Windows</label>
        <label style="display:flex; gap:8px; cursor:pointer;"><input type="checkbox" id="toggle-driver" checked onchange="renderOverlays()"/> Driver Position</label>
      </div>

      <!-- Local View Selector -->
      <div class="panel-box">
        <div class="section-title">Selected View (Crop)</div>
        <div class="local-crop-container">
          <img id="local-crop-img" class="local-crop-img" alt="Local View Crop" onclick="inspectPixel(event, 'local')" title="Click to inspect the exact stored response"/>
          <div style="flex:1; display:flex; flex-direction:column; gap:6px;">
            <div class="stat-row"><span>View ID:</span><span id="lbl-view-id" class="stat-val">0</span></div>
            <div class="stat-row"><span>Origin [x0, y0]:</span><span id="lbl-view-origin" class="stat-val">[0, 0]</span></div>
            <div class="stat-row"><span>Has Obstacle:</span><span id="lbl-view-obs" class="stat-val">-</span></div>
            <div class="stat-row"><span>Has Driver:</span><span id="lbl-view-driver" class="stat-val">-</span></div>
            <div class="stat-row"><span>Max total response:</span><span id="lbl-view-max" class="stat-val">-</span></div>
            <div class="stat-row"><span>Mean total response:</span><span id="lbl-view-mean" class="stat-val">-</span></div>
          </div>
        </div>

        <div style="display:flex; align-items:center; gap:6px; margin-top:8px;">
          <button onclick="changeView(-1)">‹ Prev View</button>
          <input type="range" id="slider-view" min="0" max="127" value="0" style="flex:1;" oninput="selectView(parseInt(this.value))"/>
          <button onclick="changeView(1)">Next View ›</button>
        </div>
      </div>

      <!-- Diagnostic Filter for Views -->
      <div class="panel-box">
        <div class="section-title">Diagnostic Filter</div>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">
          <div class="filter-chip active" id="f-all" onclick="setFilter('all')">All</div>
          <div class="filter-chip" id="f-obs" onclick="setFilter('obs')">With Obstacle</div>
          <div class="filter-chip" id="f-mixed" onclick="setFilter('mixed')">Mixed Response</div>
          <div class="filter-chip" id="f-allresp" onclick="setFilter('allresp')">All Response</div>
          <div class="filter-chip" id="f-noresp" onclick="setFilter('noresp')">No Response</div>
          <div class="filter-chip" id="f-driver" onclick="setFilter('driver')">With Driver</div>
        </div>
        <div id="filter-results-container" class="filter-grid"></div>
      </div>

      <!-- Scene Diagnostics -->
      <div class="panel-box">
        <div class="section-title">Scene Diagnostics</div>
        <div class="stat-row"><span>Driver:</span><span id="diag-driver" class="stat-val">-</span></div>
        <div class="stat-row"><span>Obstacle Count:</span><span id="diag-obs-count" class="stat-val">-</span></div>
        <div class="stat-row"><span>Family Breakdown:</span><span id="diag-families" class="stat-val">-</span></div>
        <div class="stat-row"><span>Zero response:</span><span id="diag-s0" class="stat-val">-</span></div>
        <div class="stat-row"><span>Positive response:</span><span id="diag-s1" class="stat-val">-</span></div>
        <div class="stat-row"><span>Obstacle:</span><span id="diag-s2" class="stat-val">-</span></div>
        <div class="stat-row"><span>Max total response:</span><span id="diag-rmax" class="stat-val">-</span></div>
        <div class="stat-row"><span>Mean total response (free):</span><span id="diag-rmean" class="stat-val">-</span></div>
      </div>
    </div>
  </div>
</main>

<script>
let state = {
  mode: 'gallery',
  currentPage: 0,
  pageSize: 16,
  totalScenes: 0,
  currentSceneId: 0,
  currentViewId: 0,
  sceneData: null,
  activeFilter: 'all',
  responseLayer: 'total',
};

// Initialize
fetch('/api/scenes?page=0&page_size=16')
  .then(r => r.json())
  .then(data => {
    state.totalScenes = data.total_scenes;
    loadGalleryPage(0);
  });

function setMode(mode) {
  state.mode = mode;
  document.getElementById('btn-mode-gallery').classList.toggle('active', mode === 'gallery');
  document.getElementById('btn-mode-detail').classList.toggle('active', mode === 'detail');
  document.getElementById('gallery-view').classList.toggle('active', mode === 'gallery');
  document.getElementById('detail-view').classList.toggle('active', mode === 'detail');

  if (mode === 'detail' && !state.sceneData) {
    loadScene(state.currentSceneId);
  }
}

// ---------------------------------------------------------------------------
// Gallery Functions
// ---------------------------------------------------------------------------
function loadGalleryPage(page) {
  state.currentPage = page;
  fetch(`/api/scenes?page=${page}&page_size=${state.pageSize}`)
    .then(r => r.json())
    .then(data => {
      document.getElementById('gallery-page-info').innerText =
        `Page ${data.page + 1} / ${data.total_pages} (${data.total_scenes} total scenes)`;
      const grid = document.getElementById('gallery-grid');
      grid.innerHTML = '';
      data.scenes.forEach(s => {
        const card = document.createElement('div');
        card.className = 'thumb-card';
        card.onclick = () => {
          loadScene(s.scene_id);
          setMode('detail');
        };
        card.innerHTML = `
          <img class="thumb-img" src="/api/scene/${s.scene_id}/thumbnail.png" loading="lazy" alt="Scene ${s.scene_id}"/>
          <div class="thumb-info">
            <span style="font-weight:600; color:var(--text);">Scene ${s.scene_id}</span>
            <span>${s.obstacle_count} obs</span>
          </div>
        `;
        grid.appendChild(card);
      });
    });
}

function changePage(delta) {
  const maxPages = Math.ceil(state.totalScenes / state.pageSize);
  let next = state.currentPage + delta;
  if (next >= 0 && next < maxPages) {
    loadGalleryPage(next);
  }
}

function jumpScene() {
  const val = parseInt(document.getElementById('input-jump-scene').value);
  if (!isNaN(val) && val >= 0 && val < state.totalScenes) {
    loadScene(val);
    setMode('detail');
  }
}

function randomScene() {
  const r = Math.floor(Math.random() * state.totalScenes);
  loadScene(r);
  setMode('detail');
}

// ---------------------------------------------------------------------------
// Detail Scene Functions
// ---------------------------------------------------------------------------
function loadScene(sceneId) {
  if (sceneId < 0 || sceneId >= state.totalScenes) return;
  state.currentSceneId = sceneId;
  state.responseLayer = 'total';
  state.sceneData = null;
  document.getElementById('detail-scene-input').value = sceneId;
  document.getElementById('pixel-inspector').innerText = 'Click the scene or crop to inspect exact response and source contributions.';

  // Set global raster image
  document.getElementById('global-raster-img').src = `/api/scene/${sceneId}/global.png`;

  fetch(`/api/scene/${sceneId}`)
    .then(r => r.json())
    .then(data => {
      if (state.currentSceneId !== sceneId) return;
      state.sceneData = data;
      document.getElementById('global-overlay-svg').setAttribute('viewBox', `0 0 ${data.global_size} ${data.global_size}`);
      const layer = document.getElementById('response-layer');
      layer.innerHTML = '<option value="total">Total response</option>' +
        data.drivers.map((_, i) => `<option value="${i}">Driver D${i}</option>`).join('');
      layer.value = 'total';
      updateLegend();
      // Populate diagnostics
      const drvs = data.drivers || [data.driver];
      const amps = data.driver_strengths;
      if (drvs.length === 1) {
        const amp = amps ? `, intensity ${amps[0].toFixed(3)}` : '';
        document.getElementById('diag-driver').innerText = `(${drvs[0][0].toFixed(1)}, ${drvs[0][1].toFixed(1)})${amp}`;
      } else {
        const drvStrs = drvs.map((d, i) => `D${i}: (${d[0].toFixed(1)}, ${d[1].toFixed(1)})${amps ? ` @${amps[i].toFixed(3)}` : ''}`).join(' ');
        document.getElementById('diag-driver').innerText = `${drvs.length} drivers: ${drvStrs}`;
      }
      document.getElementById('diag-obs-count').innerText = `${data.obstacle_count}`;
      const f = data.family_counts;
      document.getElementById('diag-families').innerText = `R:${f.rectangle || 0} E:${f.ellipse || 0} T:${f.triangle || 0} P:${f.polygon || 0}`;
      document.getElementById('diag-s0').innerText = `${(data.global_fractions.state_0 * 100).toFixed(1)}%`;
      document.getElementById('diag-s1').innerText = `${(data.global_fractions.state_1 * 100).toFixed(1)}%`;
      document.getElementById('diag-s2').innerText = `${(data.global_fractions.state_2 * 100).toFixed(1)}%`;
      document.getElementById('diag-rmax').innerText = formatIntensity(data.response_max);
      document.getElementById('diag-rmean').innerText = formatIntensity(data.response_mean);

      const count = data.view_diagnostics.length;
      document.getElementById('slider-view').max = count - 1;
      document.getElementById('f-all').innerText = `All (${count})`;
      state.currentViewId = Math.min(state.currentViewId, count - 1);
      selectView(state.currentViewId);
      applyFilter();
      renderOverlays();
    });
}

function formatIntensity(value) {
  return Number(value).toFixed(4);
}

function updateLegend() {
  if (!state.sceneData) return;
  const layer = state.responseLayer;
  const scale = layer === 'total' ? state.sceneData.response_scale : state.sceneData.driver_strengths[Number(layer)];
  document.getElementById('legend-max').innerText = formatIntensity(scale);
}

function setResponseLayer(layer) {
  if (!state.sceneData) return;
  state.responseLayer = layer;
  document.getElementById('global-raster-img').src = `/api/scene/${state.currentSceneId}/global.png?channel=${layer}`;
  document.getElementById('local-crop-img').src = `/api/scene/${state.currentSceneId}/view/${state.currentViewId}.png?channel=${layer}`;
  updateLegend();
}

function inspectPixel(event, surface) {
  if (!state.sceneData) return;
  const img = document.getElementById(surface === 'global' ? 'global-raster-img' : 'local-crop-img');
  const rect = img.getBoundingClientRect();
  const size = surface === 'global' ? state.sceneData.global_size : state.sceneData.view_diagnostics[state.currentViewId].size;
  const origin = surface === 'global' ? [0, 0] : state.sceneData.view_diagnostics[state.currentViewId].origin;
  const x = origin[0] + Math.min(size - 1, Math.max(0, Math.floor((event.clientX - rect.left) * size / rect.width)));
  const y = origin[1] + Math.min(size - 1, Math.max(0, Math.floor((event.clientY - rect.top) * size / rect.height)));
  const sceneId = state.currentSceneId;
  fetch(`/api/scene/${sceneId}/pixel?x=${x}&y=${y}`)
    .then(r => r.json())
    .then(pixel => {
      if (sceneId !== state.currentSceneId) return;
      const pieces = pixel.channels.map((v, i) => `D${i}: ${v.toString()}`).join(' + ');
      document.getElementById('pixel-inspector').innerText =
        `(${x}, ${y}) ${pixel.obstacle ? 'Obstacle · ' : ''}Total: ${pixel.response.toString()} = ${pieces}`;
    });
}

function changeScene(delta) {
  let next = state.currentSceneId + delta;
  if (next >= 0 && next < state.totalScenes) {
    loadScene(next);
  }
}

function randomSceneDetail() {
  const r = Math.floor(Math.random() * state.totalScenes);
  loadScene(r);
}

// ---------------------------------------------------------------------------
// View Selection & Diagnostic Filter
// ---------------------------------------------------------------------------
function selectView(viewId) {
  state.currentViewId = viewId;
  document.getElementById('slider-view').value = viewId;
  document.getElementById('lbl-view-id').innerText = `${viewId}`;
  document.getElementById('local-crop-img').src = `/api/scene/${state.currentSceneId}/view/${viewId}.png?channel=${state.responseLayer}`;

  if (state.sceneData && state.sceneData.view_diagnostics) {
    const vd = state.sceneData.view_diagnostics[viewId];
    document.getElementById('lbl-view-origin').innerText = `[${vd.origin[0]}, ${vd.origin[1]}], ${vd.size}×${vd.size}`;
    document.getElementById('lbl-view-obs').innerText = vd.has_obstacle ? 'Yes' : 'No';
    document.getElementById('lbl-view-driver').innerText = vd.has_driver ? (vd.num_drivers_in_view ? `${vd.num_drivers_in_view} driver(s)` : 'Yes') : 'No';
    document.getElementById('lbl-view-max').innerText = formatIntensity(vd.response_max);
    document.getElementById('lbl-view-mean').innerText = formatIntensity(vd.response_mean);
  }

  // Update badge highlight in filter list
  document.querySelectorAll('.view-badge').forEach(el => {
    el.classList.toggle('selected', parseInt(el.dataset.id) === viewId);
  });

  renderOverlays();
}

function changeView(delta) {
  let next = state.currentViewId + delta;
  if (next >= 0 && state.sceneData && next < state.sceneData.view_diagnostics.length) {
    selectView(next);
  }
}

function setFilter(filt) {
  state.activeFilter = filt;
  ['all', 'obs', 'mixed', 'allresp', 'noresp', 'driver'].forEach(id => {
    document.getElementById('f-' + id).classList.toggle('active', id === filt);
  });
  applyFilter();
}

function applyFilter() {
  if (!state.sceneData) return;
  const container = document.getElementById('filter-results-container');
  container.innerHTML = '';
  const list = state.sceneData.view_diagnostics;

  let matching = [];
  list.forEach(v => {
    let match = false;
    if (state.activeFilter === 'all') match = true;
    else if (state.activeFilter === 'obs' && v.has_obstacle) match = true;
    else if (state.activeFilter === 'mixed' && v.is_mixed) match = true;
    else if (state.activeFilter === 'allresp' && v.is_all_response) match = true;
    else if (state.activeFilter === 'noresp' && v.is_no_response) match = true;
    else if (state.activeFilter === 'driver' && v.has_driver) match = true;
    if (match) matching.push(v.view_id);
  });

  matching.forEach(vid => {
    const b = document.createElement('span');
    b.className = 'view-badge' + (vid === state.currentViewId ? ' selected' : '');
    b.dataset.id = vid;
    b.innerText = `${vid}`;
    b.onclick = () => selectView(vid);
    container.appendChild(b);
  });
}

// ---------------------------------------------------------------------------
// SVG Overlays: Continuous Geometry, Windows, Driver
// ---------------------------------------------------------------------------
function renderOverlays() {
  const svg = document.getElementById('global-overlay-svg');
  svg.innerHTML = '';
  if (!state.sceneData) return;

  const showGeo = document.getElementById('toggle-geo').checked;
  const showSel = document.getElementById('toggle-sel').checked;
  const showAll = document.getElementById('toggle-all').checked;
  const showDriver = document.getElementById('toggle-driver').checked;

  // 1. All adaptive views
  if (showAll) {
    state.sceneData.local_views.forEach(orig => {
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', orig[0]);
      rect.setAttribute('y', orig[1]);
      rect.setAttribute('width', orig[2]);
      rect.setAttribute('height', orig[2]);
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', '#ffffff');
      rect.setAttribute('stroke-width', '1');
      rect.setAttribute('opacity', '0.25');
      svg.appendChild(rect);
    });
  }

  // 2. Continuous Geometry
  if (showGeo) {
    state.sceneData.obstacles.forEach(obs => {
      if (obs.type === 'rectangle') {
        const deg = (obs.theta * 180 / Math.PI);
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', -obs.hw);
        rect.setAttribute('y', -obs.hh);
        rect.setAttribute('width', obs.width);
        rect.setAttribute('height', obs.height);
        rect.setAttribute('transform', `translate(${obs.cx}, ${obs.cy}) rotate(${deg})`);
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', '#ff007f');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(rect);
      } else if (obs.type === 'ellipse') {
        const deg = (obs.theta * 180 / Math.PI);
        const ell = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        ell.setAttribute('cx', 0);
        ell.setAttribute('cy', 0);
        ell.setAttribute('rx', obs.rx);
        ell.setAttribute('ry', obs.ry);
        ell.setAttribute('transform', `translate(${obs.cx}, ${obs.cy}) rotate(${deg})`);
        ell.setAttribute('fill', 'none');
        ell.setAttribute('stroke', '#ff007f');
        ell.setAttribute('stroke-width', '2');
        ell.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(ell);
      } else if (obs.type === 'triangle') {
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', `${obs.v1[0]},${obs.v1[1]} ${obs.v2[0]},${obs.v2[1]} ${obs.v3[0]},${obs.v3[1]}`);
        poly.setAttribute('fill', 'none');
        poly.setAttribute('stroke', '#ff007f');
        poly.setAttribute('stroke-width', '2');
        poly.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(poly);
      } else if (obs.type === 'polygon' && obs.vertices) {
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        const pts = obs.vertices.map(v => `${v[0].toFixed(1)},${v[1].toFixed(1)}`).join(' ');
        poly.setAttribute('points', pts);
        poly.setAttribute('fill', 'none');
        poly.setAttribute('stroke', '#ff007f');
        poly.setAttribute('stroke-width', '2');
        poly.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(poly);
      }
    });
  }

  // 3. Selected Window
  if (showSel) {
    const orig = state.sceneData.local_views[state.currentViewId];
    const selRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    selRect.setAttribute('x', orig[0]);
    selRect.setAttribute('y', orig[1]);
    selRect.setAttribute('width', orig[2]);
    selRect.setAttribute('height', orig[2]);
    selRect.setAttribute('fill', 'none');
    selRect.setAttribute('stroke', '#00f5d4');
    selRect.setAttribute('stroke-width', '3');
    svg.appendChild(selRect);
  }

  // 4. Driver Markers (Star shape with D0, D1... labels)
  if (showDriver) {
    const drivers = state.sceneData.drivers || [state.sceneData.driver];
    drivers.forEach((drv, dIdx) => {
      const dx = drv[0];
      const dy = drv[1];

      const star = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      const rOuter = 12, rInner = 5;
      let pts = [];
      for (let i = 0; i < 10; i++) {
        const r = (i % 2 === 0) ? rOuter : rInner;
        const angle = i * Math.PI / 5 - Math.PI / 2;
        pts.push(`${(dx + r * Math.cos(angle)).toFixed(1)},${(dy + r * Math.sin(angle)).toFixed(1)}`);
      }
      star.setAttribute('points', pts.join(' '));
      star.setAttribute('fill', '#00f5d4');
      star.setAttribute('stroke', '#000000');
      star.setAttribute('stroke-width', '1.5');
      svg.appendChild(star);

      const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txt.setAttribute('x', dx);
      txt.setAttribute('y', dy - 14);
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('fill', '#00f5d4');
      txt.setAttribute('font-size', '13');
      txt.setAttribute('font-weight', 'bold');
      txt.setAttribute('stroke', '#000000');
      txt.setAttribute('stroke-width', '0.6');
      txt.textContent = `D${dIdx}`;
      svg.appendChild(txt);
    });
  }
}

// ---------------------------------------------------------------------------
// Keyboard Shortcuts
// ---------------------------------------------------------------------------
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (state.mode === 'detail') {
    if (e.shiftKey && e.key === 'ArrowLeft') { changeScene(-1); e.preventDefault(); }
    else if (e.shiftKey && e.key === 'ArrowRight') { changeScene(1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { changeView(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { changeView(1); e.preventDefault(); }
  } else {
    if (e.key === 'ArrowLeft') { changePage(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { changePage(1); e.preventDefault(); }
  }
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Interactive web dataset browser for dataset-g1024-l128-binary")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open web browser")

    args = parser.parse_args()

    app = create_app(args.dataset)
    url = f"http://{args.host}:{args.port}"
    print(f"\nStarting RIND Dataset Browser on {url} ...")

    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
