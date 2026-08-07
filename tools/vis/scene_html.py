#!/usr/bin/env python3
"""Create an interactive HTML comparison for one HOI4D grasp frame."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from scene_png import (
    DEFAULT_CAD,
    DEFAULT_FRAME,
    DEFAULT_HAND,
    DEFAULT_MANO_ROOT,
    DEFAULT_SEQUENCE,
    load_hand,
    load_obj,
    load_objpose,
    make_fallback_hand,
    transform_object,
    try_make_mano_mesh,
)


DEFAULT_RGB = DEFAULT_SEQUENCE / "align_rgb" / "00074.jpg"


def image_data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def image_src(path: Path, output: Path) -> str:
    try:
        relative = path.resolve().relative_to(output.parent.resolve())
        return relative.as_posix()
    except ValueError:
        return image_data_uri(path)


def mesh_trace(name: str, vertices: np.ndarray, faces: np.ndarray, color: str, opacity: float) -> go.Mesh3d:
    return go.Mesh3d(
        name=name,
        x=vertices[:, 0].tolist(),
        y=vertices[:, 1].tolist(),
        z=vertices[:, 2].tolist(),
        i=faces[:, 0].astype(int).tolist(),
        j=faces[:, 1].astype(int).tolist(),
        k=faces[:, 2].astype(int).tolist(),
        color=color,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.45, diffuse=0.75, roughness=0.55, specular=0.25),
        lightposition=dict(x=0, y=-3, z=3),
        hoverinfo="skip",
    )


def line_trace(name: str, line: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        name=name,
        x=line[:, 0].tolist(),
        y=line[:, 1].tolist(),
        z=line[:, 2].tolist(),
        mode="lines+markers",
        line=dict(color="#16833a", width=10),
        marker=dict(color="#1ea64b", size=4),
        hoverinfo="skip",
    )


def axis_range(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.25
    return (
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    )


def build_figure(sequence: Path, frame: str, cad: Path, hand_path: Path, mano_root: Path) -> tuple[go.Figure, dict]:
    objpose = load_objpose(sequence / "objpose" / f"{frame}.json")
    hand = load_hand(hand_path)
    obj_vertices, obj_faces = load_obj(cad)
    object_vertices = transform_object(obj_vertices, objpose)

    dims = objpose["dimensions"]
    obj_dims = np.array([dims["length"], dims["width"], dims["height"]], dtype=float)
    center = objpose["center"]
    obj_center = np.array([center["x"], center["y"], center["z"]], dtype=float)
    mano_mesh = try_make_mano_mesh(hand, hand_path, mano_root)

    traces = [
        mesh_trace("object CAD", object_vertices, obj_faces, "#c9c7bd", 0.92),
    ]

    if mano_mesh:
        hand_vertices, hand_faces, side = mano_mesh
        hand_vertices = np.asarray(hand_vertices, dtype=float)
        traces.append(mesh_trace(f"{side} MANO hand", hand_vertices, hand_faces, "#159447", 0.92))
        all_points = np.vstack([np.asarray(object_vertices, dtype=float), hand_vertices])
        mode = "mano_mesh"
    else:
        lines = make_fallback_hand(hand, obj_center, obj_dims)
        for idx, line in enumerate(lines):
            traces.append(line_trace(f"fallback finger {idx}", line))
        all_points = np.vstack([object_vertices] + lines)
        side = "fallback"
        mode = "fallback_skeleton"

    x_range, y_range, z_range = axis_range(all_points)
    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(range=x_range, visible=False),
            yaxis=dict(range=y_range, visible=False),
            zaxis=dict(range=z_range, visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.25, y=-1.7, z=0.85), up=dict(x=0, y=0, z=1)),
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    meta = {
        "sequence": str(sequence),
        "frame": frame,
        "rgb": str(sequence / "align_rgb" / f"{int(frame):05d}.jpg"),
        "cad": str(cad),
        "hand_pickle": str(hand_path),
        "hand_render_mode": mode,
        "hand_side": side,
        "objpose": str(sequence / "objpose" / f"{frame}.json"),
    }
    return fig, meta


def write_html(fig: go.Figure, meta: dict, rgb: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stale_plot_json = output.with_name(f"{output.stem}_plot.json")
    if stale_plot_json.exists():
        stale_plot_json.unlink()
    plot_payload = fig.to_plotly_json()
    plot_json_text = json.dumps(plot_payload, cls=PlotlyJSONEncoder, indent=2).replace("</", "<\\/")
    rgb_uri = image_src(rgb, output)
    errors = meta.get("errors") or {}
    error_rows = "\n".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in errors.items()
    )
    error_block = (
        f"""
      <section class="errors">
        <h2>Errors</h2>
        <table>{error_rows}</table>
      </section>
"""
        if error_rows
        else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HOI4D Grasp Comparison</title>
  <script src="https://cdn.plot.ly/plotly-3.3.0.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #222; }}
    .wrap {{ display: grid; grid-template-columns: 30% 70%; height: 100vh; }}
    .panel {{ padding: 16px; box-sizing: border-box; overflow: auto; }}
    .left {{ background: #fff; border-right: 1px solid #ddd; }}
    .right {{ background: #fff; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #ddd; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e1e1e1; padding: 8px 4px; text-align: left; vertical-align: top; }}
    th {{ width: 52%; color: #555; font-weight: 600; }}
    .errors {{ margin-top: 18px; }}
    .plot {{ height: calc(100vh - 40px); }}
    .plot-status {{ padding: 16px; color: #555; }}
    #plot-data {{ display: none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel left">
      <h2>Original RGB frame</h2>
      <img src="{rgb_uri}" alt="original RGB">
{error_block}
    </div>
    <div class="panel right">
      <div id="plot" class="plot"><div class="plot-status">Loading 3D scene...</div></div>
    </div>
  </div>
  <script id="plot-data" type="application/json">
{plot_json_text}
  </script>
  <script>
    try {{
      const figure = JSON.parse(document.getElementById("plot-data").textContent);
      Plotly.newPlot("plot", figure.data, figure.layout, {{scrollZoom: true, responsive: true}});
    }} catch (error) {{
      document.getElementById("plot").innerHTML =
        '<div class="plot-status">Could not render Plotly scene: ' + error + '</div>';
    }}
  </script>
</body>
</html>
"""
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.write_text(html, encoding="utf-8")
    data = tmp_output.read_bytes()
    if not data.startswith(b"<!doctype html>") or not data.rstrip().endswith(b"</html>"):
        raise RuntimeError(f"HTML output failed validation: {output}")
    tmp_output.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_comparison_bottle_frame74.html"))
    args = parser.parse_args()
    fig, meta = build_figure(args.sequence, args.frame, args.cad, args.hand, args.mano_root)
    write_html(fig, meta, args.rgb, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
